"""Synthetic data for testing without a camera.

1. `synth_features`: a simple physical eye/head model that turns a true screen
   point into a noisy feature vector (tests calibration + regression).
2. `simulate_session`: closed-loop reading simulation — a virtual reader reads
   a document, the detector scrolls it, the reader's eyes follow the text.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .filters import OneEuro
from .reading import Params, ReadingDetector, State

# ---------------------------------------------------------------------------
# 1. features
# ---------------------------------------------------------------------------
SCREEN_W_CM, SCREEN_H_CM = 7.0, 15.0


@dataclass
class Subject:
    """Per-person / per-session constants (what calibration must absorb)."""
    gain_u: float = 0.42
    gain_v: float = 0.30
    kappa_x: float = 0.03     # angle kappa offset (rad)
    kappa_y: float = -0.02
    ear0: float = 0.30
    cam_x_cm: float = 3.5     # camera position relative to screen top-left
    cam_y_cm: float = -1.0

    @classmethod
    def random(cls, rng):
        return cls(gain_u=rng.uniform(0.35, 0.5), gain_v=rng.uniform(0.24, 0.36),
                   kappa_x=rng.normal(0, 0.03), kappa_y=rng.normal(0, 0.03),
                   ear0=rng.uniform(0.26, 0.34))


def synth_features(target_cm, head, subj: Subject, rng, noise=1.0) -> np.ndarray:
    """head = (hx_cm, hy_cm, dist_cm, yaw, pitch, roll) of the eyes w.r.t. screen."""
    X, Y = target_cm
    hx, hy, D, yaw, pitch, roll = head
    th_x = np.arctan2(X - hx, D) - yaw + subj.kappa_x
    th_y = np.arctan2(Y - hy, D) - pitch + subj.kappa_y
    # eye-in-head rotation seen by the camera, slightly rotated by roll
    u0 = subj.gain_u * np.sin(th_x)
    v0 = subj.gain_v * np.sin(th_y)
    u = u0 * np.cos(roll) - v0 * np.sin(roll)
    v = u0 * np.sin(roll) + v0 * np.cos(roll)
    ear = subj.ear0 - 0.18 * np.sin(th_y)            # eyelid follows downward gaze
    s = noise
    n = lambda sd: rng.normal(0, sd * s)
    # where the face appears in the image (camera at top centre)
    face_cx = 0.5 + (hx - subj.cam_x_cm) / D * 0.9 + n(0.002)
    face_cy = 0.5 + (hy - subj.cam_y_cm) / D * 0.9 + n(0.002)
    scale = 9.0 / D + n(0.002)
    return np.array([u + n(0.012), v + n(0.010), u + n(0.012), v + n(0.010),
                     ear + n(0.008), ear + n(0.008),
                     yaw + n(0.01), pitch + n(0.01), roll + n(0.005),
                     face_cx, face_cy, scale,
                     face_cx + n(0.001), face_cy - 0.05 + n(0.001)])


def random_head(rng, base=(3.5, 6.0, 30.0)):
    hx, hy, D = base
    return (hx + rng.normal(0, 0.8), hy + rng.normal(0, 1.0), D + rng.normal(0, 2.5),
            rng.normal(0, 0.05), rng.normal(0, 0.05), rng.normal(0, 0.03))


def grid_points(n_side=3, margin=0.1):
    xs = np.linspace(margin, 1 - margin, n_side) * SCREEN_W_CM
    ys = np.linspace(margin, 1 - margin, n_side) * SCREEN_H_CM
    return np.array([(x, y) for y in ys for x in xs])


# ---------------------------------------------------------------------------
# 2. closed-loop reading session
# ---------------------------------------------------------------------------
@dataclass
class NoiseModel:
    sd_x: float = 0.12       # fraction of column width per sample  (~0.8 cm on 7 cm)
    sd_y: float = 0.035      # fraction of viewport height          (~0.5 cm on 15 cm)
    drift_sd: float = 0.004  # random-walk bias per second
    outlier_p: float = 0.02
    dropout_p: float = 0.01  # face not found for one frame


@dataclass
class SessionResult:
    t: np.ndarray
    truth_state: np.ndarray
    pred_state: np.ndarray
    lines_read: int
    scrolls: list = field(default_factory=list)       # (t, kind, segment)
    max_truth_y_reading: float = 0.0
    ran_off_bottom: int = 0

    def state_accuracy(self, settle=2.0) -> float:
        keep = np.ones(len(self.t), bool)
        changes = np.flatnonzero(self.truth_state[1:] != self.truth_state[:-1]) + 1
        for c in changes:
            keep &= ~((self.t >= self.t[c]) & (self.t < self.t[c] + settle))
        keep &= self.t > settle
        return float((self.truth_state[keep] == self.pred_state[keep]).mean())

    def reading_f1(self, settle=2.0) -> float:
        keep = self.t > settle
        changes = np.flatnonzero(self.truth_state[1:] != self.truth_state[:-1]) + 1
        for c in changes:
            keep &= ~((self.t >= self.t[c]) & (self.t < self.t[c] + settle))
        tr = self.truth_state[keep] == State.READING.value
        pr = self.pred_state[keep] == State.READING.value
        tp, fp, fn = (tr & pr).sum(), (~tr & pr).sum(), (tr & ~pr).sum()
        return float(2 * tp / max(2 * tp + fp + fn, 1))

    def false_scrolls(self) -> int:
        return sum(1 for _, _, seg in self.scrolls if seg != State.READING.value)


def simulate_session(segments, fs=30.0, params: Params | None = None,
                     noise: NoiseModel | None = None, seed=0, filt=True) -> SessionResult:
    """segments: list of (State, seconds)."""
    rng = np.random.default_rng(seed)
    p = params or Params()
    nm = noise or NoiseModel()
    det = ReadingDetector(p)
    f = OneEuro(min_cutoff=1.0, beta=0.5)
    lh = p.line_h
    x0, x1 = 0.04, 0.96

    offset = 0.0          # scroll offset in viewport units
    line = 3              # current document line of the reader
    bias = np.zeros(2)
    t = 0.0
    dt = 1 / fs
    T, TS, PS = [], [], []
    scrolls = []
    lines_read = 0
    max_y = 0.0
    ran_off = 0

    def fixations_reading():
        nonlocal line, lines_read
        while True:
            x = x0 + rng.uniform(0.0, 0.05)
            while x < x1:
                yield (x, line), rng.uniform(0.18, 0.28)
                if rng.random() < 0.08:        # regression
                    x -= rng.uniform(0.03, 0.10)
                else:
                    x += rng.uniform(0.06, 0.12)
            line += 1
            lines_read += 1

    for seg_state, dur in segments:
        t_end = t + dur
        if seg_state == State.READING:
            gen = fixations_reading()
        while t < t_end:
            if seg_state == State.READING:
                (fx, fline), fdur = next(gen)
                tgt = lambda: (fx, (fline + 0.5) * lh - offset)
            elif seg_state == State.SKIMMING:
                fdur = rng.uniform(0.15, 0.4)
                pt = (rng.uniform(0, 1), rng.uniform(0.05, 0.95))
                tgt = lambda: pt
            else:
                fdur = 0.5
                tgt = None
            t_fix_end = min(t + fdur, t_end)
            while t < t_fix_end:
                bias += rng.normal(0, nm.drift_sd * np.sqrt(dt), 2)
                if tgt is None or rng.random() < nm.dropout_p:
                    gaze = None
                    if tgt is None and rng.random() < 0.3:
                        # looking off-screen but face visible
                        gaze_raw = np.array([rng.uniform(-0.5, 1.5), rng.uniform(1.3, 1.8)])
                        gaze = tuple(f(t, gaze_raw)) if filt else tuple(gaze_raw)
                else:
                    truth = np.array(tgt())
                    if seg_state == State.READING:
                        max_y = max(max_y, truth[1])
                        if truth[1] > 1.0:
                            ran_off += 1
                    meas = truth + bias + rng.normal(0, [nm.sd_x, nm.sd_y])
                    if rng.random() < nm.outlier_p:
                        meas = meas + rng.normal(0, 0.4, 2)
                    gaze = tuple(f(t, meas)) if filt else tuple(meas)
                evs = det.push(t, gaze)
                for e in evs:
                    if e.kind == "scroll_line":
                        offset += lh
                        scrolls.append((t, e.kind, seg_state.value))
                    elif e.kind == "scroll_half":
                        offset += 0.5
                        scrolls.append((t, e.kind, seg_state.value))
                T.append(t)
                TS.append(seg_state.value)
                PS.append(det.state.value)
                t += dt
    # count frames, not samples, for ran_off
    return SessionResult(np.array(T), np.array(TS), np.array(PS), lines_read,
                         scrolls, max_y, int(ran_off))
