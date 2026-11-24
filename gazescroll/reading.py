"""Reading detector + scroll controller.

Coordinates are normalised to the text viewport:
    x: 0 = left edge of the text column, 1 = right edge
    y: 0 = top of the viewport, 1 = bottom
`line_h` is the line pitch in the same y units.

Scrolling is driven by return sweeps (end of line -> start of next line),
which are large and survive 1-2 cm of gaze noise.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class State(str, Enum):
    READING = "OLVAS"
    SKIMMING = "PASZTAZ"
    AWAY = "FELRENEZ"


@dataclass
class Params:
    line_h: float = 0.04
    sweep_min_dx: float = 0.40       # leftward jump, fraction of column width
    sweep_window: float = 0.30       # s, the jump must happen within this
    sweep_end_x: float = 0.40        # sweep must land in the left part
    sweep_max_up: float = 0.5        # lines; a sweep must not go up more than this
    sweep_prelude: float = 0.8       # s of rightward progress required before a sweep
    sweep_refractory: float = 0.6    # s
    step_dt: float = 0.2             # s, resampling step for the forward-ratio
    fwd_max_dx: float = 0.30
    window: float = 2.5              # s, forward-ratio window
    reading_fwd_ratio: float = 0.50   # enter READING
    reading_stay_ratio: float = 0.30  # stay in READING
    sweep_memory: float = 4.0        # s, a sweep keeps READING alive this long
    away_after: float = 0.5          # s without face
    offscreen_margin: float = 0.15
    scroll_zone: float = 0.65        # sweeps below this y scroll one line
    dwell_zone: float = 0.85         # dwelling below this y ...
    dwell_time: float = 1.5          # ... for this long pages half a screen
    post_scroll_mute: float = 0.4
    manual_pause: float = 3.0
    # line_lock: scroll one line on EVERY return sweep while reading, whatever the
    # gaze y. Needs only the horizontal signal, so it works with a poor or
    # x-only calibration: the line being read stays at a constant height.
    line_lock: bool = False
    # pace: in line_lock mode, if reading continues but no scroll came for
    # pace_factor * line_time seconds (missed sweeps), scroll one line anyway.
    pace_factor: float = 1.3
    line_time: float = 2.5
    # in x-only mode a gaze far left/right is calibration drift, not "looked away"
    offscreen_away: bool = True
    lock_peak_min_x: float = 0.65    # x-only: sweep must start near the line end ...
    lock_end_x: float = 0.35         # ... and land near the line start


@dataclass
class Event:
    t: float
    kind: str           # "sweep" | "scroll_line" | "scroll_half" | "state"
    value: object = None


@dataclass
class ReadingDetector:
    p: Params = field(default_factory=Params)

    def __post_init__(self):
        self.hist: deque = deque()
        self.state = State.AWAY
        self.last_face_t = -1e9
        self.last_sweep_t = -1e9
        self.last_scroll_t = -1e9
        self.manual_until = -1e9
        self.dwell_start: float | None = None
        self.reading_since = 1e9
        self.sweep_times: list[float] = []
        self.events: list[Event] = []

    # -- public API ------------------------------------------------------------
    def manual_scroll(self, t: float):
        self.manual_until = t + self.p.manual_pause

    def push(self, t: float, gaze: tuple[float, float] | None) -> list[Event]:
        """Feed one filtered gaze sample (None = no face). Returns new events."""
        out: list[Event] = []
        p = self.p
        if gaze is not None:
            self.last_face_t = t
            self.hist.append((t, float(gaze[0]), float(gaze[1])))
        while self.hist and t - self.hist[0][0] > max(p.window, p.sweep_prelude + p.sweep_window) + 0.5:
            self.hist.popleft()

        new_state = self._classify(t, gaze)
        if gaze is not None and t - self.last_scroll_t > p.post_scroll_mute:
            sweep = self._detect_sweep(t)
            if sweep is not None:
                self.last_sweep_t = t
                self.sweep_times.append(t)
                if len(self.sweep_times) > 20:
                    self.sweep_times.pop(0)
                out.append(Event(t, "sweep", sweep))
                new_state = self._classify(t, gaze)
                if (new_state == State.READING and t >= self.manual_until
                        and (p.line_lock or sweep[1] >= p.scroll_zone)):
                    out.append(Event(t, "scroll_line", 1))
                    self.last_scroll_t = t
                    self.hist.clear()   # content moved: old samples are stale

        # pace keeper (line_lock): missed sweeps must not leave the reader behind
        if (p.line_lock and p.pace_factor > 0 and gaze is not None
                and new_state == State.READING and t >= self.manual_until
                and t - self.last_scroll_t > p.pace_factor * p.line_time
                and t - self.reading_since > p.line_time):
            out.append(Event(t, "scroll_line", 1))
            self.last_scroll_t = t

        # dwell at the very bottom -> half page
        if (gaze is not None and not p.line_lock and new_state == State.READING
                and gaze[1] >= p.dwell_zone):
            if self.dwell_start is None:
                self.dwell_start = t
            elif t - self.dwell_start >= p.dwell_time and t >= self.manual_until:
                out.append(Event(t, "scroll_half", 0.5))
                self.last_scroll_t = t
                self.dwell_start = None
                self.hist.clear()
        else:
            self.dwell_start = None

        if new_state != self.state:
            if new_state == State.READING:
                self.reading_since = t
            self.state = new_state
            out.append(Event(t, "state", new_state))
        self.events.extend(out)
        return out

    # -- internals ---------------------------------------------------------------
    def _resampled(self, t: float, span: float) -> np.ndarray:
        pts = [(tt, x, y) for tt, x, y in self.hist if t - tt <= span]
        if len(pts) < 2:
            return np.empty((0, 3))
        a = np.array(pts)
        grid = np.arange(a[0, 0], a[-1, 0] + 1e-9, self.p.step_dt)
        return np.column_stack([grid, np.interp(grid, a[:, 0], a[:, 1]),
                                np.interp(grid, a[:, 0], a[:, 2])])

    def forward_ratio(self, t: float) -> float | None:
        """Share of 200 ms steps that move right by a word-ish amount. None = too little data."""
        r = self._resampled(t, self.p.window)
        if len(r) < 4:
            return None
        d = np.diff(r[:, 1:], axis=0)
        fwd = (d[:, 0] > 0) & (d[:, 0] < self.p.fwd_max_dx) & (np.abs(d[:, 1]) < 1.5 * self.p.line_h)
        sweep_like = d[:, 0] < -self.p.sweep_min_dx * 0.75
        n = len(d) - sweep_like.sum()
        return float(fwd.sum() / n) if n > 0 else None

    def _detect_sweep(self, t: float):
        p = self.p
        recent = [(tt, x, y) for tt, x, y in self.hist if t - tt <= p.sweep_window]
        if len(recent) < 2 or t - self.last_sweep_t < p.sweep_refractory:
            return None
        xs = np.array([r[1] for r in recent])
        i_max = int(np.argmax(xs))
        x_now, y_now = recent[-1][1], recent[-1][2]
        drop = xs[i_max] - x_now
        if drop < p.sweep_min_dx or x_now > p.sweep_end_x:
            return None
        if p.line_lock and (xs[i_max] < p.lock_peak_min_x or x_now > p.lock_end_x):
            return None
        if y_now - recent[i_max][2] < -p.sweep_max_up * p.line_h * 3:  # generous for noise
            return None
        # prelude: rightward progress before the peak
        t_peak = recent[i_max][0]
        pre = [(tt, x) for tt, x, _ in self.hist if t_peak - p.sweep_prelude - 0.3 <= tt <= t_peak]
        if len(pre) < 3:
            return None
        pre = np.array(pre)
        early = pre[pre[:, 0] <= t_peak - p.sweep_prelude * 0.6]
        if len(early) == 0:
            return None
        if xs[i_max] - np.median(early[:, 1]) < 0.12:
            return None
        return (x_now, y_now)

    def _rhythmic(self, t: float) -> bool:
        """Two consecutive return sweeps about one line-time apart, the last one recent."""
        lt = self.p.line_time
        s = [x for x in self.sweep_times if t - x <= 2.6 * lt]
        if len(s) < 2 or t - s[-1] > 1.6 * lt:
            return False
        gaps = np.diff(s)
        return bool(np.any((gaps >= 0.5 * lt) & (gaps <= 1.8 * lt)))

    def _classify(self, t: float, gaze) -> State:
        p = self.p
        if gaze is None:
            return State.AWAY if t - self.last_face_t >= p.away_after else self.state
        x, y = gaze
        m = p.offscreen_margin
        if p.offscreen_away and not (-m <= x <= 1 + m and -m <= y <= 1 + m):
            return State.AWAY
        fr = self.forward_ratio(t)
        if fr is None:                      # e.g. right after a scroll cleared the history
            return self.state if self.state != State.AWAY else State.SKIMMING
        recent_sweep = t - self.last_sweep_t <= p.sweep_memory
        # hysteresis: harder to enter READING than to stay in it
        enter = p.reading_fwd_ratio - (0.05 if recent_sweep else 0.0)
        stay = p.reading_stay_ratio
        if p.line_lock:
            # no vertical signal: demand the reading rhythm (a recent return sweep)
            # and a clearly rightward drift, otherwise skimming looks like reading
            if not self._rhythmic(t):
                return State.SKIMMING
            enter, stay = enter + 0.1, stay + 0.1
        if fr >= (stay if self.state == State.READING else enter):
            return State.READING
        return State.SKIMMING


class NodDetector:
    """Deliberate head nod -> discrete command (e.g. scroll a third of a page).

    Uses the head-pitch feature. A nod = deviation from the 2 s running median
    larger than `amp` that returns below amp/3 within `max_dur` seconds.
    Natural reading movement in a real log stayed below ~0.07.
    """

    def __init__(self, amp: float = 0.08, max_dur: float = 0.8, refractory: float = 1.5):
        self.amp, self.max_dur, self.refractory = amp, max_dur, refractory
        self.hist: deque = deque()
        self.peak_t: float | None = None
        self.last_nod = -1e9

    def __call__(self, t: float, pitch: float) -> bool:
        self.hist.append((t, pitch))
        while self.hist and t - self.hist[0][0] > 2.0:
            self.hist.popleft()
        base = float(np.median([p for _, p in self.hist]))
        dev = abs(pitch - base)
        if self.peak_t is None:
            if dev > self.amp and t - self.last_nod > self.refractory and len(self.hist) > 10:
                self.peak_t = t
                self.base0 = float(np.median([p for tt, p in self.hist if tt < t - 0.3] or [base]))
            return False
        if abs(pitch - self.base0) < self.amp / 3:
            self.peak_t = None
            self.last_nod = t
            return True
        if t - self.peak_t > self.max_dur:      # held, not a nod
            self.peak_t = None
        return False
