import numpy as np
import pytest

from gazescroll import features as F
from gazescroll.filters import OneEuro
from gazescroll.model import GazeRegressor, calibration_samples
from gazescroll.reading import Params, ReadingDetector, State
from gazescroll.synth import (SCREEN_H_CM, SCREEN_W_CM, NoiseModel, Subject, grid_points,
                              random_head, simulate_session, synth_features)


# --- features ---------------------------------------------------------------
def _fake_landmarks(iris_shift=0.0):
    lm = np.zeros((478, 3))
    lm[:, :2] = 0.5
    lm[F.R_OUTER, :2] = (0.40, 0.45); lm[F.R_INNER, :2] = (0.46, 0.45)
    lm[F.L_INNER, :2] = (0.54, 0.45); lm[F.L_OUTER, :2] = (0.60, 0.45)
    lm[F.R_UPPER, :2] = (0.43, 0.44); lm[F.R_LOWER, :2] = (0.43, 0.46)
    lm[F.L_UPPER, :2] = (0.57, 0.44); lm[F.L_LOWER, :2] = (0.57, 0.46)
    lm[F.R_IRIS, :2] = (0.43 + iris_shift, 0.45)
    lm[F.L_IRIS, :2] = (0.57 + iris_shift, 0.45)
    lm[F.NOSE_TIP, :2] = (0.5, 0.52); lm[F.CHIN, :2] = (0.5, 0.7); lm[F.FOREHEAD, :2] = (0.5, 0.3)
    return lm


def test_feature_vector_shape_and_iris_direction():
    f0 = F.extract(_fake_landmarks(0.0))
    f1 = F.extract(_fake_landmarks(0.01))
    assert f0.shape == (F.N_FEATURES,)
    assert abs(f0[0]) < 1e-9 and f1[0] > 0 and f1[2] > 0   # iris moved along eye axis
    assert not F.is_blink(f0)


def test_head_pose_identity_matrix():
    assert np.allclose(F.head_pose_from_matrix(np.eye(4)), 0)


# --- gaze model -------------------------------------------------------------
def _calibrate(rng, subj, passes=2):
    X, Y = [], []
    for _ in range(passes):
        for p in grid_points(3):
            h = random_head(rng)
            s = [(i / 30, synth_features(p, h, subj, rng)) for i in range(36)]
            X.append(calibration_samples(s, 0.0)); Y.append(p)
    return GazeRegressor().fit(np.array(X), np.array(Y))


def _test_error(m, rng, subj, base=(3.5, 6.0, 30.0), n=150):
    tp = rng.uniform([0.3, 0.5], [SCREEN_W_CM - 0.3, SCREEN_H_CM - 0.5], (n, 2))
    feats = np.array([synth_features(p, random_head(rng, base), subj, rng) for p in tp])
    return np.abs(m.predict(feats) - tp).mean(0)


def test_calibration_reaches_1cm_target():
    errs = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        subj = Subject.random(rng)
        errs.append(_test_error(_calibrate(rng, subj), rng, subj))
    ex, ey = np.mean(errs, 0)
    assert ex < 1.0 and ey < 1.0, (ex, ey)


def test_online_update_recovers_after_head_shift():
    before, after = [], []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        subj = Subject.random(rng)
        m = _calibrate(rng, subj)
        shifted = (3.5, 9.0, 26.0)
        before.append(_test_error(m, rng, subj, shifted)[1])
        for _ in range(40):
            p = np.array([rng.uniform(0.3, 0.8), rng.uniform(1, 14)])   # line starts
            f = np.mean([synth_features(p, random_head(rng, shifted), subj, rng) for _ in range(4)], 0)
            m.update(f, p)
        after.append(_test_error(m, rng, subj, shifted)[1])
    assert np.mean(after) < np.mean(before)


def test_model_roundtrip():
    rng = np.random.default_rng(1)
    subj = Subject.random(rng)
    m = _calibrate(rng, subj, passes=1)
    m2 = GazeRegressor.from_dict(m.to_dict())
    f = synth_features((3, 7), random_head(rng), subj, rng)
    assert np.allclose(m.predict(f), m2.predict(f))


def test_calibration_samples_rejects_settle_and_outliers():
    base = np.ones(F.N_FEATURES)
    s = [(0.05, base * 50)] + [(0.4 + i / 30, base + 0.001 * i) for i in range(20)] + [(1.2, base * -9)]
    med = calibration_samples(s, 0.0)
    assert np.allclose(med, base, atol=0.02)


# --- filter -----------------------------------------------------------------
def test_one_euro_reduces_jitter_keeps_jumps():
    rng = np.random.default_rng(0)
    f = OneEuro(min_cutoff=1.0, beta=0.5)
    ts = np.arange(0, 4, 1 / 30)
    truth = np.where(ts < 2, 0.2, 0.8)
    out = np.array([f(t, [v + rng.normal(0, 0.1)])[0] for t, v in zip(ts, truth)])
    assert np.std(out[30:55] - 0.2) < 0.05
    assert abs(out[np.searchsorted(ts, 2.4)] - 0.8) < 0.1     # follows a big jump in < 400 ms


# --- reading detector, closed loop -------------------------------------------
SEGS = [(State.READING, 60), (State.SKIMMING, 15), (State.AWAY, 8),
        (State.READING, 60), (State.SKIMMING, 15)]


@pytest.mark.parametrize("seed", range(5))
def test_reading_session_default_noise(seed):
    r = simulate_session(SEGS, seed=seed)
    assert r.reading_f1() >= 0.9
    assert r.false_scrolls() <= 1
    assert r.ran_off_bottom == 0              # the text never ran off the screen
    assert r.max_truth_y_reading < 0.95
    assert len(r.scrolls) >= 15


def test_no_scroll_when_only_skimming_or_away():
    r = simulate_session([(State.SKIMMING, 60), (State.AWAY, 20), (State.SKIMMING, 60)], seed=3)
    assert len(r.scrolls) <= 1


def test_heavy_noise_degrades_gracefully():
    nm = NoiseModel(sd_x=0.2, sd_y=0.06, drift_sd=0.008)
    r = simulate_session(SEGS, seed=0, noise=nm)
    assert r.false_scrolls() <= 2
    assert r.ran_off_bottom == 0


def test_manual_scroll_pauses_auto_scroll():
    det = ReadingDetector(Params())
    det.manual_scroll(0.0)
    assert det.manual_until == pytest.approx(3.0)


# --- learning from manual corrections ------------------------------------------
from gazescroll.adapt import BOUNDS, ScrollAdapter  # noqa: E402
from gazescroll.synth import UserPolicy  # noqa: E402


def test_adapter_rules_and_bounds():
    ad = ScrollAdapter(Params(scroll_zone=0.65))
    ad.on_auto_scroll(10.0)
    assert ad.on_manual_scroll(11.0, -1) == "early"
    assert ad.params.scroll_zone > 0.65
    assert ad.on_manual_scroll(20.0, +1) == "late"
    assert ad.on_manual_scroll(20.2, -1) is None            # no auto-scroll just before
    for i in range(100):
        ad.on_manual_scroll(100.0 + 10 * i, +1)
    assert ad.params.scroll_zone == BOUNDS["scroll_zone"][0]


def test_adapter_learns_reading_tempo():
    ad = ScrollAdapter()
    for i in range(20):
        ad.on_sweep(2.5 * i)
    assert abs(ad.line_time - 2.5) < 0.05
    assert abs(ad.params.dwell_time - 2.0) < 0.05


def test_profile_roundtrip():
    ad = ScrollAdapter(Params(scroll_zone=0.7))
    ad.on_sweep(0.0); ad.on_sweep(3.0)
    ad2 = ScrollAdapter()
    ad2.load(ad.profile())
    assert ad2.params.scroll_zone == pytest.approx(0.7) and ad2.line_time == pytest.approx(3.0)


@pytest.mark.parametrize("start", [0.85, 0.30])
def test_closed_loop_corrections_die_out(start):
    """A badly tuned start is fixed by the user's own corrections within a few sessions."""
    ad = ScrollAdapter(Params(scroll_zone=start))
    manual = [len(simulate_session([(State.READING, 120)], seed=s, adapter=ad,
                                   user=UserPolicy()).manual) for s in range(6)]
    assert sum(manual[-3:]) <= 1, manual
    assert 0.45 <= ad.params.scroll_zone <= 0.80


# --- narrow eyes / poor calibration ----------------------------------------------
def test_blink_detector_adapts_to_narrow_eyes():
    rng = np.random.default_rng(0)
    det = F.BlinkDetector()
    f = np.zeros(F.N_FEATURES)
    flags = []
    for i in range(200):
        ear = 0.12 + rng.normal(0, 0.006)            # narrow eyes: always below 0.18
        if i in (100, 101, 102):
            ear = 0.03                                # a real blink
        f[4] = f[5] = ear
        flags.append(det(f))
    assert F.is_blink(f)                              # the fixed threshold fails ...
    assert sum(flags[20:]) <= 4 and all(flags[100:103])   # ... the adaptive one works


def test_line_lock_scrolls_with_x_only_signal():
    """Vertical gaze unknown: one line per return sweep keeps the text in view."""
    p = Params(line_lock=True)
    r = simulate_session([(State.READING, 90), (State.SKIMMING, 20)], params=p,
                         seed=2, y_blind=True)
    assert r.ran_off_bottom == 0
    assert r.false_scrolls() <= 2
    assert len(r.scrolls) >= 0.6 * r.lines_read


# --- pace keeper + head nod -----------------------------------------------------------
from gazescroll.reading import NodDetector  # noqa: E402


def test_pace_keeper_follows_even_when_sweeps_are_missed():
    """Heavy x-noise hides many sweeps; the pace keeper still keeps up."""
    nm = NoiseModel(sd_x=0.25, sd_y=0.06)
    base = simulate_session([(State.READING, 90)], params=Params(line_lock=True, pace_factor=0,
                            offscreen_away=False), noise=nm, seed=4, y_blind=True)
    paced = simulate_session([(State.READING, 90)], params=Params(line_lock=True,
                             offscreen_away=False), noise=nm, seed=4, y_blind=True)
    assert len(paced.scrolls) > len(base.scrolls)
    assert len(paced.scrolls) <= paced.lines_read + 3          # no runaway


def test_no_pace_scroll_while_skimming():
    """Known limit of the x-only mode: random skimming can mimic line ends now and then."""
    r = simulate_session([(State.SKIMMING, 60)], params=Params(line_lock=True, offscreen_away=False),
                         seed=1, y_blind=True)
    assert len(r.scrolls) <= 22


def test_nod_detector():
    nod = NodDetector()
    t = np.arange(0, 6, 1 / 30)
    pitch = 0.24 + 0.004 * np.sin(7 * t)
    pitch[(t > 3.0) & (t < 3.3)] += 0.12          # quick nod
    hits = [ti for ti, p in zip(t, pitch) if nod(ti, p)]
    assert len(hits) == 1 and 3.0 < hits[0] < 3.8
    nod2 = NodDetector()
    held = 0.24 + np.where(t > 3, 0.12, 0.0)       # head lowered and kept there
    assert not any(nod2(ti, p) for ti, p in zip(t, held))


def test_long_low_ear_is_looking_down_not_blink():
    det = F.BlinkDetector()
    f = np.zeros(F.N_FEATURES)
    out = []
    for i in range(150):
        f[4] = f[5] = 0.14 if i < 60 else 0.06      # from 2 s on: looking at the screen bottom
        out.append(det(f))
    assert any(out[60:70]) and not any(out[80:])     # first ~0.4 s rejected, then accepted


def test_rolling_norm_removes_offset_and_scale():
    from gazescroll.filters import RollingNorm
    rn = RollingNorm()
    out = None
    for i in range(600):
        t = i / 30
        x_true = (i % 75) / 75                        # sweeping lines
        out = rn(t, 300 * x_true - 250)               # badly scaled, shifted calibration
        if i == 599:
            assert out is not None
    assert abs(rn(20.0, 300 * 0.5 - 250) - 0.5) < 0.12
