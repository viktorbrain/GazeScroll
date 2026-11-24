"""Benchmark the model on synthetic data and print a Markdown table.

    python evaluate.py            # default noise levels
"""
import numpy as np

from gazescroll.model import GazeRegressor, calibration_samples
from gazescroll.reading import State
from gazescroll.synth import (SCREEN_H_CM, SCREEN_W_CM, NoiseModel, Subject, grid_points,
                              random_head, simulate_session, synth_features)

SEGS = [(State.READING, 60), (State.SKIMMING, 15), (State.AWAY, 8),
        (State.READING, 60), (State.SKIMMING, 15)]


def gaze_error(n_side, passes, seeds=20):
    out = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        subj = Subject.random(rng)
        X, Y = [], []
        for _ in range(passes):
            for p in grid_points(n_side):
                h = random_head(rng)
                s = [(i / 30, synth_features(p, h, subj, rng)) for i in range(36)]
                X.append(calibration_samples(s, 0.0)); Y.append(p)
        m = GazeRegressor().fit(np.array(X), np.array(Y))
        tp = rng.uniform([0.3, 0.5], [SCREEN_W_CM - 0.3, SCREEN_H_CM - 0.5], (150, 2))
        F = np.array([synth_features(p, random_head(rng), subj, rng) for p in tp])
        out.append(np.abs(m.predict(F) - tp).mean(0))
    return np.mean(out, 0)


def reading(nm, seeds=10):
    rs = [simulate_session(SEGS, noise=nm, seed=s) for s in range(seeds)]
    return (np.mean([r.reading_f1() for r in rs]),
            np.mean([len(r.scrolls) for r in rs]),
            np.mean([r.false_scrolls() for r in rs]),
            np.mean([r.max_truth_y_reading for r in rs]),
            sum(r.ran_off_bottom > 0 for r in rs))


if __name__ == "__main__":
    print("| Kalibráció | Hiba x (cm) | Hiba y (cm) |\n| --- | --- | --- |")
    for name, n, p in [("9 pont", 3, 1), ("9 pont × 2 kör", 3, 2), ("16 pont", 4, 1)]:
        ex, ey = gaze_error(n, p)
        print(f"| {name} | {ex:.2f} | {ey:.2f} |")
    print()
    print("| Zaj (szórás x / y, cm) | Olvasás F1 | Görgetés / 2,5 perc | Téves görgetés | Max. sor y | Kifutott |\n"
          "| --- | --- | --- | --- | --- | --- |")
    for sx, sy in [(0.08, 0.025), (0.12, 0.035), (0.20, 0.06), (0.30, 0.08)]:
        f1, n, fs, my, off = reading(NoiseModel(sd_x=sx, sd_y=sy))
        print(f"| {sx*7:.1f} / {sy*15:.1f} | {f1:.3f} | {n:.1f} | {fs:.1f} | {my:.2f} | {off}/10 |")
