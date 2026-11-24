"""One Euro filter (Casiez et al., CHI 2012) for 2-D gaze."""
from __future__ import annotations

import math

import numpy as np


class OneEuro:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x_prev = self.dx_prev = self.t_prev = None

    def __call__(self, t: float, x) -> np.ndarray:
        x = np.asarray(x, float)
        if self.t_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t_prev, 1e-4)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = np.array([self._alpha(c, dt) for c in np.atleast_1d(cutoff)])
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class RollingNorm:
    """Self-calibrating x for the x-only mode.

    Reading sweeps the whole line width, so the 10th/90th percentile of the last
    `window` seconds of raw x map to 0.1/0.9 of the text column. This absorbs the
    drift of a poor calibration (head turned, phone moved).
    """

    def __init__(self, window: float = 12.0, lo: float = 10, hi: float = 90, min_n: int = 30):
        self.window, self.lo, self.hi, self.min_n = window, lo, hi, min_n
        self.buf: list[tuple[float, float]] = []

    def __call__(self, t: float, x: float) -> float | None:
        self.buf.append((t, x))
        while self.buf and t - self.buf[0][0] > self.window:
            self.buf.pop(0)
        if len(self.buf) < self.min_n:
            return None
        a = np.array([b[1] for b in self.buf])
        lo, hi = np.percentile(a, [self.lo, self.hi])
        return float(0.1 + 0.8 * (x - lo) / max(hi - lo, 1e-6))
