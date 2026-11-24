"""Personal gaze regressor: polynomial features + ridge, with RLS online update.

Pure numpy so the same math ports 1:1 to Kotlin / JS.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# per-feature scale floors (see features.FEATURE_NAMES)
SD_FLOOR = np.array([0.02, 0.02, 0.02, 0.02,  # iris u/v
                     0.02, 0.02,              # EAR
                     0.05, 0.05, 0.05,        # yaw, pitch, roll (rad)
                     0.02, 0.02, 0.02,        # face position / scale
                     0.02, 0.02])             # eye centre

def poly2(x: np.ndarray) -> np.ndarray:
    """[1, x_i, x_i*x_j (i<=j)] for each row."""
    x = np.atleast_2d(x)
    n, d = x.shape
    iu = np.triu_indices(d)
    quad = (x[:, :, None] * x[:, None, :])[:, iu[0], iu[1]]
    return np.hstack([np.ones((n, 1)), x, quad])


@dataclass
class GazeRegressor:
    """Maps feature vectors -> screen coordinates (any unit, e.g. cm or px).

    Uses a reduced feature set for the quadratic terms to stay stable with
    only 9-16 calibration points; the rest enter linearly.
    """
    lam: float = 0.3
    # Quadratic terms only pay off with >= ~30 calibration samples (e.g. after
    # online refinement); with 9-18 points a linear model is more accurate.
    quad_idx: tuple[int, ...] = ()
    forgetting: float = 0.98
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    W: np.ndarray | None = None          # (n_basis, 2)
    P: np.ndarray | None = None          # RLS inverse covariance
    n_online: int = field(default=0)

    # --- basis -----------------------------------------------------------
    def _basis(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        Z = (X - self.mu) / self.sd
        if not self.quad_idx:
            return np.hstack([np.ones((len(Z), 1)), Z])
        q = poly2(Z[:, list(self.quad_idx)])
        rest = [i for i in range(Z.shape[1]) if i not in self.quad_idx]
        return np.hstack([q, Z[:, rest]])

    # --- batch calibration ---------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray) -> "GazeRegressor":
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        self.mu = X.mean(0)
        # floor the scale so near-constant, noisy features are not blown up
        self.sd = np.maximum(X.std(0), SD_FLOOR[: X.shape[1]])
        B = self._basis(X)
        reg = self.lam * np.eye(B.shape[1])
        reg[0, 0] = 0.0  # don't shrink the intercept
        A = B.T @ B + reg
        self.W = np.linalg.solve(A, B.T @ Y)
        self.P = np.linalg.inv(A)
        self.n_online = 0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("not calibrated")
        out = self._basis(X) @ self.W
        return out[0] if np.ndim(X) == 1 else out

    # --- online refinement (e.g. from return sweeps) ----------------------
    def update(self, x: np.ndarray, y: np.ndarray, weight: float = 1.0) -> None:
        """Recursive least squares step with forgetting factor.

        `y` may contain NaN for an unknown axis (e.g. only the line-start x and
        the line y are known) — that axis is then left untouched.
        """
        b = self._basis(x)[0]
        lam = self.forgetting
        Pb = self.P @ b
        k = Pb / (lam / weight + b @ Pb)
        err = np.asarray(y, float) - b @ self.W
        for j in range(self.W.shape[1]):
            if not np.isnan(err[j]):
                self.W[:, j] += k * err[j]
        self.P = (self.P - np.outer(k, Pb)) / lam
        self.n_online += 1

    # --- persistence ----------------------------------------------------------
    def to_dict(self) -> dict:
        return {"lam": self.lam, "quad_idx": list(self.quad_idx),
                "mu": self.mu.tolist(), "sd": self.sd.tolist(), "W": self.W.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "GazeRegressor":
        m = cls(lam=d["lam"], quad_idx=tuple(d["quad_idx"]))
        m.mu, m.sd, m.W = (np.array(d[k]) for k in ("mu", "sd", "W"))
        B = np.eye(m.W.shape[0])
        m.P = B
        return m


def calibration_samples(samples: list[tuple[float, np.ndarray]], target_t0: float,
                        settle: float = 0.3, mad_k: float = 2.5) -> np.ndarray | None:
    """Robust per-target feature: drop the first `settle` seconds, MAD-filter, median."""
    xs = np.array([f for t, f in samples if t - target_t0 >= settle])
    if len(xs) < 3:
        return None
    med = np.median(xs, 0)
    mad = np.median(np.abs(xs - med), 0) + 1e-9
    keep = np.all(np.abs(xs - med) <= mad_k * 1.4826 * mad, axis=1)
    if keep.sum() < 3:
        return med
    return np.median(xs[keep], 0)
