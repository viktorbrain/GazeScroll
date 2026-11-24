"""Learning scroll behaviour from the user's own corrections.

Labels come for free:
  * manual scroll UP shortly after an auto-scroll  -> auto-scroll was too EARLY
  * manual scroll DOWN while reading, with no recent auto-scroll -> too LATE / missed
Reading tempo (time per line) is learned from the interval between return sweeps.

The adapter nudges a few Params fields within safe bounds and can be saved as a
per-user profile (JSON / localStorage).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .reading import Params

BOUNDS = {
    "scroll_zone": (0.45, 0.85),
    "sweep_min_dx": (0.30, 0.55),
    "reading_fwd_ratio": (0.40, 0.65),
    "dwell_time": (1.0, 3.0),
    "sweep_refractory": (0.4, 1.0),
}


def _clamp(name, v):
    lo, hi = BOUNDS[name]
    return min(max(v, lo), hi)


@dataclass
class ScrollAdapter:
    params: Params = field(default_factory=Params)
    early_window: float = 3.0      # s after an auto-scroll in which "scroll up" means too early
    late_quiet: float = 0.5        # s without auto-scroll before "scroll down" means too late
    step_zone: float = 0.03
    step_dx: float = 0.02
    step_enter: float = 0.02
    tempo_alpha: float = 0.15

    def __post_init__(self):
        self.last_auto = -1e9
        self.last_sweep = None
        self.line_time: float | None = None
        self.n_early = self.n_late = 0
        self.history: list[tuple[float, str]] = []

    def new_session(self):
        """Forget timestamps (clock may restart); keep everything learned."""
        self.last_auto = -1e9
        self.last_sweep = None

    # --- observations ------------------------------------------------------------
    def on_auto_scroll(self, t: float):
        self.last_auto = t

    def on_sweep(self, t: float):
        if self.last_sweep is not None:
            dt = t - self.last_sweep
            if 0.8 <= dt <= 12.0:            # ignore gaps (looked away, skimmed)
                self.line_time = dt if self.line_time is None else \
                    (1 - self.tempo_alpha) * self.line_time + self.tempo_alpha * dt
                p = self.params
                p.dwell_time = _clamp("dwell_time", 0.8 * self.line_time)
                p.sweep_refractory = _clamp("sweep_refractory", 0.3 * self.line_time)
        self.last_sweep = t

    def on_manual_scroll(self, t: float, direction: int, reading: bool = True) -> str | None:
        """direction: -1 = user pulled content back (scrolled up), +1 = scrolled down."""
        p = self.params
        if direction < 0 and t - self.last_auto <= self.early_window:
            self.n_early += 1
            p.scroll_zone = _clamp("scroll_zone", p.scroll_zone + self.step_zone)
            p.sweep_min_dx = _clamp("sweep_min_dx", p.sweep_min_dx + self.step_dx)
            self.history.append((t, "early"))
            return "early"
        if direction > 0 and reading and t - self.last_auto > self.late_quiet:
            self.n_late += 1
            p.scroll_zone = _clamp("scroll_zone", p.scroll_zone - self.step_zone)
            p.sweep_min_dx = _clamp("sweep_min_dx", p.sweep_min_dx - self.step_dx)
            p.reading_fwd_ratio = _clamp("reading_fwd_ratio", p.reading_fwd_ratio - self.step_enter)
            self.history.append((t, "late"))
            return "late"
        return None

    # --- profile -------------------------------------------------------------------
    def profile(self) -> dict:
        p = self.params
        return {k: getattr(p, k) for k in BOUNDS} | {
            "line_time": self.line_time, "n_early": self.n_early, "n_late": self.n_late}

    def load(self, prof: dict):
        for k in BOUNDS:
            if k in prof and prof[k] is not None:
                setattr(self.params, k, _clamp(k, float(prof[k])))
        self.line_time = prof.get("line_time")
        self.n_early = int(prof.get("n_early", 0))
        self.n_late = int(prof.get("n_late", 0))
