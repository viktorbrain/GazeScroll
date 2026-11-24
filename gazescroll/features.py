"""Feature extraction from MediaPipe Face Landmarker output.

Input: 478 x 3 landmark array (normalized image coords, x/y in 0..1, z relative)
       + optional 4x4 facial transformation matrix (head pose).
Output: fixed-length feature vector for the gaze regressor.
"""
from __future__ import annotations

import numpy as np

# MediaPipe Face Mesh indices (subject's right eye = image left on a mirrored selfie)
R_OUTER, R_INNER = 33, 133
L_INNER, L_OUTER = 362, 263
R_UPPER, R_LOWER = 159, 145
L_UPPER, L_LOWER = 386, 374
R_IRIS = 468          # 468..472, center = 468
L_IRIS = 473          # 473..477, center = 473
NOSE_TIP = 1
CHIN = 152
FOREHEAD = 10

FEATURE_NAMES = [
    "r_iris_u", "r_iris_v", "l_iris_u", "l_iris_v",
    "r_ear", "l_ear",
    "yaw", "pitch", "roll",
    "face_cx", "face_cy", "face_scale",
    "eyes_cx", "eyes_cy",
]
N_FEATURES = len(FEATURE_NAMES)


def _eye_coords(lm: np.ndarray, outer: int, inner: int, upper: int, lower: int, iris: int):
    """Iris position in an eye-local frame: u along the eye axis, v perpendicular.

    Normalised by eye width so it is roughly invariant to distance from camera.
    """
    p_out, p_in = lm[outer, :2], lm[inner, :2]
    center = (p_out + p_in) / 2
    axis = p_in - p_out
    width = np.linalg.norm(axis) + 1e-9
    ex = axis / width
    ey = np.array([-ex[1], ex[0]])
    d = lm[iris, :2] - center
    u = float(d @ ex) / width
    v = float(d @ ey) / width
    ear = float(np.linalg.norm(lm[upper, :2] - lm[lower, :2]) / width)
    return u, v, ear, center, width


def head_pose_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """Yaw, pitch, roll (radians) from MediaPipe's 4x4 facial transformation matrix."""
    r = np.asarray(m, dtype=float)[:3, :3]
    sy = np.hypot(r[0, 0], r[1, 0])
    pitch = float(np.arctan2(r[2, 1], r[2, 2]))
    yaw = float(np.arctan2(-r[2, 0], sy))
    roll = float(np.arctan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll


def head_pose_from_landmarks(lm: np.ndarray) -> tuple[float, float, float]:
    """Cheap fallback when no transformation matrix is available."""
    r_c = (lm[R_OUTER, :2] + lm[R_INNER, :2]) / 2
    l_c = (lm[L_OUTER, :2] + lm[L_INNER, :2]) / 2
    eyes = (r_c + l_c) / 2
    iod = np.linalg.norm(l_c - r_c) + 1e-9
    roll = float(np.arctan2(l_c[1] - r_c[1], l_c[0] - r_c[0]))
    yaw = float((lm[NOSE_TIP, 0] - eyes[0]) / iod)
    face_h = np.linalg.norm(lm[CHIN, :2] - lm[FOREHEAD, :2]) + 1e-9
    pitch = float((lm[NOSE_TIP, 1] - eyes[1]) / face_h)
    return yaw, pitch, roll


def extract(lm: np.ndarray, transform: np.ndarray | None = None) -> np.ndarray:
    lm = np.asarray(lm, dtype=float)
    if lm.shape[0] < 478:
        raise ValueError("need 478 landmarks (enable iris refinement)")
    ru, rv, rear, rc, rw = _eye_coords(lm, R_OUTER, R_INNER, R_UPPER, R_LOWER, R_IRIS)
    lu, lv, lear, lc, lw = _eye_coords(lm, L_INNER, L_OUTER, L_UPPER, L_LOWER, L_IRIS)
    yaw, pitch, roll = (head_pose_from_matrix(transform) if transform is not None
                        else head_pose_from_landmarks(lm))
    xs, ys = lm[:, 0], lm[:, 1]
    face_cx, face_cy = float(xs.mean()), float(ys.mean())
    face_scale = float(np.linalg.norm(lm[CHIN, :2] - lm[FOREHEAD, :2]))
    eyes = (rc + lc) / 2
    return np.array([ru, rv, lu, lv, rear, lear, yaw, pitch, roll,
                     face_cx, face_cy, face_scale, eyes[0], eyes[1]])


def is_blink(features: np.ndarray, threshold: float = 0.18) -> bool:
    """Fixed-threshold blink test. Fails for narrow eyes - prefer BlinkDetector."""
    return float(min(features[4], features[5])) < threshold


class BlinkDetector:
    """Blink test relative to the person's own eye opening.

    A fixed EAR threshold (e.g. 0.18) marks almost every frame as a blink for
    people with a narrow palpebral fissure. Here the baseline is the running
    upper-median of that person's EAR, and a blink is a drop below `ratio` of it.
    """

    def __init__(self, ratio: float = 0.6, window: int = 150, warmup: int = 15):
        self.ratio, self.window, self.warmup = ratio, window, warmup
        self.buf: list[float] = []

    @property
    def baseline(self) -> float | None:
        if len(self.buf) < self.warmup:
            return None
        return float(np.percentile(self.buf, 70))

    def __call__(self, features: np.ndarray) -> bool:
        ear = float((features[4] + features[5]) / 2)
        base = self.baseline
        blink = base is not None and ear < self.ratio * base
        if not blink:                       # blinks must not drag the baseline down
            self.buf.append(ear)
            if len(self.buf) > self.window:
                self.buf.pop(0)
        return blink
