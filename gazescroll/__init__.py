"""gazescroll – camera-based gaze estimation + reading-driven auto-scroll (model skeleton)."""
from .features import extract, FEATURE_NAMES, N_FEATURES
from .model import GazeRegressor
from .filters import OneEuro
from .reading import ReadingDetector, Params, State
