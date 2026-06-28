from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CUSTOM_DATA_DIR = DATA_DIR / "custom"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SHOWCASE_DIR = OUTPUT_DIR / "showcase"

DEFAULT_SEED = 0
DEFAULT_SYNTHETIC_SAMPLES = 12
MIN_REAL_IMAGES = 3
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Detection: ignore tiny contour fragments (fraction of image area)
DETECT_MIN_AREA_RATIO = 0.008

# Decision: ALERT only for clearly overloaded / failed scenes
ALERT_MIN_OBJECTS = 15
ALERT_COVERAGE = 0.65
ALERT_OBJECTS_WITH_COVERAGE = 12
ALERT_COVERAGE_WITH_COUNT = 0.50
