"""
Centralized Configuration
-------------------------
All magic numbers, paths, and defaults live here.
Import from this module instead of hardcoding values.

Secrets (API keys) are loaded from .env file — never hardcoded.
Copy .env.example → .env and fill in your values.
"""

import os
import logging

# Load .env file if present (secrets stay out of source code)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass  # python-dotenv not installed — rely on system env vars

# ──────────────────────────── Paths ────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHAIN_FILE = os.path.join(BASE_DIR, "chain.json")
LOGO_PATH = os.path.join(BASE_DIR, "watermark", "mylogo.png")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RESULTS_WATERMARKED = os.path.join(RESULTS_DIR, "watermarked")
RESULTS_ATTACKED = os.path.join(RESULTS_DIR, "attacked")
RESULTS_EXTRACTED = os.path.join(RESULTS_DIR, "extracted")
AUDIT_LOG_FILE = os.path.join(BASE_DIR, "audit.log")
ARCHIVE_DIR = os.path.join(BASE_DIR, "chain_archive")

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

# ──────────────────────────── DBSCAN Defaults ────────────────────────────
DEFAULT_EPS: float = 2.0
DEFAULT_MIN_SAMPLES: int = 3
DEFAULT_EDGE_THRESH: float = 0.08
DBSCAN_FALLBACK_COUNT: int = 2000
DBSCAN_MIN_CLUSTER_POSITIONS: int = 200

# ──────────────────────────── Adaptive DBSCAN ────────────────────────────
ADAPTIVE_K_NEIGHBORS: int = 5            # k for k-distance graph
ADAPTIVE_EPS_RANGE: tuple = (0.5, 10.0)  # search bounds for eps
ADAPTIVE_MIN_SAMPLES_RANGE: tuple = (2, 10)

# Modality-specific presets (eps, min_samples, edge_thresh)
MODALITY_PRESETS = {
    "mri":    {"eps": 3.0, "min_samples": 5, "edge_thresh": 0.06},
    "ct":     {"eps": 2.0, "min_samples": 3, "edge_thresh": 0.08},
    "xray":   {"eps": 2.5, "min_samples": 4, "edge_thresh": 0.07},
    "ultrasound": {"eps": 3.5, "min_samples": 6, "edge_thresh": 0.05},
    "unknown": {"eps": 2.0, "min_samples": 3, "edge_thresh": 0.08},
}

# ──────────────────────────── Logo Overlay ────────────────────────────
LOGO_SCALE: float = 0.22          # 22 % of image width
LOGO_OPACITY: float = 0.65        # alpha‐blend strength
LOGO_MARGIN: int = 18             # pixels from bottom‐right corner

# ──────────────────────────── Attack Params ────────────────────────────
NOISE_SIGMA: float = 10.0
CROP_RATIO: float = 0.85
JPEG_QUALITY: int = 50
ROTATION_ANGLE: float = 5.0       # degrees
SCALE_FACTOR: float = 0.5         # downscale ratio
MEDIAN_KERNEL: int = 3             # median filter kernel size

# ──────────────────────────── Detection ────────────────────────────
LOGO_MATCH_THRESHOLD: float = 0.80

# ──────────────────────────── Blockchain ────────────────────────────
BATCH_COMMIT_SIZE: int = 10        # blocks before flush in batch mode
CHAIN_PRUNE_KEEP: int = 500        # active blocks to keep before archiving

# ──────────────────────────── Processing Mode ────────────────────────────
# "full" — maximum quality, "edge" — balanced, "mobile" — fastest
PROCESSING_MODE: str = "full"

EDGE_DOWNSCALE: float = 0.5       # downscale factor for edge mode
MOBILE_DOWNSCALE: float = 0.25    # downscale factor for mobile mode
MOBILE_DBSCAN_SUBSAMPLE: int = 500  # max edge pixels to cluster in mobile mode

# ──────────────────────────── Compliance ────────────────────────────
AUDIT_ENABLED: bool = True
ENCRYPTION_ENABLED: bool = False   # enable at-rest encryption for metadata
DATA_RETENTION_DAYS: int = 365 * 7  # 7 years (HIPAA minimum)

# ──────────────────────────── Texture Analysis ────────────────────────────
GABOR_FREQUENCIES: tuple = (0.1, 0.2, 0.3, 0.4)
GABOR_ORIENTATIONS: int = 8       # number of Gabor filter orientations
TEXTURE_BLOCK_SIZE: int = 16      # block size for texture scoring

# ──────────────────────────── GAN Detection ────────────────────────────
GAN_FFT_THRESHOLD: float = 0.3    # spectral anomaly threshold
GAN_CHI2_ALPHA: float = 0.05      # chi-square significance level

# ──────────────────────────── Ollama Cloud API ────────────────────────────
# Key is loaded from .env file or system environment — NEVER hardcode here.
OLLAMA_API_KEY: str = os.environ.get("OLLAMA_API_KEY", "")
OLLAMA_BASE_URL: str = "https://ollama.com/v1"
OLLAMA_MODEL: str = "glm-5.1:cloud"   # GLM 5.1 cloud model on Ollama
OLLAMA_TIMEOUT: int = 60               # seconds (cloud can be slower)

# ──────────────────────────── Logging ────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
LOG_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a consistent format."""
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FMT,
    )


def ensure_dirs() -> None:
    """Create all required output directories."""
    for d in (RESULTS_WATERMARKED, RESULTS_ATTACKED, RESULTS_EXTRACTED, ARCHIVE_DIR):
        os.makedirs(d, exist_ok=True)
