# core/__init__.py
from .embed import embed_watermark
from .extract import extract_watermark
from .texture_analyzer import intelligent_position_selection
from .attack_detector import detect_manipulation
from .batch import batch_embed, batch_verify
from .edge_mode import PerformanceTracker, detect_system_capability
