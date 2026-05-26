"""
Edge / Mobile Optimization Mode
---------------------------------
Lightweight processing pipeline for resource-constrained devices.
Supports downsampled embedding, reduced DBSCAN, and faster hashing.
"""

import logging
import hashlib
import time
import resource
from typing import Any, Dict, List, Tuple

import numpy as np
import cv2

from config import (
    PROCESSING_MODE,
    EDGE_DOWNSCALE,
    MOBILE_DOWNSCALE,
    MOBILE_DBSCAN_SUBSAMPLE,
    DBSCAN_FALLBACK_COUNT,
)

log = logging.getLogger(__name__)


def get_downscale_factor() -> float:
    """Return the downscale factor for the current processing mode."""
    if PROCESSING_MODE == "mobile":
        return MOBILE_DOWNSCALE
    elif PROCESSING_MODE == "edge":
        return EDGE_DOWNSCALE
    return 1.0  # full mode


def should_optimize() -> bool:
    """Check if current processing mode requires optimization."""
    return PROCESSING_MODE in ("edge", "mobile")


def downscale_image(image: np.ndarray, factor: float) -> np.ndarray:
    """Downscale an image by the given factor."""
    if factor >= 1.0:
        return image
    h, w = image.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def upscale_image(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Upscale image back to target shape (h, w)."""
    target_h, target_w = target_shape
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)


def subsample_edge_pixels(
    coords: np.ndarray, max_pixels: int = MOBILE_DBSCAN_SUBSAMPLE,
) -> np.ndarray:
    """
    Randomly subsample edge pixel coordinates for faster DBSCAN.
    Used in mobile mode to limit clustering input.
    """
    if len(coords) <= max_pixels:
        return coords
    indices = np.random.choice(len(coords), max_pixels, replace=False)
    log.debug("Subsampled edge pixels: %d → %d", len(coords), max_pixels)
    return coords[indices]


def fast_hash(data: bytes) -> str:
    """
    BLAKE2b hash — faster than SHA-256 especially on ARM/mobile.
    Falls back to SHA-256 if BLAKE2b is unavailable.
    """
    try:
        return hashlib.blake2b(data, digest_size=32).hexdigest()
    except AttributeError:
        return hashlib.sha256(data).hexdigest()


def get_memory_usage_mb() -> float:
    """Return current process memory usage in MB (Linux)."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024.0  # Convert KB to MB
    except Exception:
        return 0.0


def detect_system_capability() -> str:
    """
    Auto-detect processing mode based on available system resources.

    Returns 'full', 'edge', or 'mobile'.
    """
    import os

    try:
        cpu_count = os.cpu_count() or 1
        # Check available memory (Linux)
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    mem_kb = int(line.split()[1])
                    mem_gb = mem_kb / (1024 * 1024)
                    break
            else:
                mem_gb = 4.0  # default assumption
    except Exception:
        cpu_count = 2
        mem_gb = 4.0

    if mem_gb < 1.0 or cpu_count <= 1:
        mode = "mobile"
    elif mem_gb < 4.0 or cpu_count <= 2:
        mode = "edge"
    else:
        mode = "full"

    log.info(
        "System capability detected: %s (CPUs=%d, RAM=%.1f GB)",
        mode, cpu_count, mem_gb,
    )
    return mode


class PerformanceTracker:
    """
    Track timing and memory for each processing stage.

    Usage
    -----
    >>> tracker = PerformanceTracker()
    >>> tracker.start("embedding")
    >>> # ... do work ...
    >>> tracker.stop("embedding")
    >>> tracker.report()
    """

    def __init__(self) -> None:
        self._timers: Dict[str, float] = {}
        self._start_times: Dict[str, float] = {}
        self._memory: Dict[str, float] = {}

    def start(self, stage: str) -> None:
        self._start_times[stage] = time.time()
        self._memory[f"{stage}_start"] = get_memory_usage_mb()

    def stop(self, stage: str) -> float:
        if stage not in self._start_times:
            return 0.0
        elapsed = time.time() - self._start_times[stage]
        self._timers[stage] = elapsed
        self._memory[f"{stage}_end"] = get_memory_usage_mb()
        return elapsed

    def report(self) -> Dict[str, Any]:
        """Generate a performance report."""
        report: Dict[str, Any] = {
            "timings_ms": {k: v * 1000 for k, v in self._timers.items()},
            "memory_mb": self._memory,
            "total_ms": sum(v * 1000 for v in self._timers.values()),
        }
        log.info("Performance: total=%.0f ms, stages=%s",
                 report["total_ms"],
                 {k: f"{v:.0f}ms" for k, v in report["timings_ms"].items()})
        return report
