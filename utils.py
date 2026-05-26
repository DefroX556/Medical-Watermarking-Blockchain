"""
Utility Helpers
---------------
Hashing, bit-conversion, JSON I/O, and image quality metrics.
"""

import json
import hashlib
import logging
from typing import Any

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

log = logging.getLogger(__name__)


# ──────────────────────────── Hashing ────────────────────────────

def sha256_bytes(b: bytes) -> str:
    """Return the SHA-256 hex digest of *b*."""
    return hashlib.sha256(b).hexdigest()


# ──────────────────────────── Bit Conversion ────────────────────────────

def text_to_bits(s: str) -> str:
    """Convert a Unicode string to its binary (8-bit per char) representation."""
    return "".join(f"{ord(c):08b}" for c in s)


def bits_to_text(bstr: str) -> str:
    """Convert a binary string back to text (8-bit chunks)."""
    chars = [bstr[i : i + 8] for i in range(0, len(bstr), 8)]
    return "".join(chr(int(c, 2)) for c in chars if len(c) == 8)


# ──────────────────────────── JSON I/O ────────────────────────────

def save_json(path: str, obj: Any) -> None:
    """Atomically write *obj* as pretty-printed JSON to *path*."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    # Atomic rename — prevents partial writes on crash
    import os
    os.replace(tmp, path)
    log.debug("Saved JSON → %s", path)


def load_json(path: str) -> Any:
    """Load and return a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


# ──────────────────────────── Image Quality Metrics ────────────────────────────

def compute_psnr(orig_arr: np.ndarray, wm_arr: np.ndarray) -> float:
    """
    Compute PSNR (dB) between two uint8 images.

    If shapes differ, *wm_arr* is resized to match *orig_arr* and a warning is
    emitted.
    """
    if orig_arr.shape != wm_arr.shape:
        log.warning(
            "Shape mismatch for PSNR: %s vs %s — resizing second image",
            orig_arr.shape,
            wm_arr.shape,
        )
        wm_arr = cv2.resize(wm_arr, (orig_arr.shape[1], orig_arr.shape[0]))

    return float(cv2.PSNR(orig_arr.astype(np.uint8), wm_arr.astype(np.uint8)))


def compute_ssim_gray(orig_arr: np.ndarray, wm_arr: np.ndarray) -> float:
    """
    Compute SSIM on grayscale versions of two images.

    Handles shape mismatches by resizing the second image.
    """
    if orig_arr.shape != wm_arr.shape:
        log.warning(
            "Shape mismatch for SSIM: %s vs %s — resizing second image",
            orig_arr.shape,
            wm_arr.shape,
        )
        wm_arr = cv2.resize(wm_arr, (orig_arr.shape[1], orig_arr.shape[0]))

    og = cv2.cvtColor(orig_arr, cv2.COLOR_BGR2GRAY) if orig_arr.ndim == 3 else orig_arr.copy()
    wg = cv2.cvtColor(wm_arr, cv2.COLOR_BGR2GRAY) if wm_arr.ndim == 3 else wm_arr.copy()
    return float(ssim(og, wg, data_range=255))
