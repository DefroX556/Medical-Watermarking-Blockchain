"""
Watermark Extraction & Verification Module
-------------------------------------------
Reads LSB bits from stored embedding positions, reconstructs the watermark
text, and verifies integrity against the blockchain ledger.
"""

import time
import json
import logging
from typing import Any, Dict, Optional

import numpy as np
import cv2
from PIL import Image

from utils import bits_to_text, sha256_bytes, load_json
from blockchain import find_by_image_hash

log = logging.getLogger(__name__)


def extract_watermark(
    watermarked_path: str,
    metadata_path: str,
) -> Dict[str, Any]:
    """
    Extract watermark text and verify against the blockchain.

    Parameters
    ----------
    watermarked_path : str
        Path to the watermarked image.
    metadata_path : str
        Path to the metadata JSON created during embedding.

    Returns
    -------
    dict
        watermark_text, hashes, verification flag, matched block, retrieval latency.

    Raises
    ------
    FileNotFoundError
        If the watermarked image or metadata file is missing.
    """
    import os

    if not os.path.isfile(watermarked_path):
        raise FileNotFoundError(f"Watermarked image not found: {watermarked_path}")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    log.info("Extracting watermark from: %s", watermarked_path)

    # ── Load image ──
    img = Image.open(watermarked_path).convert("RGB")
    arr_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # ── Load metadata ──
    meta = load_json(metadata_path)
    positions = meta["positions"]
    bits_len = meta["watermark_length_bits"]

    # ── Vectorized LSB extraction ──
    h, w = arr_bgr.shape[:2]
    pos_arr = np.array(positions[:bits_len], dtype=np.int32)
    rows = np.clip(pos_arr[:, 0], 0, h - 1)
    cols = np.clip(pos_arr[:, 1], 0, w - 1)

    blue_vals = arr_bgr[rows, cols, 0].astype(np.uint8)
    bit_arr = blue_vals & np.uint8(1)
    bits = "".join(str(b) for b in bit_arr)

    watermark_text = bits_to_text(bits)
    log.info("Extracted watermark: %r (%d bits)", watermark_text, bits_len)

    # ── Hash computation ──
    with open(watermarked_path, "rb") as f:
        img_bytes = f.read()

    image_hash = sha256_bytes(img_bytes)
    watermark_hash = sha256_bytes(watermark_text.encode("utf-8"))
    coords_hash = sha256_bytes(json.dumps(positions, sort_keys=True).encode())

    # ── Blockchain verification ──
    start = time.time()
    block: Optional[Dict[str, Any]] = find_by_image_hash(image_hash)
    retrieve_latency_ms = (time.time() - start) * 1000.0

    verified = False
    if block is not None:
        data = block.get("data", {})
        if (
            data.get("image_hash") == image_hash
            and data.get("watermark_hash") == watermark_hash
            and data.get("coords_hash") == coords_hash
        ):
            verified = True

    log.info(
        "Blockchain verification: %s (latency=%.1f ms)",
        "PASSED" if verified else "FAILED",
        retrieve_latency_ms,
    )

    return {
        "watermark_text": watermark_text,
        "image_hash": image_hash,
        "watermark_hash": watermark_hash,
        "coords_hash": coords_hash,
        "verified_in_blockchain": verified,
        "block": block,
        "retrieve_latency_ms": retrieve_latency_ms,
    }
