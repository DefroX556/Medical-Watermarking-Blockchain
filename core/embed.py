"""
Watermark Embedding Module
--------------------------
Embeds both invisible (LSB) and visible (logo overlay) watermarks into images,
computes quality metrics (PSNR, SSIM), stores metadata in JSON, and records
hashes on a blockchain ledger.

Supports:
- Standard DBSCAN pixel selection
- Adaptive DBSCAN with auto-tuning (--adaptive)
- AI-based texture-intelligent placement (--ai-mode)
"""

import os
import time
import json
import logging
from typing import Any, Dict, Optional

import numpy as np
import cv2
from PIL import Image

from config import (
    LOGO_PATH,
    LOGO_SCALE,
    LOGO_OPACITY,
    LOGO_MARGIN,
    DEFAULT_EPS,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_EDGE_THRESH,
)
from utils import (
    text_to_bits,
    sha256_bytes,
    save_json,
    compute_psnr,
    compute_ssim_gray,
)
from dbscan import select_pixels_dbscan
from dbscan.adaptive import adaptive_select_pixels
from core.ai_analyzer import ai_rerank_positions
from core.texture_analyzer import intelligent_position_selection
from blockchain import add_metadata_block

log = logging.getLogger(__name__)


def embed_watermark(
    input_path: str,
    output_path: str,
    watermark_text: str,
    metadata_path: str,
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    edge_thresh: float = DEFAULT_EDGE_THRESH,
    adaptive: bool = False,
    ai_mode: bool = False,
) -> Dict[str, Any]:
    """
    Embed invisible and visible watermarks into an image.

    Parameters
    ----------
    input_path : str
        Path to the input image.
    output_path : str
        Path to save the watermarked output image.
    watermark_text : str
        Text string to embed as invisible watermark bits.
    metadata_path : str
        Path to save metadata JSON file.
    eps : float
        DBSCAN epsilon parameter for pixel clustering.
    min_samples : int
        DBSCAN minimum samples parameter.
    edge_thresh : float
        Edge threshold for stable pixel detection.
    adaptive : bool
        If True, use adaptive DBSCAN with auto-parameter tuning.
    ai_mode : bool
        If True, use texture-intelligent position selection.

    Returns
    -------
    dict
        PSNR, SSIM, hashes, metadata path, positions count, blockchain block,
        and optionally adaptive_info or texture_mode flag.

    Raises
    ------
    FileNotFoundError
        If the input image does not exist.
    ValueError
        If *watermark_text* is empty or too long for available positions.
    """

    # ── Validation ──
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")
    if not watermark_text:
        raise ValueError("watermark_text must be a non-empty string.")

    log.info("Loading image: %s", input_path)

    # ── Load original image ──
    pil_img = Image.open(input_path).convert("RGB")
    arr_rgb = np.array(pil_img)
    arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

    # ── Pixel Selection ──
    adaptive_info: Optional[Dict[str, Any]] = None

    if adaptive:
        # Feature 1: Adaptive DBSCAN with auto-tuning
        log.info("Using ADAPTIVE DBSCAN (auto-parameter tuning).")
        positions, adaptive_info = adaptive_select_pixels(gray)
        log.info(
            "Adaptive: modality=%s, eps=%.3f, clusters=%d",
            adaptive_info["modality"],
            adaptive_info["eps"],
            adaptive_info["cluster_count"],
        )
    else:
        # Standard DBSCAN
        positions = select_pixels_dbscan(
            gray, eps=eps, min_samples=min_samples, edge_thresh=edge_thresh
        )

    # Feature 3: AI-based intelligent re-ranking
    ai_info: Optional[Dict[str, Any]] = None
    if ai_mode:
        log.info("Using AI-POWERED position selection (Ollama Cloud GLM).")
        image_name = os.path.splitext(os.path.basename(input_path))[0]
        try:
            positions, ai_info = ai_rerank_positions(
                gray, positions, image_name=image_name,
            )
            log.info("AI analysis: status=%s, model=%s",
                     ai_info.get("status"), ai_info.get("model"))
        except ValueError as e:
            # Missing API key or config issue
            log.warning("AI config error: %s — falling back to local texture analysis.", e)
            positions = intelligent_position_selection(gray, positions)
            ai_info = {"status": "fallback_no_apikey", "model": "local_texture"}
        except Exception:
            log.warning("AI cloud analysis failed — falling back to local texture analysis.")
            positions = intelligent_position_selection(gray, positions)
            ai_info = {"status": "fallback_local", "model": "local_texture"}

    bits = text_to_bits(watermark_text)

    if len(bits) > len(positions):
        raise ValueError(
            f"Watermark too long: {len(bits)} bits required, "
            f"only {len(positions)} positions available."
        )

    out_bgr = arr_bgr.copy()

    # ── INVISIBLE LSB watermark (vectorized) ──
    n_bits = len(bits)
    pos_arr = np.array(positions[:n_bits], dtype=np.int32)
    rows = np.clip(pos_arr[:, 0], 0, out_bgr.shape[0] - 1)
    cols = np.clip(pos_arr[:, 1], 0, out_bgr.shape[1] - 1)
    bit_arr = np.array([int(b) for b in bits], dtype=np.uint8)

    # Clear LSB and set watermark bit — blue channel (index 0)
    blue_vals = out_bgr[rows, cols, 0].astype(np.uint8)
    blue_vals = (blue_vals & np.uint8(0xFE)) | bit_arr
    out_bgr[rows, cols, 0] = blue_vals

    used_positions = [[int(r), int(c)] for r, c in zip(rows, cols)]
    log.info("Embedded %d bits into %d positions.", n_bits, len(used_positions))

    # ── VISIBLE PNG LOGO overlay ──
    x1 = y1 = new_w = new_h = None

    if os.path.isfile(LOGO_PATH):
        logo = cv2.imread(LOGO_PATH, cv2.IMREAD_UNCHANGED)
        if logo is not None:
            new_w = int(out_bgr.shape[1] * LOGO_SCALE)
            new_h = int(logo.shape[0] * (new_w / logo.shape[1]))
            logo = cv2.resize(logo, (new_w, new_h))

            lx, ly = logo.shape[0], logo.shape[1]
            y1 = out_bgr.shape[0] - lx - LOGO_MARGIN
            y2 = out_bgr.shape[0] - LOGO_MARGIN
            x1 = out_bgr.shape[1] - ly - LOGO_MARGIN
            x2 = out_bgr.shape[1] - LOGO_MARGIN

            roi = out_bgr[y1:y2, x1:x2].astype(np.float64)

            if logo.shape[2] == 4:
                overlay = logo[:, :, :3].astype(np.float64)
                alpha = (logo[:, :, 3:].astype(np.float64) / 255.0) * LOGO_OPACITY
                out_bgr[y1:y2, x1:x2] = np.clip(
                    (1.0 - alpha) * roi + alpha * overlay, 0, 255
                ).astype(np.uint8)
            else:
                out_bgr[y1:y2, x1:x2] = np.clip(
                    (1.0 - LOGO_OPACITY) * roi + LOGO_OPACITY * logo.astype(np.float64),
                    0, 255,
                ).astype(np.uint8)
            log.info("Logo overlay applied at (%d, %d).", x1, y1)
        else:
            log.warning("Failed to decode logo at %s.", LOGO_PATH)
    else:
        log.warning("Logo file not found: %s. Skipping visible watermark.", LOGO_PATH)

    # ── Save output image ──
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(out_rgb).save(output_path)
    log.info("Watermarked image saved → %s", output_path)

    # ── PSNR & SSIM ──
    psnr_val = compute_psnr(arr_bgr, out_bgr)
    ssim_val = compute_ssim_gray(arr_bgr, out_bgr)
    log.info("Quality: PSNR=%.2f dB, SSIM=%.4f", psnr_val, ssim_val)

    # ── Metadata JSON ──
    meta: Dict[str, Any] = {
        "input_image": input_path,
        "output_image": output_path,
        "watermark_length_bits": len(bits),
        "positions": used_positions,
        "dbscan_params": {
            "eps": eps,
            "min_samples": min_samples,
            "edge_thresh": edge_thresh,
            "adaptive": adaptive,
            "ai_mode": ai_mode,
        },
        "psnr": psnr_val,
        "ssim": ssim_val,
    }

    if adaptive_info:
        meta["adaptive_info"] = adaptive_info

    if x1 is not None and y1 is not None:
        meta["logo_position"] = [x1, y1]
        meta["logo_size"] = [new_w, new_h]

    save_json(metadata_path, meta)

    # ── Blockchain storage ──
    with open(output_path, "rb") as f:
        img_bytes = f.read()

    image_hash = sha256_bytes(img_bytes)
    watermark_hash = sha256_bytes(watermark_text.encode())
    coords_hash = sha256_bytes(json.dumps(used_positions, sort_keys=True).encode())
    embed_ts = time.time()

    try:
        block = add_metadata_block(
            image_hash=image_hash,
            watermark_hash=watermark_hash,
            coords_hash=coords_hash,
            embed_timestamp=embed_ts,
        )
    except Exception:
        log.exception("Blockchain write failed — watermarked image is still saved.")
        block = None

    result: Dict[str, Any] = {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "image_hash": image_hash,
        "watermark_hash": watermark_hash,
        "coords_hash": coords_hash,
        "metadata_path": metadata_path,
        "positions_count": len(used_positions),
        "block": block,
    }

    if adaptive_info:
        result["adaptive_info"] = adaptive_info
    if ai_info:
        # Strip non-serializable numpy arrays
        ai_result_clean = {k: v for k, v in ai_info.items()
                           if not isinstance(v, np.ndarray)}
        result["ai_info"] = ai_result_clean

    return result
