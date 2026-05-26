#!/usr/bin/env python3
"""
evaluate.py — Robustness Evaluation Pipeline
=============================================
Embeds watermarks across all dataset images and measures resilience
against 9 attack types: compression, noise, cropping, rotation, scaling,
histogram equalization, contrast enhancement, median filtering, and
watermark removal attempt.

Usage
-----
    python evaluate.py                    # defaults
    python evaluate.py --watermark "XYZ"  # custom text
    python evaluate.py -v                 # verbose logging
"""

import os
import sys
import argparse
import logging
from typing import Any, Dict, Optional

import cv2
import numpy as np

from config import (
    DATASET_DIR,
    RESULTS_WATERMARKED,
    RESULTS_ATTACKED,
    LOGO_PATH,
    LOGO_MATCH_THRESHOLD,
    NOISE_SIGMA,
    CROP_RATIO,
    JPEG_QUALITY,
    ROTATION_ANGLE,
    SCALE_FACTOR,
    MEDIAN_KERNEL,
    SUPPORTED_EXTENSIONS,
    setup_logging,
    ensure_dirs,
)
from core.embed import embed_watermark
from utils import compute_psnr, compute_ssim_gray

log = logging.getLogger(__name__)


# ──────────────────────────── Attack Functions ────────────────────────────

def add_noise(img_bgr: np.ndarray, sigma: float = NOISE_SIGMA) -> np.ndarray:
    """Add Gaussian noise with standard deviation *sigma*."""
    noise = np.random.normal(0, sigma, img_bgr.shape).astype(np.float32)
    noisy = np.clip(img_bgr.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def crop_image(img_bgr: np.ndarray, ratio: float = CROP_RATIO) -> np.ndarray:
    """Crop to *ratio* of original height/width (top-left origin)."""
    h, w = img_bgr.shape[:2]
    return img_bgr[: int(h * ratio), : int(w * ratio)]


def jpeg_compress(
    img_bgr: np.ndarray, out_path: str, quality: int = JPEG_QUALITY
) -> np.ndarray:
    """Write as JPEG at *quality* then re-read to simulate lossy compression."""
    cv2.imwrite(out_path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imread(out_path)


def rotate_image(img_bgr: np.ndarray, angle: float = ROTATION_ANGLE) -> np.ndarray:
    """Rotate image by *angle* degrees and fill borders."""
    h, w = img_bgr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated


def scale_image(img_bgr: np.ndarray, factor: float = SCALE_FACTOR) -> np.ndarray:
    """Downscale by *factor* then upscale back to original size."""
    h, w = img_bgr.shape[:2]
    small = cv2.resize(img_bgr, (int(w * factor), int(h * factor)),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LANCZOS4)


def histogram_equalize(img_bgr: np.ndarray) -> np.ndarray:
    """Apply histogram equalization to each channel."""
    channels = cv2.split(img_bgr)
    eq_channels = [cv2.equalizeHist(ch) for ch in channels]
    return cv2.merge(eq_channels)


def contrast_enhance(img_bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = cv2.split(img_bgr)
    enhanced = [clahe.apply(ch) for ch in channels]
    return cv2.merge(enhanced)


def median_filter(img_bgr: np.ndarray, ksize: int = MEDIAN_KERNEL) -> np.ndarray:
    """Apply median filtering."""
    return cv2.medianBlur(img_bgr, ksize)


def watermark_removal_attempt(img_bgr: np.ndarray, meta_path: str) -> np.ndarray:
    """
    Attempt watermark removal by inpainting the logo region.
    Uses the logo position from metadata.
    """
    import json
    if not os.path.isfile(meta_path):
        return img_bgr

    meta = json.load(open(meta_path))
    if "logo_position" not in meta or "logo_size" not in meta:
        return img_bgr

    x, y = meta["logo_position"]
    w, h = meta["logo_size"]

    # Create inpainting mask over logo region
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    ah, aw = img_bgr.shape[:2]
    y_end = min(y + h, ah)
    x_end = min(x + w, aw)
    mask[y:y_end, x:x_end] = 255

    # Inpaint using Navier-Stokes method
    result = cv2.inpaint(img_bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
    return result


# ──────────────────────────── Detection ────────────────────────────

def detect_rate(
    attacked_path: str,
    meta_path: str,
    threshold: float = LOGO_MATCH_THRESHOLD,
) -> int:
    """
    Detect the visible watermark logo via template matching.

    Returns 100 (detected) or 0 (not detected).
    """
    import json

    if not os.path.isfile(attacked_path):
        log.warning("Attacked image missing: %s", attacked_path)
        return 0

    meta = json.load(open(meta_path))
    if "logo_position" not in meta or "logo_size" not in meta:
        log.warning("No logo metadata in %s — skipping detection.", meta_path)
        return 0

    x, y = meta["logo_position"]
    w, h = meta["logo_size"]

    attacked = cv2.imread(attacked_path)
    if attacked is None:
        log.warning("Cannot decode attacked image: %s", attacked_path)
        return 0

    # Guard against out-of-bounds after crop/resize
    ah, aw = attacked.shape[:2]
    if y + h > ah or x + w > aw:
        log.warning("Logo ROI out of bounds in %s.", attacked_path)
        return 0

    roi = attacked[y : y + h, x : x + w]

    if not os.path.isfile(LOGO_PATH):
        log.warning("Logo file missing: %s", LOGO_PATH)
        return 0

    logo = cv2.imread(LOGO_PATH, cv2.IMREAD_UNCHANGED)
    if logo is None:
        return 0

    logo = cv2.resize(logo, (w, h))
    res = cv2.matchTemplate(roi[:, :, :3], logo[:, :, :3], cv2.TM_CCOEFF_NORMED)
    score = float(res.max())
    log.debug("Template match score for %s: %.4f", attacked_path, score)
    return 100 if score > threshold else 0


# ──────────────────────────── Per-Image Evaluation ────────────────────────────

def evaluate_single_image(
    label: str, img_path: str, wm_text: str
) -> Optional[Dict[str, Any]]:
    """Run embedding + all attack evaluations on one image."""
    if not os.path.isfile(img_path):
        log.warning("%s: image not found at %s, skipping.", label, img_path)
        return None

    base = os.path.splitext(os.path.basename(img_path))[0]
    wm_out = os.path.join(RESULTS_WATERMARKED, f"{base}_wm.png")
    meta_out = os.path.join(RESULTS_WATERMARKED, f"{base}_meta.json")

    log.info("── Embedding: %s (%s) ──", label, img_path)
    embed_watermark(
        input_path=img_path,
        output_path=wm_out,
        watermark_text=wm_text,
        metadata_path=meta_out,
        eps=2.0,
        min_samples=3,
        edge_thresh=0.08,
    )

    orig = cv2.imread(img_path)
    wm_img = cv2.imread(wm_out)
    if orig is None or wm_img is None:
        log.error("Failed to read images for %s.", label)
        return None

    psnr_val = compute_psnr(orig, wm_img)
    ssim_val = compute_ssim_gray(orig, wm_img)

    attack_results: Dict[str, Dict[str, Any]] = {}

    # ── 1. Compression ──
    comp_path = os.path.join(RESULTS_ATTACKED, f"{base}_comp.jpg")
    comp_img = jpeg_compress(wm_img, comp_path)
    attack_results["Compression"] = {
        "psnr": compute_psnr(orig, comp_img),
        "detection_rate": detect_rate(comp_path, meta_out),
    }

    # ── 2. Noise ──
    noise_img = add_noise(wm_img)
    noise_path = os.path.join(RESULTS_ATTACKED, f"{base}_noise.png")
    cv2.imwrite(noise_path, noise_img)
    attack_results["Noise Addition"] = {
        "psnr": compute_psnr(orig, noise_img),
        "detection_rate": detect_rate(noise_path, meta_out),
    }

    # ── 3. Crop ──
    crop_img = crop_image(wm_img)
    crop_resized = cv2.resize(crop_img, (orig.shape[1], orig.shape[0]))
    crop_path = os.path.join(RESULTS_ATTACKED, f"{base}_crop.png")
    cv2.imwrite(crop_path, crop_resized)
    attack_results["Cropping"] = {
        "psnr": compute_psnr(orig, crop_resized),
        "detection_rate": detect_rate(crop_path, meta_out),
    }

    # ── 4. Rotation ──
    rot_img = rotate_image(wm_img)
    rot_path = os.path.join(RESULTS_ATTACKED, f"{base}_rot.png")
    cv2.imwrite(rot_path, rot_img)
    attack_results["Rotation"] = {
        "psnr": compute_psnr(orig, rot_img),
        "detection_rate": detect_rate(rot_path, meta_out),
    }

    # ── 5. Scaling ──
    scale_img = scale_image(wm_img)
    scale_path = os.path.join(RESULTS_ATTACKED, f"{base}_scale.png")
    cv2.imwrite(scale_path, scale_img)
    attack_results["Scaling"] = {
        "psnr": compute_psnr(orig, scale_img),
        "detection_rate": detect_rate(scale_path, meta_out),
    }

    # ── 6. Histogram Equalization ──
    hist_img = histogram_equalize(wm_img)
    hist_path = os.path.join(RESULTS_ATTACKED, f"{base}_hist.png")
    cv2.imwrite(hist_path, hist_img)
    attack_results["Histogram Eq"] = {
        "psnr": compute_psnr(orig, hist_img),
        "detection_rate": detect_rate(hist_path, meta_out),
    }

    # ── 7. Contrast Enhancement (CLAHE) ──
    clahe_img = contrast_enhance(wm_img)
    clahe_path = os.path.join(RESULTS_ATTACKED, f"{base}_clahe.png")
    cv2.imwrite(clahe_path, clahe_img)
    attack_results["Contrast (CLAHE)"] = {
        "psnr": compute_psnr(orig, clahe_img),
        "detection_rate": detect_rate(clahe_path, meta_out),
    }

    # ── 8. Median Filtering ──
    med_img = median_filter(wm_img)
    med_path = os.path.join(RESULTS_ATTACKED, f"{base}_median.png")
    cv2.imwrite(med_path, med_img)
    attack_results["Median Filter"] = {
        "psnr": compute_psnr(orig, med_img),
        "detection_rate": detect_rate(med_path, meta_out),
    }

    # ── 9. Watermark Removal ──
    removal_img = watermark_removal_attempt(wm_img, meta_out)
    removal_path = os.path.join(RESULTS_ATTACKED, f"{base}_removal.png")
    cv2.imwrite(removal_path, removal_img)
    attack_results["WM Removal"] = {
        "psnr": compute_psnr(orig, removal_img),
        "detection_rate": detect_rate(removal_path, meta_out),
    }

    log.info(
        "%s — PSNR=%.2f dB, SSIM=%.4f", label, psnr_val, ssim_val
    )
    return {
        "label": label,
        "psnr": psnr_val,
        "ssim": ssim_val,
        "attack_results": attack_results,
    }


# ──────────────────────────── CLI & Main ────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robustness evaluation pipeline.")
    p.add_argument("--dataset", "-d", type=str, default=DATASET_DIR,
                   help=f"Path to dataset directory (default: {DATASET_DIR}).")
    p.add_argument("--watermark", "-w", type=str, default="HospitalXYZ",
                   help="Watermark text to embed (default: HospitalXYZ).")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable debug logging.")
    return p.parse_args()


def print_summary(summary: Dict[str, Dict[str, Any]]) -> None:
    """Print a nicely formatted results table."""
    sep = "─" * 80
    print(f"\n{'═' * 80}")
    print("  RESULT SUMMARY")
    print(f"{'═' * 80}\n")

    # Quality table
    header = f"  {'Image':<20} {'PSNR (dB)':>12} {'SSIM':>10}"
    print(header)
    print(f"  {sep}")
    for label, info in summary.items():
        print(f"  {label:<20} {info['psnr']:>12.2f} {info['ssim']:>10.4f}")

    # Attack table
    print(f"\n  {'Image':<12} {'Attack':<20} {'PSNR (dB)':>12} {'Detection':>12}")
    print(f"  {sep}")
    for label, info in summary.items():
        for atk, vals in info["attack_results"].items():
            det = f"{vals['detection_rate']}%"
            print(f"  {label:<12} {atk:<20} {vals['psnr']:>12.2f} {det:>12}")
    print()


if __name__ == "__main__":
    cli = parse_args()
    setup_logging(level=logging.DEBUG if cli.verbose else logging.INFO)
    ensure_dirs()

    dataset_dir = cli.dataset
    if not os.path.isdir(dataset_dir):
        log.error("Dataset directory not found: %s", dataset_dir)
        sys.exit(1)

    image_set: Dict[str, str] = {}
    for f in sorted(os.listdir(dataset_dir)):
        if f.lower().endswith(SUPPORTED_EXTENSIONS):
            image_set[os.path.splitext(f)[0]] = os.path.join(dataset_dir, f)

    if not image_set:
        log.error("No images found in %s.", dataset_dir)
        sys.exit(1)

    summary: Dict[str, Dict[str, Any]] = {}
    for label, path in image_set.items():
        res = evaluate_single_image(label, path, cli.watermark)
        if res is not None:
            summary[label] = res

    print_summary(summary)
