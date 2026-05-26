#!/usr/bin/env python3
"""
run_demo.py — Single-Image Watermark Demo
==========================================
Embeds a watermark into one image and immediately extracts / verifies it.

Usage
-----
    python run_demo.py                           # auto-pick first dataset image
    python run_demo.py --image dataset/mri.png   # explicit image
    python run_demo.py --adaptive                # adaptive DBSCAN auto-tuning
    python run_demo.py --ai-mode                 # texture-intelligent placement
    python run_demo.py --adaptive --ai-mode      # combine both
    python run_demo.py -v                        # verbose / debug logging
"""

import os
import sys
import time
import argparse
import logging
import pprint

from config import (
    DATASET_DIR,
    RESULTS_WATERMARKED,
    RESULTS_EXTRACTED,
    SUPPORTED_EXTENSIONS,
    setup_logging,
    ensure_dirs,
)
from core.embed import embed_watermark
from core.extract import extract_watermark

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embed & extract a DBSCAN-guided watermark with blockchain verification.",
    )
    p.add_argument("--image", "-i", type=str, default=None,
                   help="Path to input image. Defaults to first image in dataset/.")
    p.add_argument("--watermark", "-w", type=str,
                   default="PatientID:12345;Date:2025-11-29",
                   help="Watermark text to embed.")
    p.add_argument("--eps", type=float, default=3.0,
                   help="DBSCAN epsilon (default: 3.0).")
    p.add_argument("--min-samples", type=int, default=5,
                   help="DBSCAN min_samples (default: 5).")
    p.add_argument("--edge-thresh", type=float, default=0.06,
                   help="Sobel edge threshold (default: 0.06).")
    p.add_argument("--adaptive", action="store_true",
                   help="Enable adaptive DBSCAN auto-parameter tuning.")
    p.add_argument("--ai-mode", action="store_true",
                   help="Enable AI texture-intelligent position selection.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable debug-level logging.")
    return p.parse_args()


def resolve_image(explicit: str | None) -> str:
    """Return a valid image path — either the user-supplied one or auto-detected."""
    if explicit:
        if not os.path.isfile(explicit):
            log.error("Image not found: %s", explicit)
            sys.exit(1)
        return explicit

    candidates = [
        f for f in sorted(os.listdir(DATASET_DIR))
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]
    if not candidates:
        log.error("No images found in %s. Add at least one.", DATASET_DIR)
        sys.exit(1)

    path = os.path.join(DATASET_DIR, candidates[0])
    log.info("Auto-selected image: %s", path)
    return path


def demo(args: argparse.Namespace) -> None:
    ensure_dirs()
    inp = resolve_image(args.image)
    base = os.path.splitext(os.path.basename(inp))[0]

    # Tag output filenames by mode
    mode_tag = ""
    if args.adaptive:
        mode_tag += "_adaptive"
    if args.ai_mode:
        mode_tag += "_ai"

    out = os.path.join(RESULTS_WATERMARKED, f"{base}{mode_tag}_watermarked.png")
    meta = os.path.join(RESULTS_WATERMARKED, f"{base}{mode_tag}_meta.json")

    # ── Embedding ──
    log.info("═" * 60)
    log.info("EMBEDDING%s", " (ADAPTIVE)" if args.adaptive else (" (AI-MODE)" if args.ai_mode else ""))
    log.info("═" * 60)

    t0 = time.time()
    embed_res = embed_watermark(
        inp, out, args.watermark, meta,
        eps=args.eps, min_samples=args.min_samples, edge_thresh=args.edge_thresh,
        adaptive=args.adaptive, ai_mode=args.ai_mode,
    )
    embed_ms = (time.time() - t0) * 1000

    log.info("PSNR: %.2f dB | SSIM: %.4f | Latency: %.0f ms",
             embed_res["psnr"], embed_res["ssim"], embed_ms)

    if "adaptive_info" in embed_res:
        ai = embed_res["adaptive_info"]
        log.info("Adaptive: modality=%s, eps=%.3f, clusters=%d, silhouette=%.4f",
                 ai["modality"], ai["eps"], ai["cluster_count"], ai["silhouette_score"])

    if "ai_info" in embed_res:
        ai = embed_res["ai_info"]
        log.info("AI Cloud: model=%s, status=%s, blocks=%s",
                 ai.get("model", "N/A"), ai.get("status", "N/A"),
                 ai.get("blocks_recommended", "N/A"))

    # ── Extraction & Verification ──
    log.info("═" * 60)
    log.info("EXTRACTION & VERIFICATION")
    log.info("═" * 60)

    t0 = time.time()
    extract_res = extract_watermark(out, meta)
    extract_ms = (time.time() - t0) * 1000

    log.info("Verified: %s | Retrieve latency: %.1f ms | Total: %.0f ms",
             extract_res["verified_in_blockchain"],
             extract_res["retrieve_latency_ms"],
             extract_ms)

    if args.verbose:
        log.debug("Embed result:\n%s", pprint.pformat(embed_res))
        log.debug("Extract result:\n%s", pprint.pformat(extract_res))


if __name__ == "__main__":
    cli_args = parse_args()
    setup_logging(level=logging.DEBUG if cli_args.verbose else logging.INFO)
    try:
        demo(cli_args)
    except Exception:
        log.exception("Demo failed.")
        sys.exit(1)
