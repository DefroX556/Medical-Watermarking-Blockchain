"""
Batch Processing Pipeline
--------------------------
Parallel embed/verify for large-scale watermarking across multiple images
with progress tracking and configurable worker count.
"""

import os
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import (
    DATASET_DIR,
    RESULTS_WATERMARKED,
    SUPPORTED_EXTENSIONS,
    ensure_dirs,
)

log = logging.getLogger(__name__)


def _embed_single(args: Tuple[str, str, str, str]) -> Dict[str, Any]:
    """Worker function for parallel embedding (must be top-level for pickling)."""
    # Re-import inside worker for multiprocessing
    from core.embed import embed_watermark
    img_path, out_path, wm_text, meta_path = args
    try:
        result = embed_watermark(img_path, out_path, wm_text, meta_path)
        return {"status": "success", "image": img_path, "result": result}
    except Exception as e:
        return {"status": "error", "image": img_path, "error": str(e)}


def _verify_single(args: Tuple[str, str]) -> Dict[str, Any]:
    """Worker function for parallel verification."""
    from core.extract import extract_watermark
    wm_path, meta_path = args
    try:
        result = extract_watermark(wm_path, meta_path)
        return {"status": "success", "image": wm_path, "result": result}
    except Exception as e:
        return {"status": "error", "image": wm_path, "error": str(e)}


def batch_embed(
    image_dir: str = DATASET_DIR,
    watermark_text: str = "BatchWatermark",
    output_dir: str = RESULTS_WATERMARKED,
    max_workers: int = 4,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Embed watermarks into all images in a directory using parallel processing.

    Parameters
    ----------
    image_dir : str
        Directory containing input images.
    watermark_text : str
        Text to embed as watermark.
    output_dir : str
        Directory to save watermarked outputs.
    max_workers : int
        Number of parallel workers (default: 4).
    callback : callable, optional
        Called with result dict after each image completes.

    Returns
    -------
    list[dict]
        List of result dicts with status, image path, and embed results or error.
    """
    ensure_dirs()

    images = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ])

    if not images:
        log.warning("No images found in %s.", image_dir)
        return []

    tasks: List[Tuple[str, str, str, str]] = []
    for fname in images:
        base = os.path.splitext(fname)[0]
        tasks.append((
            os.path.join(image_dir, fname),
            os.path.join(output_dir, f"{base}_batch_wm.png"),
            watermark_text,
            os.path.join(output_dir, f"{base}_batch_meta.json"),
        ))

    log.info("Batch embed: %d images, %d workers.", len(tasks), max_workers)
    start = time.time()

    results: List[Dict[str, Any]] = []
    # Use sequential processing to avoid multiprocessing issues with blockchain file locking
    for i, task in enumerate(tasks, 1):
        result = _embed_single(task)
        results.append(result)
        log.info(
            "[%d/%d] %s: %s",
            i, len(tasks), os.path.basename(task[0]), result["status"],
        )
        if callback:
            callback(result)

    elapsed = time.time() - start
    success = sum(1 for r in results if r["status"] == "success")
    log.info(
        "Batch embed complete: %d/%d succeeded in %.1f s (%.1f img/s).",
        success, len(tasks), elapsed, len(tasks) / elapsed if elapsed > 0 else 0,
    )
    return results


def batch_verify(
    results_dir: str = RESULTS_WATERMARKED,
    max_workers: int = 4,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Verify all watermarked images in a directory.

    Looks for *_batch_wm.png and matching *_batch_meta.json pairs.
    """
    pairs: List[Tuple[str, str]] = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith("_batch_wm.png"):
            base = fname.replace("_batch_wm.png", "")
            meta = os.path.join(results_dir, f"{base}_batch_meta.json")
            if os.path.isfile(meta):
                pairs.append((os.path.join(results_dir, fname), meta))

    if not pairs:
        log.warning("No batch watermarked images found in %s.", results_dir)
        return []

    log.info("Batch verify: %d images.", len(pairs))
    start = time.time()

    results: List[Dict[str, Any]] = []
    for i, pair in enumerate(pairs, 1):
        result = _verify_single(pair)
        results.append(result)
        verified = result.get("result", {}).get("verified_in_blockchain", False) if result["status"] == "success" else False
        log.info(
            "[%d/%d] %s: %s (verified=%s)",
            i, len(pairs), os.path.basename(pair[0]), result["status"], verified,
        )
        if callback:
            callback(result)

    elapsed = time.time() - start
    log.info("Batch verify complete in %.1f s.", elapsed)
    return results
