"""
Texture-Based Intelligent Watermark Placement
----------------------------------------------
Uses Gabor filter banks and GLCM texture features to score image regions
by embedding suitability — high-texture, low-perceptual-importance areas
are preferred for watermark placement.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
from skimage.filters import gabor, sobel
from skimage.feature import graycomatrix, graycoprops

from config import (
    GABOR_FREQUENCIES,
    GABOR_ORIENTATIONS,
    TEXTURE_BLOCK_SIZE,
)

log = logging.getLogger(__name__)


def compute_gabor_feature_map(gray: np.ndarray) -> np.ndarray:
    """
    Compute a per-pixel Gabor energy map by averaging responses across
    all frequency/orientation combinations.

    Parameters
    ----------
    gray : np.ndarray
        2-D grayscale image normalized to [0, 1].

    Returns
    -------
    np.ndarray
        Energy map (same shape as input) — higher values = more texture.
    """
    energy_sum = np.zeros_like(gray, dtype=np.float64)

    for freq in GABOR_FREQUENCIES:
        for i in range(GABOR_ORIENTATIONS):
            theta = i * np.pi / GABOR_ORIENTATIONS
            filt_real, filt_imag = gabor(gray, frequency=freq, theta=theta)
            energy_sum += np.sqrt(filt_real ** 2 + filt_imag ** 2)

    count = len(GABOR_FREQUENCIES) * GABOR_ORIENTATIONS
    energy_map = energy_sum / count

    log.debug(
        "Gabor energy map: mean=%.4f, max=%.4f, min=%.4f",
        np.mean(energy_map), np.max(energy_map), np.min(energy_map),
    )
    return energy_map


def compute_block_texture_scores(
    gray: np.ndarray, block_size: int = TEXTURE_BLOCK_SIZE,
) -> np.ndarray:
    """
    Divide the image into blocks and compute a texture suitability score
    for each block using GLCM features.

    Score = contrast × dissimilarity / (homogeneity + 1e-6)
    Higher score = better for embedding (complex texture, less noticeable).

    Returns
    -------
    np.ndarray
        2-D array of scores, one per block.
    """
    h, w = gray.shape
    bh = h // block_size
    bw = w // block_size
    scores = np.zeros((bh, bw), dtype=np.float64)

    quantized = (gray * 63).astype(np.uint8)

    for i in range(bh):
        for j in range(bw):
            block = quantized[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size,
            ]
            if block.size == 0:
                continue
            try:
                glcm = graycomatrix(
                    block, distances=[1], angles=[0], levels=64,
                    symmetric=True, normed=True,
                )
                contrast = float(graycoprops(glcm, "contrast")[0, 0])
                dissimilarity = float(graycoprops(glcm, "dissimilarity")[0, 0])
                homogeneity = float(graycoprops(glcm, "homogeneity")[0, 0])
                scores[i, j] = contrast * dissimilarity / (homogeneity + 1e-6)
            except Exception:
                scores[i, j] = 0.0

    log.debug("Block texture scores: shape=%s, mean=%.4f", scores.shape, np.mean(scores))
    return scores


def intelligent_position_selection(
    gray: np.ndarray,
    edge_positions: List[Tuple[int, int]],
    top_n: int = 2000,
) -> List[Tuple[int, int]]:
    """
    Re-rank DBSCAN/edge positions by texture intelligence.

    Positions in high-texture regions are preferred — watermark changes
    are less perceptible there.

    Parameters
    ----------
    gray : np.ndarray
        2-D grayscale image normalized to [0, 1].
    edge_positions : list[tuple[int, int]]
        Candidate (row, col) positions from DBSCAN.
    top_n : int
        Maximum positions to return.

    Returns
    -------
    list[tuple[int, int]]
        Re-ranked positions, best-first.
    """
    if not edge_positions:
        raise ValueError("No candidate positions to score.")

    # Compute Gabor energy map for per-pixel scoring
    energy_map = compute_gabor_feature_map(gray)

    # Compute block-level texture scores
    block_scores = compute_block_texture_scores(gray)
    bs = TEXTURE_BLOCK_SIZE
    bh, bw = block_scores.shape

    # Score each position: combine pixel-level energy + block-level texture
    scored: List[Tuple[float, int, int]] = []
    h, w = gray.shape

    for r, c in edge_positions:
        r = max(0, min(r, h - 1))
        c = max(0, min(c, w - 1))

        pixel_energy = energy_map[r, c]

        bi = min(r // bs, bh - 1)
        bj = min(c // bs, bw - 1)
        block_score = block_scores[bi, bj]

        # Combined score: weighted sum
        combined = 0.6 * pixel_energy + 0.4 * (block_score / (np.max(block_scores) + 1e-9))
        scored.append((combined, r, c))

    # Sort descending by score (highest texture first)
    scored.sort(key=lambda x: x[0], reverse=True)

    result = [(r, c) for _, r, c in scored[:top_n]]
    log.info(
        "Texture-intelligent selection: %d/%d positions retained (best score=%.4f).",
        len(result), len(edge_positions), scored[0][0] if scored else 0.0,
    )
    return result
