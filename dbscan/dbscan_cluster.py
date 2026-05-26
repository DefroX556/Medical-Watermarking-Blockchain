"""
DBSCAN Stable-Pixel Selection
------------------------------
Uses Sobel edge detection followed by DBSCAN clustering to identify
perceptually stable pixel positions suitable for LSB watermark embedding.
"""

import logging
from typing import List, Tuple

import numpy as np
from skimage.filters import sobel
from sklearn.cluster import DBSCAN

from config import (
    DEFAULT_EPS,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_EDGE_THRESH,
    DBSCAN_FALLBACK_COUNT,
    DBSCAN_MIN_CLUSTER_POSITIONS,
)

log = logging.getLogger(__name__)


def select_pixels_dbscan(
    gray_image: np.ndarray,
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    edge_thresh: float = DEFAULT_EDGE_THRESH,
) -> List[Tuple[int, int]]:
    """
    Select stable embedding positions via DBSCAN clustering of edge pixels.

    Parameters
    ----------
    gray_image : np.ndarray
        2-D grayscale image (values in [0, 1] or [0, 255]).
    eps : float
        DBSCAN neighbourhood radius.
    min_samples : int
        DBSCAN minimum cluster membership.
    edge_thresh : float
        Sobel magnitude threshold — only pixels above this are considered.

    Returns
    -------
    list[tuple[int, int]]
        (row, col) positions for watermark embedding.

    Raises
    ------
    ValueError
        If the image is empty or no edge pixels survive thresholding.
    """
    if gray_image.ndim != 2 or gray_image.size == 0:
        raise ValueError("Expected a non-empty 2-D grayscale array.")

    arr = gray_image.astype(np.float64)
    if arr.max() > 1.0:
        arr /= 255.0

    edges = sobel(arr)
    coords = np.column_stack(np.where(edges > edge_thresh))

    if coords.size == 0:
        raise ValueError(
            f"No edge pixels found with edge_thresh={edge_thresh}. "
            "Try lowering it."
        )

    log.debug(
        "Edge pixels found: %d (eps=%.2f, min_samples=%d)",
        len(coords),
        eps,
        min_samples,
    )

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_

    unique_labels = set(labels) - {-1}
    log.debug("DBSCAN clusters: %d (noise points: %d)",
              len(unique_labels), int(np.sum(labels == -1)))

    selected: List[Tuple[int, int]] = []
    for lab in sorted(unique_labels):
        cluster_coords = coords[labels == lab]
        centroid = np.mean(cluster_coords, axis=0).round().astype(int)
        selected.append((int(centroid[0]), int(centroid[1])))

    # Fallback: if DBSCAN produced too few clusters, pick top edge pixels
    if len(selected) < DBSCAN_MIN_CLUSTER_POSITIONS:
        log.warning(
            "DBSCAN returned only %d positions (min %d). "
            "Falling back to top-%d edge pixels.",
            len(selected),
            DBSCAN_MIN_CLUSTER_POSITIONS,
            DBSCAN_FALLBACK_COUNT,
        )
        flat_idx = np.argsort(edges.ravel())[::-1]
        h, w = edges.shape
        extra: List[Tuple[int, int]] = []
        for idx in flat_idx[:DBSCAN_FALLBACK_COUNT]:
            r = int(idx // w)
            c = int(idx % w)
            extra.append((r, c))
        return extra

    log.info("Selected %d stable positions via DBSCAN.", len(selected))
    return selected
