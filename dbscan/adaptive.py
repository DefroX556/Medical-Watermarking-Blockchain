"""
Adaptive DBSCAN Parameter Tuning
---------------------------------
Automatically determines optimal DBSCAN parameters (eps, min_samples) based on
image texture characteristics and modality classification.

Techniques
----------
- K-distance graph with knee/elbow detection for eps estimation
- Gabor filter bank energy for modality classification (MRI / CT / X-ray / Ultrasound)
- GLCM texture features for fine-grained parameter adjustment
- Silhouette score validation of clustering quality
"""

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
from skimage.filters import sobel, gabor
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

from config import (
    ADAPTIVE_K_NEIGHBORS,
    ADAPTIVE_EPS_RANGE,
    ADAPTIVE_MIN_SAMPLES_RANGE,
    MODALITY_PRESETS,
    DBSCAN_FALLBACK_COUNT,
    DBSCAN_MIN_CLUSTER_POSITIONS,
)

log = logging.getLogger(__name__)


# ──────────────────────────── Modality Detection ────────────────────────────

def _compute_gabor_energy(gray: np.ndarray) -> np.ndarray:
    """Compute mean Gabor filter energy across multiple frequencies/orientations."""
    energies = []
    for freq in (0.1, 0.2, 0.3, 0.4):
        for theta_idx in range(8):
            theta = theta_idx * np.pi / 8
            filt_real, filt_imag = gabor(gray, frequency=freq, theta=theta)
            energies.append(np.mean(filt_real ** 2 + filt_imag ** 2))
    return np.array(energies)


def _compute_glcm_features(gray: np.ndarray) -> Dict[str, float]:
    """Compute GLCM texture features: contrast, dissimilarity, homogeneity, energy, correlation."""
    # Quantize to 64 levels for GLCM
    quantized = (gray * 63).astype(np.uint8)
    glcm = graycomatrix(
        quantized,
        distances=[1, 3],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=64,
        symmetric=True,
        normed=True,
    )
    features = {}
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation"):
        features[prop] = float(np.mean(graycoprops(glcm, prop)))
    return features


def classify_modality(gray: np.ndarray) -> str:
    """
    Classify image modality based on texture features.

    Returns one of: 'mri', 'ct', 'xray', 'ultrasound', 'unknown'.
    """
    glcm_feat = _compute_glcm_features(gray)
    gabor_energy = _compute_gabor_energy(gray)
    mean_energy = float(np.mean(gabor_energy))
    std_energy = float(np.std(gabor_energy))

    contrast = glcm_feat["contrast"]
    homogeneity = glcm_feat["homogeneity"]
    energy = glcm_feat["energy"]

    # Heuristic classification based on texture characteristics:
    # MRI: high contrast, moderate homogeneity, complex texture
    # CT: moderate contrast, high homogeneity, structured
    # X-ray: low-moderate contrast, high homogeneity, smooth gradients
    # Ultrasound: very high contrast (speckle), low homogeneity

    if contrast > 15.0 and homogeneity < 0.3:
        modality = "ultrasound"
    elif contrast > 5.0 and mean_energy > 0.01:
        modality = "mri"
    elif homogeneity > 0.5 and energy > 0.01:
        modality = "xray"
    else:
        modality = "ct"

    log.info(
        "Modality classified: %s (contrast=%.2f, homogeneity=%.3f, "
        "energy=%.4f, gabor_mean=%.4f)",
        modality, contrast, homogeneity, energy, mean_energy,
    )
    return modality


# ──────────────────────────── K-Distance EPS Estimation ────────────────────────────

def _estimate_eps_kdist(coords: np.ndarray, k: int = ADAPTIVE_K_NEIGHBORS) -> float:
    """
    Estimate optimal eps via k-distance graph knee detection.

    Fits k-NN, sorts distances, and finds the point of maximum curvature.
    """
    if len(coords) < k + 1:
        return ADAPTIVE_EPS_RANGE[0]

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(coords)
    distances, _ = nn.kneighbors(coords)
    k_distances = np.sort(distances[:, -1])

    # Find knee: maximum second derivative
    if len(k_distances) < 3:
        return float(np.median(k_distances))

    d2 = np.diff(np.diff(k_distances))
    knee_idx = np.argmax(d2) + 1
    eps_est = float(k_distances[knee_idx])

    # Clamp to configured range
    eps_est = max(ADAPTIVE_EPS_RANGE[0], min(eps_est, ADAPTIVE_EPS_RANGE[1]))
    log.debug("K-distance estimated eps: %.3f (knee at index %d)", eps_est, knee_idx)
    return eps_est


# ──────────────────────────── Silhouette Validation ────────────────────────────

def _validate_with_silhouette(
    coords: np.ndarray, labels: np.ndarray
) -> float:
    """Compute silhouette score for clustering quality. Returns -1.0 on failure."""
    unique = set(labels) - {-1}
    if len(unique) < 2 or len(coords) < 3:
        return -1.0
    # Only score non-noise points
    mask = labels >= 0
    valid_coords = coords[mask]
    valid_labels = labels[mask]
    if len(set(valid_labels)) < 2 or len(valid_coords) < 3:
        return -1.0
    # Subsample for speed if too many points — keep coords+labels paired
    if len(valid_coords) > 5000:
        idx = np.random.choice(len(valid_coords), 5000, replace=False)
        valid_coords = valid_coords[idx]
        valid_labels = valid_labels[idx]
    try:
        return float(silhouette_score(valid_coords, valid_labels))
    except Exception:
        return -1.0


# ──────────────────────────── Main Adaptive Function ────────────────────────────

def adaptive_select_pixels(
    gray_image: np.ndarray,
    edge_thresh: Optional[float] = None,
) -> Tuple[List[Tuple[int, int]], Dict[str, any]]:
    """
    Automatically tune DBSCAN parameters and select stable embedding positions.

    Parameters
    ----------
    gray_image : np.ndarray
        2-D grayscale image (values in [0, 1] or [0, 255]).
    edge_thresh : float or None
        If None, determined from modality preset.

    Returns
    -------
    positions : list[tuple[int, int]]
        Selected (row, col) positions for embedding.
    info : dict
        Diagnostic info: modality, chosen params, silhouette score, cluster count.
    """
    if gray_image.ndim != 2 or gray_image.size == 0:
        raise ValueError("Expected a non-empty 2-D grayscale array.")

    arr = gray_image.astype(np.float64)
    if arr.max() > 1.0:
        arr /= 255.0

    # Step 1: Classify modality
    modality = classify_modality(arr)
    preset = MODALITY_PRESETS.get(modality, MODALITY_PRESETS["unknown"])

    if edge_thresh is None:
        edge_thresh = preset["edge_thresh"]

    # Step 2: Edge detection
    edges = sobel(arr)
    coords = np.column_stack(np.where(edges > edge_thresh))

    if coords.size == 0:
        raise ValueError(
            f"No edge pixels found with edge_thresh={edge_thresh}. "
            "Try lowering it."
        )

    # Step 3: Auto-tune eps via k-distance
    eps_auto = _estimate_eps_kdist(coords)

    # Step 4: Determine min_samples from preset (bounded by adaptive range)
    min_samples = max(
        ADAPTIVE_MIN_SAMPLES_RANGE[0],
        min(preset["min_samples"], ADAPTIVE_MIN_SAMPLES_RANGE[1]),
    )

    # Step 5: Run DBSCAN with auto-tuned params
    log.info(
        "Adaptive DBSCAN: eps=%.3f, min_samples=%d, edge_thresh=%.3f (modality=%s)",
        eps_auto, min_samples, edge_thresh, modality,
    )
    db = DBSCAN(eps=eps_auto, min_samples=min_samples).fit(coords)
    labels = db.labels_

    # Step 6: Validate clustering quality
    sil_score = _validate_with_silhouette(coords, labels)
    log.info("Silhouette score: %.4f", sil_score)

    # Step 7: Extract ALL pixels from valid clusters (not just centroids)
    unique_labels = set(labels) - {-1}
    selected: List[Tuple[int, int]] = []
    for lab in sorted(unique_labels):
        cluster_coords = coords[labels == lab]
        # Include all cluster member pixels — they are structurally stable
        for pt in cluster_coords:
            selected.append((int(pt[0]), int(pt[1])))

    # Fallback if too few clusters
    if len(selected) < DBSCAN_MIN_CLUSTER_POSITIONS:
        log.warning(
            "Adaptive DBSCAN: only %d positions. Falling back to top-%d edge pixels.",
            len(selected), DBSCAN_FALLBACK_COUNT,
        )
        flat_idx = np.argsort(edges.ravel())[::-1]
        h, w = edges.shape
        selected = []
        for idx in flat_idx[:DBSCAN_FALLBACK_COUNT]:
            r = int(idx // w)
            c = int(idx % w)
            selected.append((r, c))

    info = {
        "modality": modality,
        "eps": eps_auto,
        "min_samples": min_samples,
        "edge_thresh": edge_thresh,
        "cluster_count": len(unique_labels),
        "noise_points": int(np.sum(labels == -1)),
        "positions_count": len(selected),
        "silhouette_score": sil_score,
    }

    log.info(
        "Adaptive result: %d positions from %d clusters (modality=%s).",
        len(selected), len(unique_labels), modality,
    )
    return selected, info
