"""
GAN / Adversarial Attack Detection
------------------------------------
Detects image manipulation including deepfake-style alterations, adversarial
perturbations, and AI-generated content using frequency-domain analysis,
noise pattern fingerprinting, and statistical anomaly detection.
"""

import logging
from typing import Any, Dict

import numpy as np
import cv2
from scipy import stats as sp_stats

from config import GAN_FFT_THRESHOLD, GAN_CHI2_ALPHA

log = logging.getLogger(__name__)


def _compute_fft_spectrum(gray: np.ndarray) -> np.ndarray:
    """Compute the magnitude spectrum (log-scale, centered)."""
    f_transform = np.fft.fft2(gray.astype(np.float64))
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.log1p(np.abs(f_shift))
    return magnitude


def detect_spectral_anomaly(gray: np.ndarray, threshold: float = GAN_FFT_THRESHOLD) -> Dict[str, Any]:
    """
    Detect GAN artifacts via FFT spectral analysis.

    GAN-generated images often exhibit periodic artifacts visible as
    anomalous peaks in the frequency domain.

    Returns
    -------
    dict
        anomaly_score (0–1), is_suspicious (bool), peak_locations.
    """
    spectrum = _compute_fft_spectrum(gray)
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2

    # Exclude DC component (center 5x5)
    mask = np.ones_like(spectrum, dtype=bool)
    mask[cy - 2 : cy + 3, cx - 2 : cx + 3] = False

    masked_spectrum = spectrum[mask]
    mean_val = np.mean(masked_spectrum)
    std_val = np.std(masked_spectrum) + 1e-9

    # Find anomalous peaks (> 3 sigma from mean)
    z_scores = (spectrum - mean_val) / std_val
    z_scores[~mask] = 0  # ignore DC

    peak_mask = z_scores > 3.0
    peak_count = int(np.sum(peak_mask))
    peak_ratio = peak_count / spectrum.size

    anomaly_score = min(1.0, peak_ratio / threshold)

    log.info(
        "Spectral analysis: peaks=%d, ratio=%.6f, anomaly_score=%.4f",
        peak_count, peak_ratio, anomaly_score,
    )

    return {
        "anomaly_score": anomaly_score,
        "is_suspicious": anomaly_score > 0.5,
        "peak_count": peak_count,
        "peak_ratio": peak_ratio,
        "spectrum_mean": float(mean_val),
        "spectrum_std": float(std_val),
    }


def detect_noise_pattern(gray: np.ndarray) -> Dict[str, Any]:
    """
    PRNU-style noise pattern analysis for manipulation detection.

    Extracts the noise residual (image - denoised) and checks for
    statistical uniformity. Manipulated regions show different noise patterns.
    """
    denoised = cv2.GaussianBlur(gray, (5, 5), 1.0)
    noise_residual = gray.astype(np.float64) - denoised.astype(np.float64)

    # Split into quadrants and compare noise statistics
    h, w = noise_residual.shape
    quadrants = [
        noise_residual[: h // 2, : w // 2],       # top-left
        noise_residual[: h // 2, w // 2 :],        # top-right
        noise_residual[h // 2 :, : w // 2],        # bottom-left
        noise_residual[h // 2 :, w // 2 :],        # bottom-right
    ]

    q_means = [float(np.mean(q)) for q in quadrants]
    q_stds = [float(np.std(q)) for q in quadrants]

    # Variance ratio test — manipulated regions have different noise variance
    max_std = max(q_stds)
    min_std = min(q_stds) + 1e-9
    variance_ratio = max_std / min_std

    is_suspicious = variance_ratio > 2.0

    log.info(
        "Noise pattern: variance_ratio=%.3f, suspicious=%s",
        variance_ratio, is_suspicious,
    )

    return {
        "variance_ratio": variance_ratio,
        "quadrant_means": q_means,
        "quadrant_stds": q_stds,
        "is_suspicious": is_suspicious,
        "noise_mean": float(np.mean(noise_residual)),
        "noise_std": float(np.std(noise_residual)),
    }


def detect_lsb_anomaly(gray: np.ndarray, alpha: float = GAN_CHI2_ALPHA) -> Dict[str, Any]:
    """
    Chi-square test on the LSB plane to detect steganographic or
    adversarial tampering.

    A natural image has roughly uniform LSB distribution. Embedding
    or adversarial perturbation alters this distribution.
    """
    lsb_plane = gray.astype(np.uint8) & 1

    # Count pairs of adjacent pixels
    h, w = lsb_plane.shape
    pairs_h = lsb_plane[:, :-1] * 2 + lsb_plane[:, 1:]  # horizontal pairs
    # 4 possible pairs: 00, 01, 10, 11
    observed = np.zeros(4)
    for v in range(4):
        observed[v] = np.sum(pairs_h == v)

    total = np.sum(observed)
    expected = np.full(4, total / 4.0)

    chi2, p_value = sp_stats.chisquare(observed, expected)

    is_tampered = p_value < alpha

    log.info(
        "LSB chi-square: chi2=%.2f, p=%.6f, tampered=%s",
        chi2, p_value, is_tampered,
    )

    return {
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "is_tampered": is_tampered,
        "observed_distribution": observed.tolist(),
    }


def detect_manipulation(image_path: str) -> Dict[str, Any]:
    """
    Run full manipulation detection pipeline on an image.

    Combines spectral analysis, noise pattern analysis, and LSB anomaly
    detection into a single confidence report.

    Parameters
    ----------
    image_path : str
        Path to the image to analyze.

    Returns
    -------
    dict
        Overall confidence score, per-detector results, verdict.
    """
    import os
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to decode image: {image_path}")

    gray = img.astype(np.float64)
    if gray.max() > 1.0:
        gray_norm = gray / 255.0
    else:
        gray_norm = gray

    # Run all detectors
    spectral = detect_spectral_anomaly(gray_norm)
    noise = detect_noise_pattern(img)
    lsb = detect_lsb_anomaly(img)

    # Aggregate confidence: weighted combination
    score = (
        0.4 * spectral["anomaly_score"]
        + 0.3 * (1.0 if noise["is_suspicious"] else 0.0)
        + 0.3 * (1.0 if lsb["is_tampered"] else 0.0)
    )

    if score > 0.6:
        verdict = "HIGH_RISK"
    elif score > 0.3:
        verdict = "MEDIUM_RISK"
    else:
        verdict = "LOW_RISK"

    log.info(
        "Manipulation detection: verdict=%s, score=%.3f (%s)",
        verdict, score, image_path,
    )

    return {
        "image_path": image_path,
        "overall_score": score,
        "verdict": verdict,
        "spectral_analysis": spectral,
        "noise_pattern": noise,
        "lsb_anomaly": lsb,
    }
