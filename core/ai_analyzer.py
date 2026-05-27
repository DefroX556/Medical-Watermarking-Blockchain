"""
AI-Powered Image Analysis via Ollama Cloud
--------------------------------------------
Uses Ollama Cloud's OpenAI-compatible API with GLM models to perform
intelligent semantic analysis of medical images for optimal watermark
placement. The AI evaluates texture complexity, diagnostic importance
of regions, and recommends embedding zones that minimize perceptual impact.

Requires: Ollama API key (set in config or OLLAMA_API_KEY env var).
API Endpoint: https://ollama.com/v1/chat/completions (OpenAI-compatible)
"""

import os
import json
import base64
import logging
from typing import Any, Dict, List, Optional, Tuple

import requests
import numpy as np
import cv2

from config import (
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    TEXTURE_BLOCK_SIZE,
)

log = logging.getLogger(__name__)


def _get_api_key() -> str:
    """Resolve API key: env var takes priority over config."""
    key = os.environ.get("OLLAMA_API_KEY", OLLAMA_API_KEY)
    if not key:
        raise ValueError(
            "Ollama API key not set. Either set OLLAMA_API_KEY env var "
            "or configure OLLAMA_API_KEY in config.py"
        )
    return key


def _encode_image_thumbnail(gray: np.ndarray, max_size: int = 256) -> str:
    """
    Create a small thumbnail and encode as base64 JPEG.
    Keeps payload small and avoids sending full medical images to cloud.
    """
    h, w = gray.shape[:2]
    scale = min(max_size / h, max_size / w, 1.0)
    if scale < 1.0:
        thumb = cv2.resize(gray, (int(w * scale), int(h * scale)))
    else:
        thumb = gray

    if thumb.max() <= 1.0:
        thumb = (thumb * 255).astype(np.uint8)

    _, buffer = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buffer).decode("ascii")


def _compute_region_stats(gray: np.ndarray, block_size: int = TEXTURE_BLOCK_SIZE) -> List[Dict[str, Any]]:
    """Compute per-block statistics for AI analysis."""
    if gray.max() > 1.0:
        arr = gray / 255.0
    else:
        arr = gray

    h, w = arr.shape
    bh, bw = h // block_size, w // block_size
    regions = []

    for i in range(bh):
        for j in range(bw):
            block = arr[i * block_size:(i + 1) * block_size,
                       j * block_size:(j + 1) * block_size]
            regions.append({
                "row": i, "col": j,
                "mean": round(float(np.mean(block)), 4),
                "std": round(float(np.std(block)), 4),
                "min": round(float(np.min(block)), 4),
                "max": round(float(np.max(block)), 4),
            })

    return regions


def _call_ollama_api(
    messages: List[Dict[str, Any]],
    model: str = OLLAMA_MODEL,
    temperature: float = 0.3,
) -> Optional[str]:
    """
    Call Ollama Cloud's OpenAI-compatible chat completions API.

    Tries the ``openai`` Python SDK first (cleanest), then falls back
    to raw ``requests``.

    Returns the assistant's response text, or None on failure.
    """
    api_key = _get_api_key()

    # ── Method 1: OpenAI SDK (OpenAI-compatible) ──
    try:
        from openai import OpenAI

        log.info("Calling Ollama Cloud via OpenAI SDK (model=%s)...", model)
        client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key=api_key,
        )
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=1024,
        )
        msg = response.choices[0].message
        content = msg.content or ""

        # GLM-5.1 is a reasoning model — output may be in 'reasoning' field
        reasoning = getattr(msg, "reasoning", None) or ""
        if not content and reasoning:
            log.info("GLM reasoning model detected — using reasoning field.")
            content = reasoning

        log.info("Ollama SDK response received (%d chars).", len(content))
        return content

    except ImportError:
        log.debug("openai SDK not installed — falling back to raw HTTP.")
    except Exception as e:
        log.warning("Ollama SDK call failed: %s — trying raw HTTP.", e)

    # ── Method 2: Raw HTTP requests (fallback) ──
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1024,
    }

    try:
        log.info("Calling Ollama Cloud via HTTP (model=%s)...", model)
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )

        if resp.status_code != 200:
            log.error("Ollama API error: %d — %s", resp.status_code, resp.text[:500])
            return None

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        log.info("Ollama HTTP response received (%d chars).", len(content))
        return content

    except requests.exceptions.Timeout:
        log.error("Ollama API timeout after %d seconds.", OLLAMA_TIMEOUT)
        return None
    except requests.exceptions.ConnectionError:
        log.error("Ollama API connection failed. Check network.")
        return None
    except Exception:
        log.exception("Ollama API call failed unexpectedly.")
        return None


def _parse_ai_regions(response_text: str, bh: int, bw: int) -> List[Tuple[int, int]]:
    """
    Parse AI response to extract recommended block coordinates.

    Expected format: JSON array of {"row": int, "col": int, "score": float}
    Falls back to regex extraction if JSON parsing fails.
    """
    import re

    # Try to extract JSON from response — handle multiple formats
    # Find ALL JSON arrays in the text (greedy to get the longest one)
    json_matches = re.findall(r'\[[\s\S]*?\]', response_text)
    for jm in json_matches:
        try:
            blocks = json.loads(jm)
            if not isinstance(blocks, list) or len(blocks) == 0:
                continue
            result = []
            for b in blocks:
                if isinstance(b, dict):
                    r = int(b.get("row", b.get("r", -1)))
                    c = int(b.get("col", b.get("c", -1)))
                elif isinstance(b, (list, tuple)) and len(b) >= 2:
                    r, c = int(b[0]), int(b[1])
                else:
                    continue
                if 0 <= r < bh and 0 <= c < bw:
                    result.append((r, c))
            if result:
                log.info("AI recommended %d blocks for embedding.", len(result))
                return result
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
            continue

    # Fallback: extract any (row, col) pairs from text
    pairs = re.findall(r'row["\s:]+(\d+)[,\s]+col["\s:]+(\d+)', response_text, re.IGNORECASE)
    if pairs:
        result = [(int(r), int(c)) for r, c in pairs if int(r) < bh and int(c) < bw]
        if result:
            log.info("AI recommended %d blocks (regex parse).", len(result))
            return result

    log.warning("Could not parse AI response. Using all blocks.")
    return [(r, c) for r in range(bh) for c in range(bw)]


def ai_analyze_image(
    gray: np.ndarray,
    image_name: str = "medical_image",
    block_size: int = TEXTURE_BLOCK_SIZE,
) -> Dict[str, Any]:
    """
    Use Ollama Cloud GLM model to analyze a medical image and recommend
    optimal embedding regions.

    Parameters
    ----------
    gray : np.ndarray
        2-D grayscale image.
    image_name : str
        Name/label for logging.
    block_size : int
        Block size for region analysis.

    Returns
    -------
    dict
        ai_regions, ai_response, block_scores, model, status.
    """
    h, w = gray.shape[:2]
    bh, bw = h // block_size, w // block_size

    # Compute region statistics
    regions = _compute_region_stats(gray, block_size)

    # Keep only top-20 most textured blocks for the prompt (compact)
    sorted_regions = sorted(regions, key=lambda r: r["std"], reverse=True)
    top_blocks = sorted_regions[:20]

    # Prompt designed for GLM-5.1 reasoning model:
    # The model reasons in the 'reasoning' field, so we ask it to
    # produce a clear JSON array that our parser can extract.
    prompt = (
        f"Task: Select best blocks for watermark embedding in a medical image.\n"
        f"Image grid: {bw} columns x {bh} rows.\n"
        f"Below are 20 high-texture blocks with their statistics:\n\n"
        f"{json.dumps(top_blocks, indent=1)}\n\n"
        f"Selection criteria:\n"
        f"1. High std (good perceptual masking)\n"
        f"2. Edge/corner blocks preferred (non-diagnostic)\n"
        f"3. Avoid center blocks (likely diagnostic content)\n\n"
        f"Return a JSON array of selected blocks sorted by suitability:\n"
        f'[{{"row":0,"col":0,"score":0.9}}, ...]\n'
        f"Include at least 10 blocks. Output ONLY the JSON array, nothing else."
    )

    messages = [
        {"role": "user", "content": prompt},
    ]

    response_text = _call_ollama_api(messages)

    if not response_text or len(response_text.strip()) == 0:
        log.warning("AI returned empty response — falling back to texture-based selection.")
        # Fallback: use top-std blocks directly
        all_blocks = [(r["row"], r["col"]) for r in sorted_regions if r["std"] > 0.03]
        if not all_blocks:
            all_blocks = [(r["row"], r["col"]) for r in regions]
        return {
            "ai_regions": all_blocks,
            "ai_response": None,
            "model": OLLAMA_MODEL,
            "status": "fallback_empty",
        }

    ai_blocks = _parse_ai_regions(response_text, bh, bw)

    # Build block score map — all AI-recommended blocks get high score
    # Use equal weighting (1.0) for all recommended blocks to avoid
    # over-concentrating watermark in a single block
    block_scores = np.zeros((bh, bw), dtype=np.float64)
    for i, (r, c) in enumerate(ai_blocks):
        # Score decays gently: first block = 1.0, last = 0.5
        block_scores[r, c] = 1.0 - 0.5 * (i / max(len(ai_blocks) - 1, 1))

    return {
        "ai_regions": ai_blocks,
        "ai_response": response_text[:500],  # truncate for logging
        "block_scores": block_scores,
        "model": OLLAMA_MODEL,
        "status": "success",
        "blocks_recommended": len(ai_blocks),
    }


def ai_rerank_positions(
    gray: np.ndarray,
    edge_positions: List[Tuple[int, int]],
    image_name: str = "medical_image",
    block_size: int = TEXTURE_BLOCK_SIZE,
    top_n: int = 2000,
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """
    Re-rank DBSCAN positions using AI analysis.

    Positions in AI-recommended blocks get higher priority.
    Falls back to local texture analysis if AI is unavailable.

    Parameters
    ----------
    gray : np.ndarray
        2-D grayscale image.
    edge_positions : list[tuple[int, int]]
        Candidate positions from DBSCAN.
    image_name : str
        Image label for logging.
    block_size : int
        Block size for AI region analysis.
    top_n : int
        Maximum positions to return.

    Returns
    -------
    positions : list[tuple[int, int]]
        AI-reranked positions.
    ai_info : dict
        AI analysis metadata.
    """
    ai_result = ai_analyze_image(gray, image_name, block_size)

    h, w = gray.shape[:2]
    bh, bw = h // block_size, w // block_size

    # Score each position by whether it falls in an AI-recommended block
    ai_block_set = set(ai_result["ai_regions"])

    scored = []
    for r, c in edge_positions:
        r = max(0, min(r, h - 1))
        c = max(0, min(c, w - 1))
        bi = min(r // block_size, bh - 1)
        bj = min(c // block_size, bw - 1)

        if (bi, bj) in ai_block_set:
            # Position is in an AI-recommended block — boost score
            try:
                rank_idx = ai_result["ai_regions"].index((bi, bj))
            except ValueError:
                rank_idx = len(ai_result["ai_regions"])
            score = 1.0 - (rank_idx / max(len(ai_result["ai_regions"]), 1))
        else:
            score = 0.1  # low priority

        scored.append((score, r, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    positions = [(r, c) for _, r, c in scored[:top_n]]

    log.info(
        "AI re-ranking: %d/%d positions retained (AI status=%s, model=%s).",
        len(positions), len(edge_positions),
        ai_result["status"], ai_result["model"],
    )

    return positions, ai_result
