"""
Simple File-Based Blockchain
-----------------------------
SHA-256 hash-chained ledger persisted to ``chain.json``.
Provides O(1) hash-indexed lookups, chain integrity verification,
and thread-safe read/write operations.
"""

import json
import time
import os
import hashlib
import logging
import threading
import fcntl
from typing import Any, Dict, List, Optional

from config import CHAIN_FILE

log = logging.getLogger(__name__)

# ──────────────────────────── Thread Safety ────────────────────────────
_lock = threading.Lock()

# ──────────────────────────── In-Memory Cache ────────────────────────────
_chain_cache: Optional[List[Dict[str, Any]]] = None
_hash_index: Dict[str, List[Dict[str, Any]]] = {}


# ──────────────────────────── Block ────────────────────────────

class SimpleBlock:
    """A single block in the chain."""

    __slots__ = ("index", "timestamp", "data", "previous_hash", "hash")

    def __init__(
        self,
        index: int,
        timestamp: float,
        data: Dict[str, Any],
        previous_hash: str,
    ) -> None:
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.compute_hash()

    def compute_hash(self) -> str:
        """Compute SHA-256 over the canonical JSON representation."""
        block_string = json.dumps(
            {
                "index": self.index,
                "timestamp": self.timestamp,
                "data": self.data,
                "previous_hash": self.previous_hash,
            },
            sort_keys=True,
        )
        return hashlib.sha256(block_string.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "hash": self.hash,
        }


# ──────────────────────────── Index Builder ────────────────────────────

def _build_index(chain: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Build a dict mapping image_hash → list of matching blocks."""
    idx: Dict[str, List[Dict[str, Any]]] = {}
    for block in chain:
        ih = block.get("data", {}).get("image_hash")
        if ih:
            idx.setdefault(ih, []).append(block)
    return idx


# ──────────────────────────── Chain I/O ────────────────────────────

def _read_file_locked() -> List[Dict[str, Any]]:
    """Read chain.json with advisory file lock."""
    if not os.path.exists(CHAIN_FILE):
        genesis = SimpleBlock(0, time.time(), {"note": "genesis"}, "0")
        chain = [genesis.to_dict()]
        _write_file_locked(chain)
        log.info("Created genesis block → %s", CHAIN_FILE)
        return chain

    with open(CHAIN_FILE, "r") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            chain = json.load(f)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return chain


def _write_file_locked(chain: List[Dict[str, Any]]) -> None:
    """Write chain.json atomically with advisory file lock."""
    tmp = CHAIN_FILE + ".tmp"
    with open(tmp, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            json.dump(chain, f, indent=2)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    os.replace(tmp, CHAIN_FILE)


def load_chain() -> List[Dict[str, Any]]:
    """
    Load the blockchain from disk (cached after first call).

    Returns
    -------
    list[dict]
        The full chain as a list of block dicts.
    """
    global _chain_cache, _hash_index
    with _lock:
        if _chain_cache is not None:
            return _chain_cache
        chain = _read_file_locked()
        _chain_cache = chain
        _hash_index = _build_index(chain)
        log.debug("Loaded chain: %d blocks, %d indexed hashes.",
                  len(chain), len(_hash_index))
        return chain


def _invalidate_cache() -> None:
    """Clear the in-memory cache so next load_chain() re-reads disk."""
    global _chain_cache, _hash_index
    _chain_cache = None
    _hash_index = {}


def save_chain(chain: List[Dict[str, Any]]) -> None:
    """Persist chain to disk and rebuild cache."""
    global _chain_cache, _hash_index
    with _lock:
        _write_file_locked(chain)
        _chain_cache = chain
        _hash_index = _build_index(chain)
    log.debug("Saved chain: %d blocks.", len(chain))


# ──────────────────────────── Integrity ────────────────────────────

def verify_chain(chain: Optional[List[Dict[str, Any]]] = None) -> bool:
    """
    Validate hash linkage and block hash integrity for every block.

    Returns True if the entire chain is valid, False otherwise.
    Logs each corruption found.
    """
    if chain is None:
        chain = load_chain()

    if not chain:
        log.error("Chain is empty.")
        return False

    valid = True
    for i, block in enumerate(chain):
        # Recompute hash
        expected = hashlib.sha256(
            json.dumps(
                {
                    "index": block["index"],
                    "timestamp": block["timestamp"],
                    "data": block["data"],
                    "previous_hash": block["previous_hash"],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

        if block["hash"] != expected:
            log.error(
                "Block %d: hash mismatch (stored=%s, computed=%s)",
                i, block["hash"][:16], expected[:16],
            )
            valid = False

        # Check linkage
        if i > 0 and block["previous_hash"] != chain[i - 1]["hash"]:
            log.error(
                "Block %d: previous_hash does not match block %d hash.",
                i, i - 1,
            )
            valid = False

    if valid:
        log.info("Chain integrity verified: %d blocks OK.", len(chain))
    return valid


# ──────────────────────────── Block Operations ────────────────────────────

def add_metadata_block(
    image_hash: str,
    watermark_hash: str,
    coords_hash: str,
    embed_timestamp: float,
) -> Dict[str, Any]:
    """
    Append a new watermarking-metadata block to the chain.

    Returns the new block as a dict.
    """
    with _lock:
        chain = _read_file_locked()
        last = chain[-1]
        block = SimpleBlock(
            index=last["index"] + 1,
            timestamp=time.time(),
            data={
                "image_hash": image_hash,
                "watermark_hash": watermark_hash,
                "coords_hash": coords_hash,
                "embed_timestamp": embed_timestamp,
            },
            previous_hash=last["hash"],
        )
        chain.append(block.to_dict())
        _write_file_locked(chain)

        # Update cache
        global _chain_cache, _hash_index
        _chain_cache = chain
        _hash_index = _build_index(chain)

    log.info("Added block #%d (image_hash=%s…).", block.index, image_hash[:12])
    return block.to_dict()


def find_by_image_hash(image_hash: str) -> Optional[Dict[str, Any]]:
    """Return the *first* block matching *image_hash*, or ``None``."""
    load_chain()  # ensure cache populated
    with _lock:
        matches = _hash_index.get(image_hash, [])
    return matches[0] if matches else None


def find_all_by_image_hash(image_hash: str) -> List[Dict[str, Any]]:
    """Return *all* blocks matching *image_hash* (audit trail)."""
    load_chain()
    with _lock:
        return list(_hash_index.get(image_hash, []))
