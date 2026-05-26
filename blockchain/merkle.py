"""
Merkle Tree Implementation
--------------------------
Binary hash tree for efficient batch verification of blockchain blocks.
Supports proof generation and verification for individual blocks.
"""

import hashlib
import logging
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)


class MerkleTree:
    """
    A binary Merkle tree built from a list of data hashes.

    Attributes
    ----------
    root : str or None
        The Merkle root hash. None if tree is empty.
    leaves : list[str]
        The original leaf hashes.
    """

    def __init__(self, hashes: List[str]) -> None:
        self.leaves = list(hashes)
        self._tree: List[List[str]] = []
        self.root: Optional[str] = None
        if self.leaves:
            self._build()

    @staticmethod
    def _hash_pair(a: str, b: str) -> str:
        """Hash two hex strings together."""
        combined = (a + b).encode()
        return hashlib.sha256(combined).hexdigest()

    def _build(self) -> None:
        """Build the tree bottom-up."""
        current_level = self.leaves[:]
        self._tree = [current_level[:]]

        while len(current_level) > 1:
            next_level: List[str] = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                next_level.append(self._hash_pair(left, right))
            self._tree.append(next_level)
            current_level = next_level

        self.root = current_level[0] if current_level else None
        log.debug("Merkle tree built: %d leaves → root=%s…", len(self.leaves), self.root[:12] if self.root else "None")

    def get_proof(self, index: int) -> List[Tuple[str, str]]:
        """
        Generate a Merkle proof for the leaf at *index*.

        Returns a list of (hash, side) tuples where side is 'L' or 'R'.
        """
        if index < 0 or index >= len(self.leaves):
            raise IndexError(f"Leaf index {index} out of range [0, {len(self.leaves)})")

        proof: List[Tuple[str, str]] = []
        idx = index

        for level in self._tree[:-1]:  # skip root level
            if idx % 2 == 0:
                sibling_idx = idx + 1
                side = "R"
            else:
                sibling_idx = idx - 1
                side = "L"

            if sibling_idx < len(level):
                proof.append((level[sibling_idx], side))
            else:
                proof.append((level[idx], "R"))  # odd leaf, duplicate self

            idx //= 2

        return proof

    @classmethod
    def verify_proof(
        cls, leaf_hash: str, proof: List[Tuple[str, str]], root: str
    ) -> bool:
        """Verify a Merkle proof against a known root."""
        current = leaf_hash
        for sibling, side in proof:
            if side == "L":
                current = cls._hash_pair(sibling, current)
            else:
                current = cls._hash_pair(current, sibling)
        return current == root

    def __len__(self) -> int:
        return len(self.leaves)

    def __repr__(self) -> str:
        return f"MerkleTree(leaves={len(self.leaves)}, root={self.root[:16] if self.root else None}…)"


def compute_merkle_root(hashes: List[str]) -> Optional[str]:
    """Convenience function: compute the Merkle root of a list of hashes."""
    if not hashes:
        return None
    tree = MerkleTree(hashes)
    return tree.root
