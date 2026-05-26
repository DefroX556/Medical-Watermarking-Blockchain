"""
Metadata Encryption at Rest
-----------------------------
AES-256-GCM encryption for metadata JSON files to satisfy
HIPAA and GDPR data-at-rest requirements.

Uses the ``cryptography`` package if available, otherwise provides
a fallback XOR-based obfuscation (NOT cryptographically secure — for
demonstration only).
"""

import os
import json
import hashlib
import hmac
import logging
import base64
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger(__name__)

# Try to use the cryptography library; fall back gracefully
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False
    log.warning(
        "cryptography package not installed — encryption will use "
        "XOR obfuscation (NOT secure). Install: pip install cryptography"
    )


class MetadataEncryptor:
    """
    Encrypt/decrypt metadata JSON files using AES-256-GCM.

    Parameters
    ----------
    key : bytes or None
        32-byte encryption key. If None, a deterministic key is derived
        from *passphrase*.
    passphrase : str
        Used to derive a key via PBKDF2 if *key* is not provided.
    """

    def __init__(
        self,
        key: Optional[bytes] = None,
        passphrase: str = "medical-watermark-default-key",
    ) -> None:
        if key and len(key) == 32:
            self._key = key
        else:
            # Derive key from passphrase via PBKDF2
            self._key = hashlib.pbkdf2_hmac(
                "sha256",
                passphrase.encode(),
                b"watermark-salt-v1",
                iterations=100_000,
            )
        log.debug("Encryptor initialized (crypto=%s).", _HAS_CRYPTO)

    def encrypt_json(self, data: Dict[str, Any], output_path: str) -> str:
        """
        Encrypt a dict as JSON and write to *output_path*.

        File format: base64(nonce + ciphertext + tag)
        """
        plaintext = json.dumps(data, indent=2).encode("utf-8")

        if _HAS_CRYPTO:
            nonce = os.urandom(12)
            aesgcm = AESGCM(self._key)
            ciphertext = aesgcm.encrypt(nonce, plaintext, None)
            payload = nonce + ciphertext
        else:
            # Fallback XOR obfuscation (NOT secure — demo only)
            payload = self._xor_bytes(plaintext, self._key)

        encoded = base64.b64encode(payload).decode("ascii")

        with open(output_path, "w") as f:
            f.write(encoded)

        log.info("Encrypted metadata → %s (%d bytes)", output_path, len(encoded))
        return output_path

    def decrypt_json(self, input_path: str) -> Dict[str, Any]:
        """
        Read and decrypt an encrypted metadata file.
        """
        with open(input_path, "r") as f:
            encoded = f.read()

        payload = base64.b64decode(encoded)

        if _HAS_CRYPTO:
            nonce = payload[:12]
            ciphertext = payload[12:]
            aesgcm = AESGCM(self._key)
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        else:
            plaintext = self._xor_bytes(payload, self._key)

        data = json.loads(plaintext.decode("utf-8"))
        log.info("Decrypted metadata ← %s", input_path)
        return data

    @staticmethod
    def _xor_bytes(data: bytes, key: bytes) -> bytes:
        """Simple XOR obfuscation (NOT secure — fallback only)."""
        key_len = len(key)
        return bytes(d ^ key[i % key_len] for i, d in enumerate(data))

    def compute_hmac(self, data: bytes) -> str:
        """Compute HMAC-SHA256 for integrity verification."""
        return hmac.new(self._key, data, hashlib.sha256).hexdigest()

    def verify_hmac(self, data: bytes, expected_hmac: str) -> bool:
        """Verify an HMAC-SHA256 signature."""
        computed = self.compute_hmac(data)
        return hmac.compare_digest(computed, expected_hmac)
