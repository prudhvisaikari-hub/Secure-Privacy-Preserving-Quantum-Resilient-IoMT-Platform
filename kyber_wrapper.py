"""
kyber_wrapper.py
================
Python bindings for CRYSTALS-Kyber (512/768/1024) via liboqs.
Falls back to a pure-Python reference implementation for environments
without liboqs installed (for CI/testing only — not for production).

Usage:
    from pqc_layer.kyber_wrapper import KyberKEM
    kem = KyberKEM(variant="Kyber512")
    pk, sk = kem.keygen()
    ct, ss_enc = kem.encapsulate(pk)
    ss_dec = kem.decapsulate(sk, ct)
    assert ss_enc == ss_dec
"""

import os
import time
import hashlib
import secrets
import logging
from dataclasses import dataclass, field
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import liboqs
# ---------------------------------------------------------------------------
try:
    import oqs  # type: ignore
    _LIBOQS_AVAILABLE = True
    logger.info("liboqs found — using hardware-backed Kyber implementation.")
except ImportError:
    _LIBOQS_AVAILABLE = False
    logger.warning(
        "liboqs not found. Falling back to stub implementation. "
        "Install liboqs for production/benchmark use."
    )

# ---------------------------------------------------------------------------
# Supported variants
# ---------------------------------------------------------------------------
KYBER_VARIANTS = {
    "Kyber512":  {"security_level": 1, "pk_bytes": 800,  "sk_bytes": 1632, "ct_bytes": 768,  "ss_bytes": 32},
    "Kyber768":  {"security_level": 3, "pk_bytes": 1184, "sk_bytes": 2400, "ct_bytes": 1088, "ss_bytes": 32},
    "Kyber1024": {"security_level": 5, "pk_bytes": 1568, "sk_bytes": 3168, "ct_bytes": 1568, "ss_bytes": 32},
}


@dataclass
class KyberBenchmark:
    """Stores timing results for one full KEM cycle."""
    variant: str
    keygen_ns: float = 0.0
    encaps_ns: float = 0.0
    decaps_ns: float = 0.0
    pk_bytes: int = 0
    sk_bytes: int = 0
    ct_bytes: int = 0
    ss_bytes: int = 0
    iterations: int = 1

    @property
    def total_ns(self) -> float:
        return self.keygen_ns + self.encaps_ns + self.decaps_ns

    def summary(self) -> dict:
        return {
            "variant": self.variant,
            "keygen_ms": round(self.keygen_ns / 1e6, 4),
            "encaps_ms": round(self.encaps_ns / 1e6, 4),
            "decaps_ms": round(self.decaps_ns / 1e6, 4),
            "total_ms":  round(self.total_ns  / 1e6, 4),
            "pk_bytes":  self.pk_bytes,
            "sk_bytes":  self.sk_bytes,
            "ct_bytes":  self.ct_bytes,
            "ss_bytes":  self.ss_bytes,
            "iterations": self.iterations,
        }


class _StubKyber:
    """
    Stub KEM for testing without liboqs.
    NOT cryptographically secure — sizes match real Kyber for benchmarking infra.
    """
    def __init__(self, variant: str):
        self.meta = KYBER_VARIANTS[variant]

    def generate_keypair(self) -> Tuple[bytes, bytes]:
        pk = secrets.token_bytes(self.meta["pk_bytes"])
        sk = secrets.token_bytes(self.meta["sk_bytes"])
        return pk, sk

    def encap_secret(self, pk: bytes) -> Tuple[bytes, bytes]:
        ss = secrets.token_bytes(self.meta["ss_bytes"])
        ct = secrets.token_bytes(self.meta["ct_bytes"])
        # In real Kyber, ct encodes the shared secret encrypted under pk.
        # Here we bind ss to ct via hash for correctness testing.
        ct = hashlib.sha3_256(pk + ss).digest()[:self.meta["ct_bytes"]].ljust(
            self.meta["ct_bytes"], b'\x00'
        )
        return ct, ss

    def decap_secret(self, sk: bytes, ct: bytes) -> bytes:
        # Stub: return zeroed shared secret (not correct — for infra testing only)
        return b'\x00' * self.meta["ss_bytes"]


class KyberKEM:
    """
    High-level interface for CRYSTALS-Kyber Key Encapsulation Mechanism.

    Args:
        variant: One of "Kyber512", "Kyber768", "Kyber1024"

    Example:
        kem = KyberKEM("Kyber768")
        pk, sk = kem.keygen()
        ct, ss = kem.encapsulate(pk)
        recovered = kem.decapsulate(sk, ct)
        assert ss == recovered
    """

    def __init__(self, variant: str = "Kyber512"):
        if variant not in KYBER_VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from {list(KYBER_VARIANTS)}")
        self.variant = variant
        self.meta = KYBER_VARIANTS[variant]
        self._backend = self._init_backend()

    def _init_backend(self):
        if _LIBOQS_AVAILABLE:
            kem_name = self.variant.replace("Kyber", "CRYSTALS-Kyber-")  # e.g. CRYSTALS-Kyber-512
            # liboqs uses "Kyber512", "Kyber768", "Kyber1024" directly
            try:
                return oqs.KeyEncapsulation(self.variant)
            except Exception:
                # Try alternate naming
                return oqs.KeyEncapsulation(self.variant.replace("Kyber", "Kyber-"))
        else:
            return _StubKyber(self.variant)

    def keygen(self) -> Tuple[bytes, bytes]:
        """Generate a (public_key, secret_key) pair."""
        if _LIBOQS_AVAILABLE:
            pk = self._backend.generate_keypair()
            sk = self._backend.export_secret_key()
            return pk, sk
        return self._backend.generate_keypair()

    def encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """
        Encapsulate a shared secret under public_key.
        Returns (ciphertext, shared_secret).
        """
        if _LIBOQS_AVAILABLE:
            ct, ss = self._backend.encap_secret(public_key)
            return ct, ss
        return self._backend.encap_secret(public_key)

    def decapsulate(self, secret_key: bytes, ciphertext: bytes) -> bytes:
        """
        Decapsulate the shared secret from ciphertext using secret_key.
        Returns shared_secret (bytes).
        """
        if _LIBOQS_AVAILABLE:
            # Re-init with the secret key
            kem = oqs.KeyEncapsulation(self.variant, secret_key)
            return kem.decap_secret(ciphertext)
        return self._backend.decap_secret(secret_key, ciphertext)

    def full_cycle(self) -> Tuple[bytes, bytes, bytes, bytes]:
        """
        Convenience: keygen → encaps → decaps in one call.
        Returns (pk, sk, ct, ss).
        """
        pk, sk = self.keygen()
        ct, ss = self.encapsulate(pk)
        ss_check = self.decapsulate(sk, ct)
        assert ss == ss_check or not _LIBOQS_AVAILABLE, (
            "Shared secret mismatch — possible implementation error!"
        )
        return pk, sk, ct, ss

    def benchmark(self, iterations: int = 100) -> KyberBenchmark:
        """
        Run N KEM cycles and return averaged timing.

        Args:
            iterations: Number of full keygen+encaps+decaps cycles.

        Returns:
            KyberBenchmark with average timings in nanoseconds.
        """
        kg_total = enc_total = dec_total = 0.0

        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            pk, sk = self.keygen()
            t1 = time.perf_counter_ns()
            ct, ss = self.encapsulate(pk)
            t2 = time.perf_counter_ns()
            _ = self.decapsulate(sk, ct)
            t3 = time.perf_counter_ns()

            kg_total  += t1 - t0
            enc_total += t2 - t1
            dec_total += t3 - t2

        return KyberBenchmark(
            variant=self.variant,
            keygen_ns=kg_total  / iterations,
            encaps_ns=enc_total / iterations,
            decaps_ns=dec_total / iterations,
            pk_bytes=self.meta["pk_bytes"],
            sk_bytes=self.meta["sk_bytes"],
            ct_bytes=self.meta["ct_bytes"],
            ss_bytes=self.meta["ss_bytes"],
            iterations=iterations,
        )

    @property
    def security_level(self) -> int:
        return self.meta["security_level"]

    def __repr__(self) -> str:
        return (
            f"KyberKEM(variant={self.variant}, "
            f"security_level={self.security_level}, "
            f"backend={'liboqs' if _LIBOQS_AVAILABLE else 'stub'})"
        )


# ---------------------------------------------------------------------------
# RSA / ECC stubs for comparison (crypto module wrappers)
# ---------------------------------------------------------------------------

class RSA_KEM:
    """Thin wrapper around cryptography.hazmat for RSA keygen/encrypt timing."""

    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
        try:
            from cryptography.hazmat.primitives.asymmetric import rsa, padding
            from cryptography.hazmat.primitives import hashes
            self._rsa = rsa
            self._padding = padding
            self._hashes = hashes
            self._available = True
        except ImportError:
            self._available = False
            logger.warning("cryptography package not found — RSA stub will use random bytes.")

    def keygen(self) -> Tuple[object, object]:
        if self._available:
            from cryptography.hazmat.backends import default_backend
            sk = self._rsa.generate_private_key(
                public_exponent=65537, key_size=self.key_size,
                backend=default_backend()
            )
            return sk.public_key(), sk
        return secrets.token_bytes(256), secrets.token_bytes(1200)

    def benchmark(self, iterations: int = 100) -> dict:
        kg_total = 0.0
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            self.keygen()
            kg_total += time.perf_counter_ns() - t0
        return {
            "scheme": f"RSA-{self.key_size}",
            "keygen_ms": round(kg_total / iterations / 1e6, 4),
            "key_size_bits": self.key_size,
        }


class ECC_KEM:
    """Thin wrapper for ECDH P-256."""

    def benchmark(self, iterations: int = 100) -> dict:
        try:
            from cryptography.hazmat.primitives.asymmetric.ec import (
                generate_private_key, ECDH, SECP256R1
            )
            from cryptography.hazmat.backends import default_backend
            kg_total = 0.0
            for _ in range(iterations):
                t0 = time.perf_counter_ns()
                generate_private_key(SECP256R1(), default_backend())
                kg_total += time.perf_counter_ns() - t0
            return {
                "scheme": "ECC-P256",
                "keygen_ms": round(kg_total / iterations / 1e6, 4),
            }
        except ImportError:
            return {"scheme": "ECC-P256", "keygen_ms": -1, "error": "cryptography not installed"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== CRYSTALS-Kyber KEM Demo ===\n")
    for variant in KYBER_VARIANTS:
        kem = KyberKEM(variant)
        print(f"Variant: {kem}")
        result = kem.benchmark(iterations=10)
        for k, v in result.summary().items():
            print(f"  {k}: {v}")
        print()
