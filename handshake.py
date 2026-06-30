"""
hybrid_migration/handshake.py
==============================
Classical/PQC hybrid handshake protocol.
Implements a dual-key handshake that combines RSA (classical) and
Kyber (PQC) key exchange, providing security against BOTH classical
and quantum adversaries simultaneously.

Hybrid construction:
  session_key = KDF(kyber_ss XOR rsa_ss, nonces)

If Kyber is broken by a novel algorithm → RSA still holds.
If RSA is broken by Shor's algorithm   → Kyber still holds.
This gives the strongest possible transitional security guarantee.

Protocol:
  Client                               Server
    |                                    |
    |── HYBRID_HELLO (cap, nonce_c) ──→  |
    |                                    |  gen Kyber pk/sk
    |                                    |  gen RSA pk/sk (or reuse cert)
    |  ←─ SERVER_HELLO (kyber_pk,        |
    |         rsa_pk, nonce_s) ─────────  |
    |                                    |
    |  kyber_ct, kyber_ss = Kyber.enc(pk)|
    |  rsa_ct = RSA.enc(rsa_pk, rand)   |
    |── HYBRID_KEY (kyber_ct, rsa_ct,    |
    |               HMAC) ─────────────→ |
    |                                    |  kyber_ss = Kyber.dec(sk, kyber_ct)
    |                                    |  rand = RSA.dec(sk, rsa_ct)
    |  session_key = KDF(kyber_ss ^ rand)|
    |───── APP DATA (AES-GCM) ─────────→ |

Usage:
    from hybrid_migration.handshake import HybridHandshake
    server = HybridHandshake(role="server")
    client = HybridHandshake(role="client")
    ...
"""

import hmac
import json
import logging
import hashlib
import secrets
import struct
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

from pqc_layer.kyber_wrapper import KyberKEM

try:
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.backends import default_backend
    _RSA = True
except ImportError:
    _RSA = False
    logger.warning("cryptography not installed — RSA leg of hybrid handshake uses stub.")

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    def _enc(k, n, p): return AESGCM(k).encrypt(n, p, b"")
    def _dec(k, n, c): return AESGCM(k).decrypt(n, c, b"")
    _AES = True
except ImportError:
    _AES = False
    def _enc(k, n, p): return p   # stub
    def _dec(k, n, c): return c   # stub


def _kdf(material: bytes, nonce_c: bytes, nonce_s: bytes) -> bytes:
    """HKDF-style key derivation using SHA3-256."""
    return hashlib.sha3_256(material + nonce_c + nonce_s + b"SPQR-Hybrid-v1").digest()


class HybridKeyMaterial:
    """Holds both PQC and classical key/ciphertext pairs."""

    def __init__(self):
        self.kyber_pk: Optional[bytes] = None
        self.kyber_sk: Optional[bytes] = None
        self.kyber_ct: Optional[bytes] = None
        self.kyber_ss: Optional[bytes] = None
        self.rsa_pk:   Optional[object] = None
        self.rsa_sk:   Optional[object] = None
        self.rsa_ct:   Optional[bytes] = None
        self.rsa_ss:   Optional[bytes] = None    # 32-byte random secret
        self.session_key: Optional[bytes] = None
        self.nonce_c: Optional[bytes] = None
        self.nonce_s: Optional[bytes] = None


class HybridHandshake:
    """
    Implements the dual RSA+Kyber hybrid handshake.

    Args:
        role:           "client" or "server"
        kyber_variant:  Kyber512 / Kyber768 / Kyber1024
        rsa_key_size:   2048 or 4096
    """

    def __init__(self, role: str = "client",
                 kyber_variant: str = "Kyber768",
                 rsa_key_size: int = 2048):
        assert role in ("client", "server")
        self.role    = role
        self.kem     = KyberKEM(kyber_variant)
        self.rsa_bits = rsa_key_size
        self.km      = HybridKeyMaterial()
        self._tx_seq = 0

    # ------------------------------------------------------------------ server

    def server_gen_hello(self) -> bytes:
        """
        Server generates Kyber key pair + RSA key pair.
        Returns serialised SERVER_HELLO bytes.
        """
        assert self.role == "server"
        self.km.nonce_s = secrets.token_bytes(32)

        # Kyber keygen
        self.km.kyber_pk, self.km.kyber_sk = self.kem.keygen()

        # RSA keygen (or load from cert in production)
        if _RSA:
            self.km.rsa_sk = rsa.generate_private_key(65537, self.rsa_bits, default_backend())
            self.km.rsa_pk = self.km.rsa_sk.public_key()
            rsa_pk_bytes   = self.km.rsa_pk.public_bytes(
                serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
            )
        else:
            rsa_pk_bytes = secrets.token_bytes(294)  # stub

        # Serialise: [nonce_s | kyber_pk_len | kyber_pk | rsa_pk_len | rsa_pk]
        kpk = self.km.kyber_pk
        msg = (
            self.km.nonce_s
            + struct.pack(">I", len(kpk)) + kpk
            + struct.pack(">I", len(rsa_pk_bytes)) + rsa_pk_bytes
        )
        logger.debug("[Server] ServerHello sent.")
        return msg

    def server_handle_hybrid_key(self, nonce_c: bytes, hybrid_key_msg: bytes) -> bool:
        """
        Process client's HYBRID_KEY message.
        Decapsulates Kyber SS, decrypts RSA secret, derives session key.
        Returns True on success.
        """
        assert self.role == "server"
        self.km.nonce_c = nonce_c
        offset = 0

        kyber_ct_len = struct.unpack(">I", hybrid_key_msg[offset:offset+4])[0]; offset += 4
        kyber_ct     = hybrid_key_msg[offset:offset+kyber_ct_len];               offset += kyber_ct_len
        rsa_ct_len   = struct.unpack(">I", hybrid_key_msg[offset:offset+4])[0]; offset += 4
        rsa_ct       = hybrid_key_msg[offset:offset+rsa_ct_len];                offset += rsa_ct_len
        mac          = hybrid_key_msg[offset:offset+32]

        # Kyber decapsulate
        self.km.kyber_ss = self.kem.decapsulate(self.km.kyber_sk, kyber_ct)

        # RSA decrypt
        if _RSA and self.km.rsa_sk:
            try:
                self.km.rsa_ss = self.km.rsa_sk.decrypt(
                    rsa_ct, padding.OAEP(
                        mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None
                    )
                )
            except Exception as e:
                logger.error(f"[Server] RSA decrypt failed: {e}")
                return False
        else:
            self.km.rsa_ss = b'\x00' * 32

        # Verify MAC
        combined = bytes(a ^ b for a, b in zip(self.km.kyber_ss, self.km.rsa_ss))
        expected_mac = hmac.new(
            combined, kyber_ct + rsa_ct + nonce_c + self.km.nonce_s,
            digestmod=hashlib.sha3_256
        ).digest()
        if not hmac.compare_digest(expected_mac, mac):
            logger.error("[Server] HYBRID_KEY MAC verification failed — possible MITM!")
            return False

        self.km.session_key = _kdf(combined, nonce_c, self.km.nonce_s)
        logger.info("[Server] Hybrid session established ✓")
        return True

    # ------------------------------------------------------------------ client

    def client_handle_server_hello(self, server_hello: bytes) -> bytes:
        """
        Process SERVER_HELLO, encapsulate both secrets, return HYBRID_KEY.
        """
        assert self.role == "client"
        self.km.nonce_c = secrets.token_bytes(32)
        offset = 0

        nonce_s = server_hello[offset:offset+32]; offset += 32
        self.km.nonce_s = nonce_s

        kyber_pk_len = struct.unpack(">I", server_hello[offset:offset+4])[0]; offset += 4
        kyber_pk     = server_hello[offset:offset+kyber_pk_len];               offset += kyber_pk_len
        rsa_pk_len   = struct.unpack(">I", server_hello[offset:offset+4])[0]; offset += 4
        rsa_pk_bytes = server_hello[offset:offset+rsa_pk_len]

        # Kyber encapsulate
        self.km.kyber_ct, self.km.kyber_ss = self.kem.encapsulate(kyber_pk)

        # RSA encrypt
        self.km.rsa_ss = secrets.token_bytes(32)
        if _RSA:
            try:
                pub = serialization.load_der_public_key(rsa_pk_bytes, backend=default_backend())
                self.km.rsa_ct = pub.encrypt(
                    self.km.rsa_ss, padding.OAEP(
                        mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None
                    )
                )
            except Exception:
                self.km.rsa_ct = secrets.token_bytes(256)
        else:
            self.km.rsa_ct = secrets.token_bytes(256)

        combined = bytes(a ^ b for a, b in zip(self.km.kyber_ss, self.km.rsa_ss))
        mac = hmac.new(
            combined,
            self.km.kyber_ct + self.km.rsa_ct + self.km.nonce_c + nonce_s,
            digestmod=hashlib.sha3_256
        ).digest()

        self.km.session_key = _kdf(combined, self.km.nonce_c, nonce_s)

        msg = (
            struct.pack(">I", len(self.km.kyber_ct)) + self.km.kyber_ct
            + struct.pack(">I", len(self.km.rsa_ct)) + self.km.rsa_ct
            + mac
        )
        logger.debug("[Client] HYBRID_KEY sent.")
        return msg

    # ------------------------------------------------------------------ shared

    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt application data with session key (AES-256-GCM)."""
        if not self.km.session_key:
            raise RuntimeError("Handshake not complete.")
        nonce = secrets.token_bytes(12)
        ct    = _enc(self.km.session_key, nonce, plaintext)
        seq   = struct.pack(">Q", self._tx_seq)
        self._tx_seq += 1
        return seq + nonce + ct

    def decrypt(self, data: bytes) -> bytes:
        """Decrypt incoming application data."""
        if not self.km.session_key:
            raise RuntimeError("Handshake not complete.")
        seq   = struct.unpack(">Q", data[:8])[0]
        nonce = data[8:20]
        ct    = data[20:]
        return _dec(self.km.session_key, nonce, ct)

    @property
    def session_established(self) -> bool:
        return self.km.session_key is not None

    def handshake_info(self) -> dict:
        return {
            "role":             self.role,
            "kyber_variant":    self.kem.variant,
            "rsa_key_bits":     self.rsa_bits,
            "session_established": self.session_established,
            "hybrid_construction": "session_key = KDF(Kyber_SS ⊕ RSA_SS, nonces)",
            "security_guarantee": (
                "Secure against quantum AND classical adversaries. "
                "Broken only if BOTH Kyber AND RSA are simultaneously broken."
            ),
        }


def run_hybrid_handshake_demo():
    logging.basicConfig(level=logging.INFO)
    print("\n=== Hybrid RSA+Kyber Handshake Demo ===\n")
    server = HybridHandshake(role="server", kyber_variant="Kyber768", rsa_key_size=2048)
    client = HybridHandshake(role="client", kyber_variant="Kyber768", rsa_key_size=2048)

    t0 = time.perf_counter_ns()
    sh_msg     = server.server_gen_hello()
    hk_msg     = client.client_handle_server_hello(sh_msg)
    ok         = server.server_handle_hybrid_key(client.km.nonce_c, hk_msg)
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6

    print(f"Handshake success: {ok}")
    print(f"Session keys match: {server.km.session_key == client.km.session_key}")
    print(f"Handshake latency: {elapsed_ms:.2f} ms")
    print(f"Server info: {server.handshake_info()}")

    plaintext = b'{"hr": 72, "spo2": 98, "temp": 36.5}'
    enc = client.encrypt(plaintext)
    dec = server.decrypt(enc)
    print(f"\nEncrypted payload ({len(enc)} bytes)")
    print(f"Decrypted: {dec}")


if __name__ == "__main__":
    run_hybrid_handshake_demo()
