"""
secure_channel.py
=================
DTLS-style secure telemetry protocol for IoMT sensors using Kyber KEM.

Protocol flow (simplified):
  Sensor (Client)                    Edge Gateway (Server)
      |                                      |
      |--- [HELLO: sensor_id, nonce_c] ----> |
      |                                      |  gen pk, sk = Kyber.keygen()
      | <-- [SERVER_HELLO: pk, nonce_s] ---- |
      |                                      |
      |  ct, ss = Kyber.encaps(pk)           |
      |--- [CLIENT_KEY: ct, HMAC(ss,msg)] -> |
      |                                      |  ss = Kyber.decaps(sk, ct)
      |  session_key = KDF(ss, nonces)       |  session_key = KDF(ss, nonces)
      |--- [DATA: AES-GCM(session_key, telemetry)] -->|
      |                                      |

Usage:
    server = SecureServer("Kyber768")
    client = SecureClient("Kyber768")

    # Handshake
    hello = client.send_hello(sensor_id="sensor_001")
    server_hello = server.handle_hello(hello)
    key_msg = client.handle_server_hello(server_hello)
    server.handle_client_key(key_msg)

    # Secure messaging
    payload = client.send_data({"hr": 72, "spo2": 98, "temp": 36.5})
    decrypted = server.receive_data(payload)
"""

import os
import json
import time
import hmac
import struct
import hashlib
import secrets
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from pqc_layer.kyber_wrapper import KyberKEM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AES-GCM wrapper (stdlib + PyCryptodome fallback)
# ---------------------------------------------------------------------------
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    def _aes_gcm_encrypt(key: bytes, nonce: bytes, plaintext: bytes, aad: bytes = b"") -> bytes:
        return AESGCM(key).encrypt(nonce, plaintext, aad)
    def _aes_gcm_decrypt(key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes = b"") -> bytes:
        return AESGCM(key).decrypt(nonce, ciphertext, aad)
    _AES_AVAILABLE = True
except ImportError:
    logger.warning("cryptography not installed — using XOR stub cipher (NOT secure)")
    _AES_AVAILABLE = False
    def _aes_gcm_encrypt(key: bytes, nonce: bytes, plaintext: bytes, aad: bytes = b"") -> bytes:
        key_stream = hashlib.shake_128(key + nonce).digest(len(plaintext))
        return bytes(a ^ b for a, b in zip(plaintext, key_stream))
    def _aes_gcm_decrypt(key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes = b"") -> bytes:
        return _aes_gcm_encrypt(key, nonce, ciphertext, aad)


# ---------------------------------------------------------------------------
# Key Derivation Function
# ---------------------------------------------------------------------------
def _kdf(shared_secret: bytes, nonce_c: bytes, nonce_s: bytes, label: bytes = b"SPQR-IoMT-v1") -> bytes:
    """HKDF-style KDF using SHA3-256."""
    ikm = shared_secret + nonce_c + nonce_s + label
    return hashlib.sha3_256(ikm).digest()  # 256-bit session key


# ---------------------------------------------------------------------------
# Message structures
# ---------------------------------------------------------------------------
@dataclass
class HelloMessage:
    sensor_id: str
    nonce_c: bytes  # 32-byte random nonce

    def to_bytes(self) -> bytes:
        sid = self.sensor_id.encode()
        return struct.pack(">H", len(sid)) + sid + self.nonce_c

    @classmethod
    def from_bytes(cls, data: bytes) -> "HelloMessage":
        sid_len = struct.unpack(">H", data[:2])[0]
        sid = data[2:2 + sid_len].decode()
        nonce = data[2 + sid_len:2 + sid_len + 32]
        return cls(sensor_id=sid, nonce_c=nonce)


@dataclass
class ServerHelloMessage:
    public_key: bytes
    nonce_s: bytes  # 32-byte random nonce
    kyber_variant: str = "Kyber512"

    def to_bytes(self) -> bytes:
        var = self.kyber_variant.encode()
        return (
            struct.pack(">H", len(var)) + var
            + struct.pack(">I", len(self.public_key)) + self.public_key
            + self.nonce_s
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "ServerHelloMessage":
        offset = 0
        var_len = struct.unpack(">H", data[offset:offset+2])[0]; offset += 2
        variant = data[offset:offset+var_len].decode(); offset += var_len
        pk_len  = struct.unpack(">I", data[offset:offset+4])[0]; offset += 4
        pk      = data[offset:offset+pk_len]; offset += pk_len
        nonce_s = data[offset:offset+32]
        return cls(public_key=pk, nonce_s=nonce_s, kyber_variant=variant)


@dataclass
class ClientKeyMessage:
    ciphertext: bytes
    mac: bytes  # HMAC-SHA3-256 over (ct + nonce_c + nonce_s)

    def to_bytes(self) -> bytes:
        return struct.pack(">I", len(self.ciphertext)) + self.ciphertext + self.mac

    @classmethod
    def from_bytes(cls, data: bytes) -> "ClientKeyMessage":
        ct_len = struct.unpack(">I", data[:4])[0]
        ct  = data[4:4 + ct_len]
        mac = data[4 + ct_len:4 + ct_len + 32]
        return cls(ciphertext=ct, mac=mac)


@dataclass
class DataMessage:
    nonce: bytes          # 12-byte AES-GCM nonce
    ciphertext: bytes     # encrypted telemetry
    sequence: int = 0

    def to_bytes(self) -> bytes:
        return struct.pack(">IQ", len(self.ciphertext), self.sequence) + self.nonce + self.ciphertext

    @classmethod
    def from_bytes(cls, data: bytes) -> "DataMessage":
        ct_len, seq = struct.unpack(">IQ", data[:12])
        nonce = data[12:24]
        ct    = data[24:24 + ct_len]
        return cls(nonce=nonce, ciphertext=ct, sequence=seq)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
class SecureServer:
    """
    IoMT Edge Gateway — holds Kyber key pair, decapsulates session key,
    and decrypts incoming telemetry.
    """

    def __init__(self, kyber_variant: str = "Kyber512"):
        self.kem = KyberKEM(kyber_variant)
        self.kyber_variant = kyber_variant
        self._sessions: Dict[str, Dict] = {}

    def handle_hello(self, hello_bytes: bytes) -> bytes:
        """Process ClientHello → generate key pair → return ServerHello."""
        hello = HelloMessage.from_bytes(hello_bytes)
        sensor_id = hello.sensor_id
        nonce_s = secrets.token_bytes(32)
        pk, sk = self.kem.keygen()
        self._sessions[sensor_id] = {
            "sk": sk, "pk": pk,
            "nonce_c": hello.nonce_c,
            "nonce_s": nonce_s,
            "session_key": None,
            "rx_seq": -1,
        }
        logger.info(f"[Server] Handshake started with {sensor_id}")
        return ServerHelloMessage(public_key=pk, nonce_s=nonce_s,
                                  kyber_variant=self.kyber_variant).to_bytes()

    def handle_client_key(self, sensor_id: str, ck_bytes: bytes) -> bool:
        """Decapsulate shared secret, verify MAC, derive session key."""
        sess = self._sessions[sensor_id]
        ck = ClientKeyMessage.from_bytes(ck_bytes)

        # Decapsulate
        ss = self.kem.decapsulate(sess["sk"], ck.ciphertext)

        # Verify HMAC
        expected_mac = hmac.new(
            ss, ck.ciphertext + sess["nonce_c"] + sess["nonce_s"],
            digestmod=hashlib.sha3_256
        ).digest()
        if not hmac.compare_digest(expected_mac, ck.mac):
            logger.error(f"[Server] MAC verification failed for {sensor_id}")
            return False

        sess["session_key"] = _kdf(ss, sess["nonce_c"], sess["nonce_s"])
        logger.info(f"[Server] Session established with {sensor_id}")
        return True

    def receive_data(self, sensor_id: str, data_bytes: bytes) -> Optional[Dict[str, Any]]:
        """Decrypt and return telemetry from authenticated sensor."""
        sess = self._sessions.get(sensor_id)
        if not sess or not sess.get("session_key"):
            raise RuntimeError(f"No session for {sensor_id}")
        msg = DataMessage.from_bytes(data_bytes)
        if msg.sequence <= sess["rx_seq"]:
            logger.warning(f"[Server] Replay detected: seq={msg.sequence}")
            return None
        plaintext = _aes_gcm_decrypt(sess["session_key"], msg.nonce, msg.ciphertext)
        sess["rx_seq"] = msg.sequence
        return json.loads(plaintext)

    def session_info(self, sensor_id: str) -> dict:
        sess = self._sessions.get(sensor_id, {})
        return {
            "sensor_id": sensor_id,
            "session_active": sess.get("session_key") is not None,
            "kyber_variant": self.kyber_variant,
        }


# ---------------------------------------------------------------------------
# Client (Sensor)
# ---------------------------------------------------------------------------
class SecureClient:
    """
    IoMT Sensor — initiates handshake, encapsulates shared secret,
    and encrypts telemetry payloads.
    """

    def __init__(self, sensor_id: str, kyber_variant: str = "Kyber512"):
        self.kem = KyberKEM(kyber_variant)
        self.sensor_id = sensor_id
        self._state: Dict = {}
        self._tx_seq = 0

    def send_hello(self) -> bytes:
        """Build and return a ClientHello message."""
        nonce_c = secrets.token_bytes(32)
        self._state["nonce_c"] = nonce_c
        return HelloMessage(sensor_id=self.sensor_id, nonce_c=nonce_c).to_bytes()

    def handle_server_hello(self, sh_bytes: bytes) -> bytes:
        """
        Process ServerHello: encapsulate shared secret, build ClientKey.
        Returns ClientKey bytes for server.
        """
        sh = ServerHelloMessage.from_bytes(sh_bytes)
        self._state["nonce_s"] = sh.nonce_s
        ct, ss = self.kem.encapsulate(sh.public_key)

        mac = hmac.new(
            ss,
            ct + self._state["nonce_c"] + sh.nonce_s,
            digestmod=hashlib.sha3_256
        ).digest()

        self._state["session_key"] = _kdf(ss, self._state["nonce_c"], sh.nonce_s)
        logger.info(f"[Client:{self.sensor_id}] Session key derived.")
        return ClientKeyMessage(ciphertext=ct, mac=mac).to_bytes()

    def send_data(self, payload: Dict[str, Any]) -> bytes:
        """Encrypt and return a DataMessage."""
        if not self._state.get("session_key"):
            raise RuntimeError("Handshake not complete — no session key.")
        nonce = secrets.token_bytes(12)
        plaintext = json.dumps(payload).encode()
        ct = _aes_gcm_encrypt(self._state["session_key"], nonce, plaintext)
        msg = DataMessage(nonce=nonce, ciphertext=ct, sequence=self._tx_seq)
        self._tx_seq += 1
        return msg.to_bytes()


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------
def run_demo(variant: str = "Kyber512"):
    logging.basicConfig(level=logging.INFO)
    print(f"\n=== SPQR-IoMT Secure Channel Demo ({variant}) ===\n")

    server = SecureServer(variant)
    client = SecureClient("sensor_ventilator_001", variant)

    # --- Handshake ---
    t0 = time.perf_counter_ns()
    hello = client.send_hello()
    sh    = server.handle_hello(hello)
    ck    = client.handle_server_hello(sh)
    ok    = server.handle_client_key(client.sensor_id, ck)
    handshake_ms = (time.perf_counter_ns() - t0) / 1e6

    print(f"Handshake success: {ok}")
    print(f"Handshake latency: {handshake_ms:.3f} ms")
    print(f"Server session info: {server.session_info(client.sensor_id)}")

    # --- Data exchange ---
    telemetry = {
        "sensor_id": "sensor_ventilator_001",
        "timestamp": time.time(),
        "heart_rate": 72,
        "spo2": 98,
        "rr": 16,
        "tidal_volume_ml": 480,
        "peep_cmH2O": 5,
    }

    t1 = time.perf_counter_ns()
    data_msg = client.send_data(telemetry)
    encrypt_ms = (time.perf_counter_ns() - t1) / 1e6

    t2 = time.perf_counter_ns()
    received = server.receive_data(client.sensor_id, data_msg)
    decrypt_ms = (time.perf_counter_ns() - t2) / 1e6

    print(f"\nTelemetry payload ({len(data_msg)} bytes on wire):")
    print(f"  Encrypt: {encrypt_ms:.3f} ms | Decrypt: {decrypt_ms:.3f} ms")
    print(f"  Received: {received}")


if __name__ == "__main__":
    for v in ["Kyber512", "Kyber768", "Kyber1024"]:
        run_demo(v)
