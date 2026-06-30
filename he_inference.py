"""
he_inference.py
===============
Homomorphic Encryption for privacy-preserving vitals inference using TenSEAL (CKKS scheme).

Workflow:
  1. Hospital trains a logistic regression / simple model locally.
  2. Patient data is encrypted client-side (CKKS).
  3. Encrypted data is sent to an untrusted edge server.
  4. Edge server runs inference ON ENCRYPTED DATA (no decryption).
  5. Encrypted result is returned to client and decrypted locally.

This implements "Client-Aided HE Inference" for a linear model:
    risk_score = sigmoid(W·x + b)   computed under CKKS encryption.

For non-linear activations (sigmoid), we use a polynomial approximation.

Usage:
    from federated_learning.he_inference import HEInferenceEngine
    engine = HEInferenceEngine()
    W, b = engine.train_plaintext_model(X_train, y_train)
    encrypted_score = engine.encrypted_inference(X_patient)
    score = engine.decrypt_result(encrypted_score)
"""

import time
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TenSEAL import (CKKS HE)
# ---------------------------------------------------------------------------
try:
    import tenseal as ts
    _TENSEAL_AVAILABLE = True
    logger.info("TenSEAL found — using CKKS homomorphic encryption.")
except ImportError:
    _TENSEAL_AVAILABLE = False
    logger.warning(
        "TenSEAL not installed. Simulating HE operations with plaintext + timing estimates. "
        "Install via: pip install tenseal"
    )

# ---------------------------------------------------------------------------
# Polynomial sigmoid approximation for CKKS
# ---------------------------------------------------------------------------

def poly_sigmoid(x: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    Degree-3 polynomial approximation of sigmoid for use under HE.
    Approximation: σ(x) ≈ 0.5 + 0.197x - 0.004x³  (valid for |x| < 5)
    From: "Logistic Regression on Homomorphically Encrypted Data" (Bonte & Vercauteren, 2018)
    """
    return 0.5 + 0.197 * x - 0.004 * (x ** 3)


# ---------------------------------------------------------------------------
# CKKS Context factory
# ---------------------------------------------------------------------------

def create_ckks_context(poly_modulus_degree: int = 8192,
                        coeff_mod_bit_sizes: Optional[List[int]] = None,
                        global_scale: float = 2**40) -> "ts.Context":
    """
    Create a TenSEAL CKKS context.

    Args:
        poly_modulus_degree: Ring dimension (8192 = 128-bit security, 16384 = stronger)
        coeff_mod_bit_sizes: Coefficient modulus chain (controls depth)
        global_scale: Scaling factor for CKKS encoding (2^40 balances precision/noise)

    Returns:
        ts.Context ready for encryption.
    """
    if not _TENSEAL_AVAILABLE:
        return None

    if coeff_mod_bit_sizes is None:
        coeff_mod_bit_sizes = [60, 40, 40, 60]  # supports ~2 multiplications

    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    ctx.global_scale = global_scale
    ctx.generate_galois_keys()   # needed for vector rotations / dot products
    return ctx


# ---------------------------------------------------------------------------
# Simple logistic regression model
# ---------------------------------------------------------------------------

class LinearVitalsClassifier:
    """
    Logistic regression on vitals features.
    Suitable for exact HE inference (linear + polynomial sigmoid approximation).
    """

    def __init__(self, n_features: int = 5):
        self.n_features = n_features
        self.W = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray,
              lr: float = 0.1, epochs: int = 200) -> "LinearVitalsClassifier":
        """Train via gradient descent (no HE needed for training)."""
        N = len(X)
        self.W = np.zeros(self.n_features, dtype=np.float64)
        self.b = 0.0

        for epoch in range(epochs):
            logits = X @ self.W + self.b
            preds  = 1 / (1 + np.exp(-logits))        # sigmoid
            errors = preds - y.flatten()
            dW = (X.T @ errors) / N
            db = errors.mean()
            self.W -= lr * dW
            self.b  -= lr * db

            if epoch % 50 == 0:
                loss = -np.mean(y * np.log(preds + 1e-8) + (1-y) * np.log(1 - preds + 1e-8))
                logger.debug(f"  Epoch {epoch}: BCE loss = {loss:.4f}")

        self.is_trained = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Plaintext prediction."""
        logits = X @ self.W + self.b
        return 1 / (1 + np.exp(-logits))

    def predict_proba_poly(self, X: np.ndarray) -> np.ndarray:
        """Prediction using polynomial sigmoid (matches HE computation)."""
        logits = X @ self.W + self.b
        return poly_sigmoid(logits)


# ---------------------------------------------------------------------------
# HE Inference Engine
# ---------------------------------------------------------------------------

class HEInferenceEngine:
    """
    Privacy-preserving inference on encrypted vitals data using CKKS HE.

    The model (weights W, bias b) is known to the server (public).
    Patient features x are encrypted by the client — server NEVER sees them in plaintext.
    """

    def __init__(self,
                 n_features: int = 5,
                 poly_modulus_degree: int = 8192):
        self.n_features = n_features
        self.poly_modulus_degree = poly_modulus_degree
        self.model = LinearVitalsClassifier(n_features)
        self.ctx = create_ckks_context(poly_modulus_degree) if _TENSEAL_AVAILABLE else None
        self.timings: Dict[str, float] = {}

    def train_plaintext_model(self,
                              X: Optional[np.ndarray] = None,
                              y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Train the plaintext model. In federated setting, W and b come from FL server.
        Returns (W, b).
        """
        if X is None or y is None:
            # Generate synthetic training data
            np.random.seed(42)
            N = 1000
            X_neg = np.random.randn(750, self.n_features) * np.array([10, 2, 3, 15, 0.4])
            X_pos = np.random.randn(250, self.n_features) * np.array([15, 4, 5, 20, 0.8]) \
                    + np.array([20, -6, 8, -30, 1.7])
            X = np.vstack([X_neg, X_pos]).astype(np.float64)
            y = np.array([0]*750 + [1]*250, dtype=np.float64)

        # Normalize
        self.X_mean = X.mean(axis=0)
        self.X_std  = X.std(axis=0) + 1e-8
        X_norm = (X - self.X_mean) / self.X_std

        t0 = time.perf_counter_ns()
        self.model.train(X_norm, y)
        self.timings["train_plaintext_ms"] = (time.perf_counter_ns() - t0) / 1e6

        logger.info(
            f"[HE] Model trained: W={np.round(self.model.W, 3)}, b={self.model.b:.3f}"
        )
        return self.model.W, self.model.b

    def client_encrypt(self, x: np.ndarray) -> Any:
        """
        CLIENT-SIDE: Normalize and encrypt a feature vector.
        Returns encrypted vector (or plaintext stub if TenSEAL unavailable).
        """
        if not hasattr(self, "X_mean"):
            raise RuntimeError("Train model first to get normalization params.")

        x_norm = (x - self.X_mean) / self.X_std

        t0 = time.perf_counter_ns()
        if _TENSEAL_AVAILABLE and self.ctx:
            enc = ts.ckks_vector(self.ctx, x_norm.tolist())
        else:
            enc = {"_stub": True, "data": x_norm}  # plaintext stub
        self.timings["encrypt_ms"] = (time.perf_counter_ns() - t0) / 1e6

        return enc

    def server_infer(self, encrypted_x: Any) -> Any:
        """
        SERVER-SIDE: Run inference on encrypted data.
        Computes: logit = W·x_enc + b  (all under HE)
        Then applies polynomial sigmoid approximation.
        Server learns NOTHING about x.
        """
        if not self.model.is_trained:
            raise RuntimeError("No model weights available on server.")

        W = self.model.W.tolist()
        b = float(self.model.b)

        t0 = time.perf_counter_ns()
        if _TENSEAL_AVAILABLE and isinstance(encrypted_x, ts.CKKSVector):
            # Encrypted dot product
            enc_logit = encrypted_x.dot(W)  # W·x (encrypted)
            enc_logit += b                  # + b (plaintext addition)
            # Polynomial sigmoid approximation under HE
            # σ(t) ≈ 0.5 + 0.197t - 0.004t³
            enc_result = 0.5 + enc_logit * 0.197 + (enc_logit ** 3) * (-0.004)
        else:
            # Stub: plaintext computation (simulated HE)
            x_norm = encrypted_x["data"]
            logit = float(np.dot(W, x_norm) + b)
            enc_result = {"_stub": True, "logit": logit, "score": poly_sigmoid(np.array([logit]))[0]}

        self.timings["server_infer_ms"] = (time.perf_counter_ns() - t0) / 1e6
        return enc_result

    def client_decrypt(self, encrypted_result: Any) -> float:
        """
        CLIENT-SIDE: Decrypt inference result.
        Returns risk score in [0, 1].
        """
        t0 = time.perf_counter_ns()
        if _TENSEAL_AVAILABLE and isinstance(encrypted_result, ts.CKKSVector):
            score = encrypted_result.decrypt()[0]
        else:
            score = encrypted_result.get("score", 0.5)
        self.timings["decrypt_ms"] = (time.perf_counter_ns() - t0) / 1e6

        return float(np.clip(score, 0.0, 1.0))

    def end_to_end(self, x: np.ndarray) -> Tuple[float, Dict]:
        """
        Full client→server→client HE inference pipeline.
        Returns (risk_score, timing_dict).
        """
        enc_x      = self.client_encrypt(x)
        enc_result = self.server_infer(enc_x)
        score      = self.client_decrypt(enc_result)
        total_ms   = sum(self.timings.values())

        # Plaintext reference for accuracy check
        x_norm = (x - self.X_mean) / self.X_std
        ref_score = float(self.model.predict_proba_poly(x_norm.reshape(1, -1))[0])
        error = abs(score - ref_score)

        report = {
            **self.timings,
            "total_ms": round(total_ms, 4),
            "risk_score": round(score, 4),
            "reference_score": round(ref_score, 4),
            "approximation_error": round(error, 6),
            "tenseal_available": _TENSEAL_AVAILABLE,
        }
        return score, report

    def benchmark(self, n_samples: int = 20) -> dict:
        """Benchmark HE inference latency over n_samples patients."""
        latencies = []
        errors = []

        # Generate test samples
        np.random.seed(99)
        X_test = np.random.randn(n_samples, self.n_features).astype(np.float64)

        for i in range(n_samples):
            _, report = self.end_to_end(X_test[i])
            latencies.append(report["total_ms"])
            errors.append(report["approximation_error"])

        return {
            "n_samples": n_samples,
            "mean_latency_ms": round(float(np.mean(latencies)), 4),
            "std_latency_ms":  round(float(np.std(latencies)), 4),
            "max_latency_ms":  round(float(np.max(latencies)), 4),
            "mean_approx_error": round(float(np.mean(errors)), 6),
            "max_approx_error":  round(float(np.max(errors)), 6),
            "poly_modulus_degree": self.poly_modulus_degree,
            "tenseal_available": _TENSEAL_AVAILABLE,
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n=== SPQR-IoMT Homomorphic Encryption Inference Demo ===\n")

    engine = HEInferenceEngine(n_features=5)

    # Train on synthetic data
    W, b = engine.train_plaintext_model()
    print(f"Model trained: W={np.round(W, 3)}, b={b:.3f}\n")

    # Single patient inference
    patient = np.array([95.0, 91.0, 24.0, 90.0, 38.5])  # deteriorating vitals
    score, report = engine.end_to_end(patient)
    print(f"Patient vitals: {dict(zip(['HR','SpO2','RR','SBP','Temp'], patient))}")
    print(f"Risk score (HE): {score:.4f}")
    print(f"Report: {report}\n")

    # Benchmark
    print("Running HE benchmark (20 samples)...")
    bench = engine.benchmark(n_samples=20)
    print(f"Benchmark results: {bench}")
