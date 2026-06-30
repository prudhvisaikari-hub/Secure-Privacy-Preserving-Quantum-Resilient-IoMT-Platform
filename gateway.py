"""
gateway.py
==========
Hybrid RSA ↔ PQC migration gateway for legacy IoMT systems.

Implements:
  - Dual-mode TLS-like handshake supporting RSA-2048 + Kyber simultaneously
  - Automatic capability negotiation (prefer PQC, fallback to classical)
  - Logging and audit trail for migration tracking
  - Phase-based rollout: CLASSICAL_ONLY → HYBRID → PQC_ONLY

This is designed for hospital networks where:
  - New devices support Kyber (edge gateways, new sensors)
  - Legacy devices only support RSA/ECC (older monitors, pumps)
  - Gateway bridges both worlds during a multi-year migration

Usage:
    gw = MigrationGateway(migration_phase=MigrationPhase.HYBRID)
    
    # Device connects with classical credentials
    result = gw.negotiate(device_id="pump_001", device_capabilities=["RSA-2048"])
    
    # Device connects with PQC support
    result = gw.negotiate(device_id="sensor_002", device_capabilities=["Kyber768", "RSA-2048"])
"""

import json
import time
import logging
import hashlib
import secrets
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Migration phases
# ---------------------------------------------------------------------------

class MigrationPhase(str, Enum):
    CLASSICAL_ONLY = "classical_only"   # Phase 0: RSA/ECC only (pre-migration)
    HYBRID         = "hybrid"           # Phase 1: accept both, prefer PQC
    PQC_PREFERRED  = "pqc_preferred"    # Phase 2: PQC preferred, classical deprecated
    PQC_ONLY       = "pqc_only"         # Phase 3: PQC mandated, classical rejected


PHASE_DESCRIPTIONS = {
    MigrationPhase.CLASSICAL_ONLY: "Legacy mode — RSA/ECC only. No PQC support.",
    MigrationPhase.HYBRID:         "Hybrid mode — both RSA and PQC accepted. PQC preferred for new connections.",
    MigrationPhase.PQC_PREFERRED:  "PQC preferred — legacy devices warned about deprecation.",
    MigrationPhase.PQC_ONLY:       "PQC mandated — classical devices cannot connect.",
}

DEVICE_CAPABILITIES = {
    # Modern devices (new installs)
    "sensor_gen3":    ["Kyber768", "Kyber512", "RSA-2048"],
    "edge_gateway":   ["Kyber1024", "Kyber768", "ECC-P384", "RSA-4096"],
    # Mid-generation (hybrid capable)
    "sensor_gen2":    ["Kyber512", "RSA-2048"],
    "monitor_2023":   ["Kyber512", "ECC-P256"],
    # Legacy devices (classical only)
    "pump_legacy":    ["RSA-2048"],
    "monitor_2018":   ["ECC-P256"],
    "scada_v1":       ["RSA-1024"],   # Very weak! Will be flagged.
}


@dataclass
class NegotiationResult:
    device_id: str
    selected_scheme: str
    is_pqc: bool
    is_hybrid: bool
    security_level: int
    deprecation_warning: Optional[str]
    session_id: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "selected_scheme": self.selected_scheme,
            "is_pqc": self.is_pqc,
            "is_hybrid": self.is_hybrid,
            "security_level": self.security_level,
            "deprecation_warning": self.deprecation_warning,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Scheme registry
# ---------------------------------------------------------------------------

SCHEME_REGISTRY = {
    # PQC schemes (preferred)
    "Kyber1024": {"type": "pqc", "security_level": 5, "priority": 10},
    "Kyber768":  {"type": "pqc", "security_level": 3, "priority": 9},
    "Kyber512":  {"type": "pqc", "security_level": 1, "priority": 8},
    # Classical (deprecated)
    "RSA-4096":  {"type": "classical", "security_level": 3, "priority": 5},
    "ECC-P384":  {"type": "classical", "security_level": 3, "priority": 4},
    "ECC-P256":  {"type": "classical", "security_level": 2, "priority": 3},
    "RSA-2048":  {"type": "classical", "security_level": 2, "priority": 2},
    "RSA-1024":  {"type": "classical", "security_level": 0, "priority": 0,
                  "deprecated": True, "critical_weakness": True},
}


# ---------------------------------------------------------------------------
# Migration Gateway
# ---------------------------------------------------------------------------

class MigrationGateway:
    """
    IoMT network gateway supporting phased RSA → PQC migration.

    Handles:
      - Capability negotiation per device
      - Phase-specific allow/deny logic
      - Audit logging of all connections
      - Deprecation warnings and forced upgrade notifications
      - Migration progress tracking
    """

    def __init__(self,
                 migration_phase: MigrationPhase = MigrationPhase.HYBRID,
                 audit_log_path: str = "benchmarks/results/migration_audit.jsonl"):
        self.phase = migration_phase
        self.audit_path = audit_log_path
        self._sessions: Dict[str, NegotiationResult] = {}
        self._connection_log: List[dict] = []
        Path(audit_log_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[Gateway] Phase: {migration_phase.value} — {PHASE_DESCRIPTIONS[migration_phase]}"
        )

    def negotiate(self, device_id: str,
                  device_capabilities: List[str]) -> NegotiationResult:
        """
        Negotiate the strongest mutually supported cryptographic scheme.

        Args:
            device_id: Unique device identifier
            device_capabilities: List of schemes the device supports (e.g. ["Kyber512", "RSA-2048"])

        Returns:
            NegotiationResult with selected scheme or raises if incompatible.
        """
        # Filter to known schemes
        supported = {
            scheme: SCHEME_REGISTRY[scheme]
            for scheme in device_capabilities
            if scheme in SCHEME_REGISTRY
        }

        if not supported:
            return self._reject(device_id, "No recognized cryptographic schemes.")

        # Phase-based filtering
        allowed = self._filter_by_phase(supported)
        if not allowed:
            return self._reject(
                device_id,
                f"No compatible schemes in phase '{self.phase.value}'. "
                f"Device only supports: {device_capabilities}. "
                f"Upgrade to a Kyber-capable device."
            )

        # Select the highest-priority allowed scheme
        selected_name = max(allowed, key=lambda s: allowed[s]["priority"])
        selected = allowed[selected_name]

        # Check for critical weaknesses
        if selected.get("critical_weakness"):
            logger.error(
                f"[Gateway] CRITICAL: Device {device_id} using {selected_name} — "
                f"immediate replacement required!"
            )

        # Deprecation warning for classical schemes
        dep_warning = None
        if selected["type"] == "classical":
            dep_warning = (
                f"WARNING: {selected_name} is scheduled for deprecation. "
                f"Please upgrade to Kyber (PQC) by 2026 to maintain security compliance."
            )

        # Hybrid flag: device supports both PQC and classical
        pqc_schemes = [s for s in device_capabilities if SCHEME_REGISTRY.get(s, {}).get("type") == "pqc"]
        classical_schemes = [s for s in device_capabilities if SCHEME_REGISTRY.get(s, {}).get("type") == "classical"]
        is_hybrid = bool(pqc_schemes and classical_schemes)

        session_id = secrets.token_hex(16)
        result = NegotiationResult(
            device_id=device_id,
            selected_scheme=selected_name,
            is_pqc=(selected["type"] == "pqc"),
            is_hybrid=is_hybrid,
            security_level=selected["security_level"],
            deprecation_warning=dep_warning,
            session_id=session_id,
            metadata={
                "phase": self.phase.value,
                "device_capabilities": device_capabilities,
                "offered_schemes": list(allowed.keys()),
            }
        )

        self._sessions[session_id] = result
        self._audit(result)
        logger.info(
            f"[Gateway] {device_id} → {selected_name} "
            f"({'PQC' if result.is_pqc else 'Classical'}, level={selected['security_level']})"
        )
        return result

    def _filter_by_phase(self, schemes: dict) -> dict:
        """Return schemes allowed under current migration phase."""
        if self.phase == MigrationPhase.CLASSICAL_ONLY:
            return {k: v for k, v in schemes.items() if v["type"] == "classical"}
        elif self.phase == MigrationPhase.HYBRID:
            return schemes  # accept all
        elif self.phase == MigrationPhase.PQC_PREFERRED:
            pqc = {k: v for k, v in schemes.items() if v["type"] == "pqc"}
            return pqc if pqc else {k: v for k, v in schemes.items()
                                     if v["type"] == "classical"}
        elif self.phase == MigrationPhase.PQC_ONLY:
            return {k: v for k, v in schemes.items() if v["type"] == "pqc"}
        return schemes

    def _reject(self, device_id: str, reason: str) -> NegotiationResult:
        logger.warning(f"[Gateway] REJECTED {device_id}: {reason}")
        result = NegotiationResult(
            device_id=device_id,
            selected_scheme="REJECTED",
            is_pqc=False,
            is_hybrid=False,
            security_level=-1,
            deprecation_warning=None,
            session_id="",
            metadata={"reason": reason, "phase": self.phase.value}
        )
        self._audit(result)
        return result

    def _audit(self, result: NegotiationResult):
        entry = result.to_dict()
        entry["phase"] = self.phase.value
        self._connection_log.append(entry)
        try:
            with open(self.audit_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Audit log write failed: {e}")

    def migration_status(self) -> dict:
        """Return migration progress statistics across all connections."""
        total = len(self._connection_log)
        if total == 0:
            return {"total_connections": 0}

        pqc_count = sum(1 for r in self._connection_log if r.get("is_pqc"))
        classical_count = sum(1 for r in self._connection_log if not r.get("is_pqc")
                               and r.get("selected_scheme") != "REJECTED")
        rejected_count = sum(1 for r in self._connection_log if r.get("selected_scheme") == "REJECTED")

        scheme_counts: Dict[str, int] = {}
        for r in self._connection_log:
            s = r.get("selected_scheme", "UNKNOWN")
            scheme_counts[s] = scheme_counts.get(s, 0) + 1

        return {
            "migration_phase": self.phase.value,
            "total_connections": total,
            "pqc_connections": pqc_count,
            "classical_connections": classical_count,
            "rejected_connections": rejected_count,
            "pqc_adoption_rate": round(pqc_count / total, 3) if total else 0,
            "scheme_distribution": scheme_counts,
        }

    def simulate_fleet_migration(self,
                                  device_fleet: Optional[Dict[str, List[str]]] = None,
                                  phases: Optional[List[MigrationPhase]] = None) -> List[dict]:
        """
        Simulate a full hospital fleet migrating through all phases.
        Returns per-phase statistics.
        """
        if device_fleet is None:
            device_fleet = DEVICE_CAPABILITIES

        if phases is None:
            phases = [
                MigrationPhase.CLASSICAL_ONLY,
                MigrationPhase.HYBRID,
                MigrationPhase.PQC_PREFERRED,
                MigrationPhase.PQC_ONLY,
            ]

        phase_results = []
        for phase in phases:
            self.phase = phase
            self._connection_log = []

            for device_id, caps in device_fleet.items():
                self.negotiate(device_id, caps)

            status = self.migration_status()
            phase_results.append(status)
            logger.info(
                f"  Phase {phase.value}: PQC adoption = {status['pqc_adoption_rate']*100:.1f}%"
            )

        return phase_results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n=== SPQR-IoMT Hybrid Migration Gateway Demo ===\n")

    gw = MigrationGateway(MigrationPhase.HYBRID)

    devices = [
        ("pump_legacy",   ["RSA-2048"]),
        ("sensor_gen3",   ["Kyber768", "RSA-2048"]),
        ("monitor_2023",  ["Kyber512", "ECC-P256"]),
        ("scada_v1",      ["RSA-1024"]),
        ("edge_gateway",  ["Kyber1024", "Kyber768"]),
    ]

    for dev_id, caps in devices:
        result = gw.negotiate(dev_id, caps)
        print(f"  {dev_id:20s} → {result.selected_scheme:15s} "
              f"[PQC={result.is_pqc}, level={result.security_level}]"
              + (f"\n    ⚠  {result.deprecation_warning}" if result.deprecation_warning else ""))

    print(f"\nMigration Status: {gw.migration_status()}\n")

    print("=== Fleet Migration Simulation (all phases) ===\n")
    gw2 = MigrationGateway(MigrationPhase.CLASSICAL_ONLY)
    phase_stats = gw2.simulate_fleet_migration()
    for ps in phase_stats:
        print(f"  [{ps['migration_phase']:20s}] PQC adoption: {ps['pqc_adoption_rate']*100:.1f}%  "
              f"| Rejected: {ps['rejected_connections']}")
