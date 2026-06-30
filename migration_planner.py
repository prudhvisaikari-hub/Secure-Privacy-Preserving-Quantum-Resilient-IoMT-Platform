"""
hybrid_migration/migration_planner.py
======================================
Phased RSA → PQC migration planner for hospital IoMT fleets.
Models the real-world rollout timeline, cost, risk, and compliance
status of migrating from classical to post-quantum cryptography.

Produces:
  - Per-phase device status counts
  - Cost projection over time
  - Compliance risk score (HIPAA/NIST alignment)
  - Recommended migration schedule

Usage:
    planner = MigrationPlanner(n_devices=200, budget_usd=50000)
    plan = planner.generate_plan()
    planner.print_plan(plan)
    planner.save_plan(plan, "docs/migration_plan.json")
"""

import json
import math
import time
import logging
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device type registry
# ---------------------------------------------------------------------------

DEVICE_TYPES: Dict[str, dict] = {
    "infusion_pump":        {"count_frac": 0.20, "crypto": "RSA-2048",  "replaceable": False, "upgrade_cost": 200,  "criticality": "high"},
    "vitals_monitor":       {"count_frac": 0.25, "crypto": "RSA-2048",  "replaceable": True,  "upgrade_cost": 150,  "criticality": "high"},
    "ecg_sensor":           {"count_frac": 0.15, "crypto": "ECC-P256",  "replaceable": True,  "upgrade_cost": 100,  "criticality": "medium"},
    "edge_gateway":         {"count_frac": 0.10, "crypto": "RSA-4096",  "replaceable": True,  "upgrade_cost": 500,  "criticality": "critical"},
    "nurse_call_system":    {"count_frac": 0.08, "crypto": "RSA-1024",  "replaceable": False, "upgrade_cost": 800,  "criticality": "low"},
    "scada_controller":     {"count_frac": 0.05, "crypto": "None",      "replaceable": False, "upgrade_cost": 2000, "criticality": "critical"},
    "wireless_sensor_gen3": {"count_frac": 0.17, "crypto": "Kyber512",  "replaceable": True,  "upgrade_cost": 50,   "criticality": "medium"},
}

MIGRATION_PHASES = [
    {
        "phase": 1, "name": "Inventory & Assessment",
        "months": 2, "actions": [
            "Enumerate all devices and cryptographic capabilities",
            "Classify by criticality and replaceability",
            "Identify RSA-1024 / no-crypto devices (immediate risk)",
            "Establish key management infrastructure",
        ],
        "pqc_migration_fraction": 0.0,
    },
    {
        "phase": 2, "name": "Quick Wins — Edge Gateways",
        "months": 2, "actions": [
            "Upgrade all edge gateways to Kyber768 (software update)",
            "Deploy hybrid RSA+Kyber gateway for legacy device support",
            "Retire RSA-1024 devices or isolate on VLAN",
            "Enable PQC certificate authority (CA) for new devices",
        ],
        "pqc_migration_fraction": 0.12,
    },
    {
        "phase": 3, "name": "New Device Procurement",
        "months": 3, "actions": [
            "All new device purchases must support Kyber512+",
            "Update procurement policy and vendor requirements",
            "Deploy PQC firmware update for gen-2 sensor fleet",
            "Pilot PQC telemetry protocol on one ward",
        ],
        "pqc_migration_fraction": 0.35,
    },
    {
        "phase": 4, "name": "Fleet-Wide Migration",
        "months": 4, "actions": [
            "Phased OTA firmware rollout (20% per month)",
            "Legacy device gateway wrapper deployment",
            "Penetration testing of migrated devices",
            "Staff training on new key management procedures",
        ],
        "pqc_migration_fraction": 0.75,
    },
    {
        "phase": 5, "name": "Completion & Hardening",
        "months": 3, "actions": [
            "Decommission remaining RSA-only devices",
            "Achieve 100% PQC coverage or gateway-protected coverage",
            "External audit and NIST compliance certification",
            "Document migration for HIPAA/MDR compliance record",
        ],
        "pqc_migration_fraction": 1.0,
    },
]


# ---------------------------------------------------------------------------
# Risk model
# ---------------------------------------------------------------------------

def quantum_risk_score(years_from_now: int = 0) -> float:
    """
    Probability that a CRQC capable of breaking RSA-2048 exists
    within `years_from_now` years, based on expert consensus.
    Uses a logistic model calibrated to Mosca's inequality estimates.
    """
    # Expert consensus: 50% chance by 2035 (≈10 years from 2025)
    # Logistic: p(t) = 1 / (1 + exp(-k*(t - t_half)))
    t_half = 10.0  # years from 2025
    k      = 0.4
    p = 1 / (1 + math.exp(-k * (years_from_now - t_half)))
    return round(p, 4)


def compliance_risk(pqc_fraction: float, has_rsa1024: bool = False) -> str:
    """
    Assess HIPAA / NIST SP 800-131A compliance risk level.
    NIST deprecates RSA-2048 after 2030.
    """
    if has_rsa1024:
        return "CRITICAL — RSA-1024 is immediately non-compliant (NIST deprecated 2013)"
    if pqc_fraction < 0.25:
        return "HIGH — Less than 25% PQC; exposure to HNDL attacks now"
    if pqc_fraction < 0.60:
        return "MEDIUM — Partial migration; complete before 2030 NIST deadline"
    if pqc_fraction < 1.0:
        return "LOW — Strong progress; finalise migration and document for audit"
    return "COMPLIANT — Full PQC coverage achieved"


# ---------------------------------------------------------------------------
# Migration cost model
# ---------------------------------------------------------------------------

@dataclass
class MigrationCostEstimate:
    hardware_usd:   float = 0.0
    software_usd:   float = 0.0
    labour_usd:     float = 0.0
    testing_usd:    float = 0.0
    training_usd:   float = 0.0
    contingency_usd: float = 0.0
    total_usd:      float = 0.0

    def as_dict(self) -> dict:
        return {k: round(v, 2) for k, v in asdict(self).items()}


# ---------------------------------------------------------------------------
# Migration Planner
# ---------------------------------------------------------------------------

class MigrationPlanner:
    """
    Generates a phased PQC migration plan for a hospital IoMT fleet.

    Args:
        n_devices:   Total number of IoMT devices in the hospital.
        budget_usd:  Available migration budget (USD).
        staff_rate:  IT staff hourly rate (USD/hr, default $75).
        start_year:  Migration start year (for risk projection).
    """

    def __init__(self,
                 n_devices:  int   = 150,
                 budget_usd: float = 50_000,
                 staff_rate: float = 75.0,
                 start_year: int   = 2025):
        self.n_devices  = n_devices
        self.budget     = budget_usd
        self.staff_rate = staff_rate
        self.start_year = start_year

    def _device_inventory(self) -> List[dict]:
        inventory = []
        for dev_type, spec in DEVICE_TYPES.items():
            count = max(1, int(self.n_devices * spec["count_frac"]))
            inventory.append({
                "type":             dev_type,
                "count":            count,
                "current_crypto":   spec["crypto"],
                "is_pqc":           spec["crypto"].startswith("Kyber"),
                "replaceable":      spec["replaceable"],
                "upgrade_cost_ea":  spec["upgrade_cost"],
                "total_cost":       count * spec["upgrade_cost"],
                "criticality":      spec["criticality"],
            })
        return inventory

    def _cost_estimate(self, inventory: List[dict]) -> MigrationCostEstimate:
        hw  = sum(d["total_cost"] for d in inventory if not d["is_pqc"])
        sw  = self.n_devices * 30        # $30/device for OTA infrastructure
        lab = (self.n_devices * 2) * self.staff_rate  # 2 hr/device avg
        tst = hw * 0.15                  # 15% of hardware for testing
        trn = 5 * 8 * self.staff_rate    # 5 staff × 8 hr training
        tot = hw + sw + lab + tst + trn
        cng = tot * 0.10                 # 10% contingency
        return MigrationCostEstimate(
            hardware_usd=round(hw, 2), software_usd=round(sw, 2),
            labour_usd=round(lab, 2),  testing_usd=round(tst, 2),
            training_usd=round(trn, 2), contingency_usd=round(cng, 2),
            total_usd=round(tot + cng, 2)
        )

    def _phase_device_status(self, inventory: List[dict], pqc_frac: float) -> dict:
        n_pqc_now = sum(d["count"] for d in inventory if d["is_pqc"])
        n_total   = sum(d["count"] for d in inventory)
        n_target  = int(n_total * pqc_frac)
        migrated  = max(n_pqc_now, n_target)
        legacy    = n_total - migrated
        return {
            "total_devices":     n_total,
            "pqc_migrated":      migrated,
            "legacy_remaining":  legacy,
            "pqc_fraction":      round(migrated / max(n_total, 1), 3),
        }

    def generate_plan(self) -> dict:
        inventory = self._device_inventory()
        cost      = self._cost_estimate(inventory)
        has_rsa1024 = any(d["current_crypto"] == "RSA-1024" for d in inventory)

        phases_out = []
        cumulative_month = 0
        for ph in MIGRATION_PHASES:
            cumulative_month += ph["months"]
            year_offset = cumulative_month / 12
            status = self._phase_device_status(inventory, ph["pqc_migration_fraction"])

            phases_out.append({
                "phase":         ph["phase"],
                "name":          ph["name"],
                "duration_months": ph["months"],
                "end_month":     cumulative_month,
                "end_year":      self.start_year + year_offset,
                "actions":       ph["actions"],
                "device_status": status,
                "quantum_risk_at_end": quantum_risk_score(int(year_offset)),
                "compliance_risk": compliance_risk(status["pqc_fraction"], has_rsa1024 and ph["phase"] == 1),
            })

        total_months = sum(p["months"] for p in MIGRATION_PHASES)
        return {
            "hospital_fleet": {
                "n_devices": self.n_devices,
                "inventory": inventory,
                "has_rsa1024_devices": has_rsa1024,
            },
            "cost_estimate": cost.as_dict(),
            "budget_usd": self.budget,
            "budget_sufficient": cost.total_usd <= self.budget,
            "total_duration_months": total_months,
            "completion_year": self.start_year + total_months / 12,
            "phases": phases_out,
            "risk_summary": {
                "current_quantum_risk": quantum_risk_score(0),
                "quantum_risk_at_completion": quantum_risk_score(int(total_months / 12)),
                "immediate_actions_required": [
                    d["type"] for d in inventory
                    if d["current_crypto"] in ("RSA-1024", "None")
                ],
            },
            "recommendations": self._recommendations(inventory, cost),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _recommendations(self, inventory: List[dict], cost: MigrationCostEstimate) -> List[str]:
        recs = []
        rsa1024 = [d["type"] for d in inventory if d["current_crypto"] == "RSA-1024"]
        if rsa1024:
            recs.append(f"IMMEDIATE: Isolate or replace RSA-1024 devices: {rsa1024}")
        if cost.total_usd > self.budget:
            overage = cost.total_usd - self.budget
            recs.append(f"Budget gap: ${overage:,.0f}. Prioritise edge gateways and critical devices first.")
        recs += [
            "Deploy hybrid RSA+Kyber gateways immediately to protect legacy devices now.",
            "Mandate Kyber512+ in all new device procurement contracts.",
            "Use liboqs-based Kyber for software-upgradeable devices (zero hardware cost).",
            "Apply for NHS Cyber Security Programme / HSCC grants for PQC migration funding.",
        ]
        return recs

    def print_plan(self, plan: dict):
        print(f"\n{'='*60}")
        print(f"  SPQR-IoMT Migration Plan — {plan['hospital_fleet']['n_devices']} devices")
        print(f"{'='*60}")
        print(f"  Total cost:    ${plan['cost_estimate']['total_usd']:>10,.2f}")
        print(f"  Budget:        ${plan['budget_usd']:>10,.2f}  {'✓' if plan['budget_sufficient'] else '✗ OVERAGE'}")
        print(f"  Duration:      {plan['total_duration_months']} months (complete {plan['completion_year']:.1f})")
        print(f"\n  {'Phase':>5} | {'Name':30s} | {'PQC%':>6} | {'Risk':40s}")
        print(f"  {'-'*90}")
        for ph in plan["phases"]:
            ds = ph["device_status"]
            print(f"  {ph['phase']:>5} | {ph['name']:30s} | {ds['pqc_fraction']*100:>5.0f}% | {ph['compliance_risk'][:40]}")
        print(f"\n  Immediate actions required: {plan['risk_summary']['immediate_actions_required'] or 'None'}")
        print()

    def save_plan(self, plan: dict, path: str = "docs/migration_plan.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(plan, f, indent=2, default=str)
        logger.info(f"Migration plan saved to {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    planner = MigrationPlanner(n_devices=150, budget_usd=50_000)
    plan = planner.generate_plan()
    planner.print_plan(plan)
    planner.save_plan(plan)
