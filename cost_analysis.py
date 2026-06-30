"""
qkd_comparison/cost_analysis.py
================================
Detailed cost/feasibility comparison of QKD vs PQC deployment
for hospital IoMT networks.

Covers:
  - CAPEX / OPEX breakdown (hardware, fiber, maintenance)
  - Scalability modelling (O(n) PQC vs O(n²) QKD links)
  - Key generation rate comparison
  - Total Cost of Ownership (TCO) over 3, 5, 10 years
  - Break-even analysis
  - Recommendation matrix by hospital size

Usage:
    from qkd_comparison.cost_analysis import CostAnalyzer
    ca = CostAnalyzer(n_hospitals=10, avg_distance_km=15)
    report = ca.full_report()
    ca.print_report(report)
    ca.save_report(report, "benchmarks/results/qkd_vs_pqc_cost.json")
"""

import json
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost parameters (USD, 2025 estimates)
# ---------------------------------------------------------------------------

@dataclass
class QKDCostParams:
    """QKD system cost parameters."""
    transmitter_per_node_usd:  float = 80_000   # Alice + Bob QKD hardware
    receiver_per_node_usd:     float = 70_000
    fiber_install_per_km_usd:  float = 5_000    # dedicated dark fiber
    fiber_lease_per_km_yr_usd: float = 1_200    # if leasing
    repeater_per_unit_usd:     float = 250_000  # quantum repeater (80 km+)
    annual_maintenance_pct:    float = 0.12     # 12% of CAPEX/yr
    calibration_hrs_per_yr:    float = 200      # technician hours/yr/node
    tech_rate_usd_hr:          float = 120      # specialist rate

@dataclass
class PQCCostParams:
    """PQC (Kyber) system cost parameters."""
    hsm_per_node_usd:          float = 800      # Hardware Security Module
    software_license_per_node: float = 50       # liboqs / commercial PQC lib
    integration_hrs_per_node:  float = 8        # dev hours per device
    dev_rate_usd_hr:           float = 90
    annual_maintenance_pct:    float = 0.05     # 5% of CAPEX/yr
    firmware_update_hrs:       float = 2        # IT hours per device/yr


QKD_DEFAULTS  = QKDCostParams()
PQC_DEFAULTS  = PQCCostParams()

HOSPITAL_PROFILES = {
    "small":   {"n_nodes": 3,  "avg_dist_km": 5,   "budget_usd": 200_000},
    "medium":  {"n_nodes": 8,  "avg_dist_km": 12,  "budget_usd": 1_000_000},
    "large":   {"n_nodes": 20, "avg_dist_km": 20,  "budget_usd": 5_000_000},
    "regional":{"n_nodes": 50, "avg_dist_km": 40,  "budget_usd": 20_000_000},
}


# ---------------------------------------------------------------------------
# QKD Cost Model
# ---------------------------------------------------------------------------

class QKDCostModel:

    def __init__(self, n_nodes: int, avg_dist_km: float,
                 params: QKDCostParams = QKD_DEFAULTS,
                 use_leased_fiber: bool = False):
        self.n       = n_nodes
        self.dist    = avg_dist_km
        self.p       = params
        self.leased  = use_leased_fiber
        self.n_links = n_nodes * (n_nodes - 1) // 2  # full-mesh links

    def needs_repeaters(self) -> bool:
        return self.dist > 80

    def n_repeaters_per_link(self) -> int:
        if not self.needs_repeaters():
            return 0
        return math.ceil(self.dist / 80) - 1

    def capex(self) -> dict:
        hardware  = self.n * (self.p.transmitter_per_node_usd + self.p.receiver_per_node_usd)
        if self.leased:
            fiber = 0.0  # no CAPEX for leased fiber
        else:
            fiber = self.n_links * self.dist * self.p.fiber_install_per_km_usd
        repeaters = (self.n_links * self.n_repeaters_per_link() * self.p.repeater_per_unit_usd
                     if self.needs_repeaters() else 0)
        total = hardware + fiber + repeaters
        return {
            "hardware_usd":   round(hardware, 2),
            "fiber_usd":      round(fiber, 2),
            "repeaters_usd":  round(repeaters, 2),
            "total_capex_usd": round(total, 2),
        }

    def annual_opex(self) -> dict:
        cap   = self.capex()["total_capex_usd"]
        maint = cap * self.p.annual_maintenance_pct
        calib = self.n * self.p.calibration_hrs_per_yr * self.p.tech_rate_usd_hr
        fiber_lease = (self.n_links * self.dist * self.p.fiber_lease_per_km_yr_usd
                       if self.leased else 0)
        total = maint + calib + fiber_lease
        return {
            "maintenance_usd":  round(maint, 2),
            "calibration_usd":  round(calib, 2),
            "fiber_lease_usd":  round(fiber_lease, 2),
            "total_opex_yr_usd": round(total, 2),
        }

    def tco(self, years: int = 5) -> float:
        return self.capex()["total_capex_usd"] + self.annual_opex()["total_opex_yr_usd"] * years

    def key_rate_bps(self) -> float:
        """Approximate secret key rate per link (BB84, 1 GHz source)."""
        from qkd_comparison.channel_noise import FiberChannel
        ch = FiberChannel(self.dist)
        return ch.secure_key_rate(source_rate_hz=1e9)

    def full_breakdown(self, years: int = 5) -> dict:
        cap  = self.capex()
        opex = self.annual_opex()
        return {
            "technology":       "QKD (BB84, single-photon fiber)",
            "n_nodes":          self.n,
            "n_links":          self.n_links,
            "avg_distance_km":  self.dist,
            "needs_repeaters":  self.needs_repeaters(),
            "capex":            cap,
            "annual_opex":      opex,
            f"tco_{years}yr_usd": round(self.tco(years), 2),
            "key_rate_per_link_bps": round(self.key_rate_bps(), 1),
            "max_range_km":     80 if not self.needs_repeaters() else self.dist,
            "dedicated_fiber_required": True,
            "pq_security_type": "Information-theoretic (unconditional)",
            "operational_complexity": "Very High",
            "scalability":      f"O(n²) = {self.n_links} links for {self.n} nodes",
        }


# ---------------------------------------------------------------------------
# PQC Cost Model
# ---------------------------------------------------------------------------

class PQCCostModel:

    def __init__(self, n_nodes: int,
                 params: PQCCostParams = PQC_DEFAULTS,
                 use_hsm: bool = True):
        self.n      = n_nodes
        self.p      = params
        self.use_hsm = use_hsm

    def capex(self) -> dict:
        hardware = self.n * (self.p.hsm_per_node_usd if self.use_hsm else 0)
        software = self.n * self.p.software_license_per_node
        integration = self.n * self.p.integration_hrs_per_node * self.p.dev_rate_usd_hr
        total = hardware + software + integration
        return {
            "hardware_hsm_usd":   round(hardware, 2),
            "software_usd":       round(software, 2),
            "integration_usd":    round(integration, 2),
            "total_capex_usd":    round(total, 2),
        }

    def annual_opex(self) -> dict:
        cap   = self.capex()["total_capex_usd"]
        maint = cap * self.p.annual_maintenance_pct
        updates = self.n * self.p.firmware_update_hrs * (self.p.dev_rate_usd_hr / 2)
        total = maint + updates
        return {
            "maintenance_usd":    round(maint, 2),
            "firmware_updates_usd": round(updates, 2),
            "total_opex_yr_usd":  round(total, 2),
        }

    def tco(self, years: int = 5) -> float:
        return self.capex()["total_capex_usd"] + self.annual_opex()["total_opex_yr_usd"] * years

    def key_rate_bps(self) -> float:
        """Kyber512 keygen rate on RPi4 class device (~3200 keygens/sec)."""
        return 3200 * 256  # 256-bit shared secret per keygen

    def full_breakdown(self, years: int = 5) -> dict:
        cap  = self.capex()
        opex = self.annual_opex()
        return {
            "technology":         "PQC (CRYSTALS-Kyber512, NIST FIPS 203)",
            "n_nodes":            self.n,
            "n_links":            "Unlimited (software)",
            "avg_distance_km":    "Unlimited (IP network)",
            "needs_repeaters":    False,
            "capex":              cap,
            "annual_opex":        opex,
            f"tco_{years}yr_usd": round(self.tco(years), 2),
            "key_rate_per_link_bps": round(self.key_rate_bps(), 1),
            "max_range_km":       "Unlimited",
            "dedicated_fiber_required": False,
            "pq_security_type":   "Computational (MLWE hardness, NIST Level 1+)",
            "operational_complexity": "Low",
            "scalability":        f"O(n) = {self.n} software installs",
        }


# ---------------------------------------------------------------------------
# Unified Analyzer
# ---------------------------------------------------------------------------

class CostAnalyzer:

    def __init__(self, n_nodes: int = 10, avg_distance_km: float = 15,
                 years: int = 5, use_leased_fiber: bool = False):
        self.n     = n_nodes
        self.dist  = avg_distance_km
        self.years = years
        self.qkd   = QKDCostModel(n_nodes, avg_distance_km,
                                   use_leased_fiber=use_leased_fiber)
        self.pqc   = PQCCostModel(n_nodes)

    def full_report(self) -> dict:
        qkd_data = self.qkd.full_breakdown(self.years)
        pqc_data = self.pqc.full_breakdown(self.years)

        tco_key   = f"tco_{self.years}yr_usd"
        qkd_tco   = qkd_data[tco_key]
        pqc_tco   = pqc_data[tco_key]
        ratio     = qkd_tco / max(pqc_tco, 1)

        # Break-even: when does QKD cost match PQC cost? (never, in practice)
        pqc_annual = self.pqc.annual_opex()["total_opex_yr_usd"]
        qkd_annual = self.qkd.annual_opex()["total_opex_yr_usd"]
        pqc_cap    = self.pqc.capex()["total_capex_usd"]
        qkd_cap    = self.qkd.capex()["total_capex_usd"]

        # Profile comparisons for all hospital sizes
        profile_comparison = []
        for size, profile in HOSPITAL_PROFILES.items():
            q = QKDCostModel(profile["n_nodes"], profile["avg_dist_km"])
            p = PQCCostModel(profile["n_nodes"])
            q_tco = q.tco(5)
            p_tco = p.tco(5)
            profile_comparison.append({
                "hospital_size":       size,
                "n_nodes":             profile["n_nodes"],
                "avg_dist_km":         profile["avg_dist_km"],
                "budget_usd":          profile["budget_usd"],
                "qkd_tco_5yr_usd":     round(q_tco, 2),
                "pqc_tco_5yr_usd":     round(p_tco, 2),
                "cost_ratio":          round(q_tco / max(p_tco, 1), 1),
                "qkd_within_budget":   q_tco <= profile["budget_usd"],
                "pqc_within_budget":   p_tco <= profile["budget_usd"],
                "recommendation":      "PQC" if p_tco < q_tco else "QKD",
            })

        return {
            "analysis_params": {
                "n_nodes": self.n,
                "avg_distance_km": self.dist,
                "years": self.years,
            },
            "qkd": qkd_data,
            "pqc": pqc_data,
            "comparison": {
                f"qkd_tco_{self.years}yr_usd": qkd_tco,
                f"pqc_tco_{self.years}yr_usd": pqc_tco,
                "cost_ratio_qkd_vs_pqc":       round(ratio, 1),
                "pqc_annual_saving_vs_qkd":     round(qkd_annual - pqc_annual, 2),
                "qkd_key_rate_bps":             qkd_data["key_rate_per_link_bps"],
                "pqc_key_rate_bps":             pqc_data["key_rate_per_link_bps"],
                "key_rate_ratio_pqc_vs_qkd":    round(
                    pqc_data["key_rate_per_link_bps"] /
                    max(qkd_data["key_rate_per_link_bps"], 1), 1),
            },
            "hospital_profiles": profile_comparison,
            "recommendation": self._recommendation(ratio, qkd_data, pqc_data),
        }

    def _recommendation(self, cost_ratio: float, qkd: dict, pqc: dict) -> dict:
        return {
            "primary":   "PQC (CRYSTALS-Kyber)",
            "rationale": (
                f"PQC is {cost_ratio:.0f}× cheaper than QKD for {self.n} nodes over {self.years} years. "
                "Kyber provides NIST-standardised quantum resistance (FIPS 203) with "
                "unlimited range, zero dedicated infrastructure, and low operational complexity. "
                "QKD provides unconditional information-theoretic security but is only "
                "justified for ultra-high-security point-to-point links with unlimited budget."
            ),
            "when_to_use_qkd": [
                "Government / military links requiring unconditional security",
                "Regulatory mandate for information-theoretic security",
                "Budget > $500K per link and distance < 80 km",
                "Long-term classified data requiring 100+ year security",
            ],
            "when_to_use_pqc": [
                "All standard hospital IoMT deployments",
                "Any deployment with > 5 nodes (mesh QKD becomes prohibitively expensive)",
                "Deployments with budget constraints",
                "Deployments requiring security over any network topology",
                "Immediate quantum threat mitigation (deploy today vs. QKD lead times of 12–18 months)",
            ],
        }

    def print_report(self, report: dict):
        cmp = report["comparison"]
        qkd = report["qkd"]
        pqc = report["pqc"]
        y   = self.years

        print(f"\n{'='*65}")
        print(f"  QKD vs PQC Cost Analysis  ({self.n} nodes, {self.dist} km avg, {y} yr)")
        print(f"{'='*65}")
        print(f"  {'Metric':40s} {'QKD':>10} {'PQC':>10}")
        print(f"  {'-'*62}")

        metrics = [
            ("CAPEX (USD)",             qkd["capex"]["total_capex_usd"],     pqc["capex"]["total_capex_usd"]),
            ("Annual OPEX (USD/yr)",     qkd["annual_opex"]["total_opex_yr_usd"], pqc["annual_opex"]["total_opex_yr_usd"]),
            (f"TCO {y} years (USD)",     cmp[f"qkd_tco_{y}yr_usd"],          cmp[f"pqc_tco_{y}yr_usd"]),
            ("Key rate per link (bps)",  cmp["qkd_key_rate_bps"],             cmp["pqc_key_rate_bps"]),
            ("Dedicated fiber required", "YES",                               "NO"),
            ("Max range",               "80 km",                             "Unlimited"),
            ("Scalability",             f"O(n²)",                            "O(n)"),
        ]
        for label, qval, pval in metrics:
            qs = f"${qval:,.0f}" if isinstance(qval, (int, float)) and "YES" not in str(qval) else str(qval)
            ps = f"${pval:,.0f}" if isinstance(pval, (int, float)) and "NO"  not in str(pval) else str(pval)
            print(f"  {label:40s} {qs:>10} {ps:>10}")

        print(f"\n  Cost ratio (QKD/PQC): {cmp['cost_ratio_qkd_vs_pqc']}×")
        print(f"  Recommendation: {report['recommendation']['primary']}")
        print(f"\n  Profile comparison:")
        print(f"  {'Size':12s} {'Nodes':>6} {'QKD TCO':>12} {'PQC TCO':>12} {'Ratio':>7} {'Rec':>6}")
        print(f"  {'-'*55}")
        for p in report["hospital_profiles"]:
            print(f"  {p['hospital_size']:12s} {p['n_nodes']:>6} "
                  f"${p['qkd_tco_5yr_usd']:>11,.0f} ${p['pqc_tco_5yr_usd']:>11,.0f} "
                  f"{p['cost_ratio']:>6.0f}× {p['recommendation']:>6}")

    def save_report(self, report: dict, path: str = "benchmarks/results/qkd_vs_pqc_cost.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Cost analysis saved to {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ca = CostAnalyzer(n_nodes=10, avg_distance_km=15, years=5)
    report = ca.full_report()
    ca.print_report(report)
    ca.save_report(report)
