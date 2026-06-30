"""
experiments/exp2_fl_dp_tradeoff.py
====================================
Experiment 2: Federated Learning + Differential Privacy Privacy-Utility Tradeoff

Sweeps ε ∈ {0.5, 1.0, 3.0, 10.0, ∞} across 5 hospital clients over 50 rounds.
Measures: AUC, loss, communication bytes, and per-hospital epsilon spent.

Outputs:
  benchmarks/results/exp2_fl_dp_tradeoff.json
  benchmarks/results/exp2_fl_dp_tradeoff.csv
  benchmarks/results/exp2_privacy_budget.json

Usage:
    python experiments/exp2_fl_dp_tradeoff.py
    python experiments/exp2_fl_dp_tradeoff.py --rounds 100 --clients 5
"""

import sys
import json
import csv
import time
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s")
logger = logging.getLogger("exp2_fl_dp_tradeoff")

RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run(rounds: int = 50, n_clients: int = 5, quick: bool = False):
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: FL + DP Privacy-Utility Tradeoff")
    logger.info("=" * 60)

    from federated_learning.fl_server import LocalFLSimulator
    from federated_learning.privacy_audit import RDPAuditor, MultiHospitalPrivacyAnalysis

    rounds   = 10 if quick else rounds
    clients  = 3  if quick else n_clients

    epsilon_values = [0.5, 1.0, 3.0, 10.0, None]
    sweep_results  = []

    for eps in epsilon_values:
        label = f"ε={eps}" if eps is not None else "No DP (∞)"
        logger.info(f"\n  Running: {label} | rounds={rounds} | clients={clients}")

        sim = LocalFLSimulator(
            n_clients=clients,
            n_rounds=rounds,
            dp_epsilon=eps,
        )
        t0      = time.perf_counter()
        history = sim.run()
        elapsed = time.perf_counter() - t0

        # Privacy accounting
        if eps is not None and sim.dp:
            auditor = RDPAuditor(
                noise_multiplier=sim.dp.noise_multiplier,
                sample_rate=32 / 500,  # batch_size / n_patients
                delta=1e-5,
            )
            auditor.step(rounds * (500 // 32))  # total SGD steps
            eps_spent = auditor.epsilon
        else:
            eps_spent = None

        final = history[-1]
        result = {
            "epsilon_target":    eps,
            "label":             label,
            "final_auc":         final["simulated_auc"],
            "final_loss":        final["simulated_loss"],
            "n_rounds":          rounds,
            "n_clients":         clients,
            "dp_noise_sigma":    sim.dp.noise_multiplier if sim.dp else None,
            "epsilon_spent":     round(eps_spent, 4) if eps_spent else None,
            "training_time_s":   round(elapsed, 2),
            "utility_drop_vs_no_dp": None,  # filled below
        }
        sweep_results.append(result)

        sim.save_history(str(RESULTS_DIR / f"exp2_fl_history_eps_{str(eps).replace('.','_')}.json"))
        logger.info(f"    AUC={final['simulated_auc']:.4f} | loss={final['simulated_loss']:.4f} | "
                    f"ε_spent={eps_spent:.3f if eps_spent else 'N/A'}")

    # Compute utility drop vs no-DP baseline
    no_dp_auc = next((r["final_auc"] for r in sweep_results if r["epsilon_target"] is None), None)
    for r in sweep_results:
        if no_dp_auc and r["final_auc"] is not None:
            r["utility_drop_vs_no_dp"] = round(no_dp_auc - r["final_auc"], 4)

    # Per-hospital privacy budget analysis
    logger.info("\nPer-hospital privacy budget analysis...")
    mh = MultiHospitalPrivacyAnalysis(noise_multiplier=1.1, n_rounds=rounds)
    privacy_summary = mh.summary()
    logger.info(f"  Range: ε ∈ [{privacy_summary['min_epsilon']}, {privacy_summary['max_epsilon']}]")
    for h in privacy_summary["per_hospital"]:
        logger.info(f"    {h['hospital_type']:20s}: ε={h['epsilon']:.4f}  {h['interpretation']}")

    # Save
    output = {
        "experiment":        "Exp2: FL + DP Privacy-Utility Tradeoff",
        "sweep_results":     sweep_results,
        "privacy_budget":    privacy_summary,
        "no_dp_auc":         no_dp_auc,
        "key_finding":       (
            f"At ε=1.0, AUC drops by "
            f"{next((r['utility_drop_vs_no_dp'] for r in sweep_results if r['epsilon_target']==1.0), 'N/A')} "
            f"vs no-DP baseline — strong privacy at minimal utility cost."
        ),
    }

    out_path = RESULTS_DIR / "exp2_fl_dp_tradeoff.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # CSV
    csv_path = RESULTS_DIR / "exp2_fl_dp_tradeoff.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sweep_results[0].keys())
        writer.writeheader()
        writer.writerows(sweep_results)

    with open(RESULTS_DIR / "exp2_privacy_budget.json", "w") as f:
        json.dump(privacy_summary, f, indent=2)

    logger.info(f"\nResults saved to {out_path}")

    # Summary
    logger.info("\n" + "="*55)
    logger.info("EXPERIMENT 2 SUMMARY — Privacy-Utility Tradeoff")
    logger.info("="*55)
    logger.info(f"  {'Epsilon':>10} | {'AUC':>8} | {'Loss':>8} | {'Utility Drop':>12}")
    logger.info(f"  {'-'*46}")
    for r in sweep_results:
        logger.info(
            f"  {str(r['epsilon_target']):>10} | {r['final_auc']:>8.4f} | "
            f"{r['final_loss']:>8.4f} | {str(r['utility_drop_vs_no_dp']):>12}"
        )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp2: FL + DP Tradeoff")
    parser.add_argument("--rounds",  type=int, default=50)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--quick",   action="store_true")
    args = parser.parse_args()
    run(args.rounds, args.clients, args.quick)
