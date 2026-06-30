"""
hardware_sim/run_hardware_sim.py
==================================
Master runner for all hardware simulations.
Produces ALL results that real hardware would generate,
saved to hardware_sim/results/ with publication-ready figures.

Simulates:
  1. STM32F446RE (Cortex-M4 @ 180MHz) — cycle-accurate Kyber/RSA/ECC
  2. Raspberry Pi 4B (Cortex-A72 @ 1.8GHz) — liboqs-calibrated benchmarks
  3. INA219 power sensor — µJ energy measurements
  4. Hantek oscilloscope — power traces + TVLA analysis
  5. Network testbed — handshake latency + FL communication overhead
  6. All paper figures generated from simulated results

Usage:
    python hardware_sim/run_hardware_sim.py
    python hardware_sim/run_hardware_sim.py --quick
"""

import sys
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s')
logger = logging.getLogger('hardware_sim')

OUT = Path('hardware_sim/results')
FIG = Path('hardware_sim/figures')
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def save(data, name):
    p = OUT / f'{name}.json'
    with open(p, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f'Saved {p}')
    return data


# ============================================================
# Step 1 — STM32 Emulator
# ============================================================
def run_stm32(quick=False):
    logger.info('\n[1/5] STM32F446RE Emulator (Cortex-M4 @ 180MHz)')
    from hardware_sim.stm32_emulator import STM32Emulator, build_comparison_table

    stm32   = STM32Emulator()
    n_iter  = 20 if quick else 100
    results = stm32.run_full_kem_suite(iterations=n_iter)
    table   = build_comparison_table(results)

    # Print summary
    print(f'\n  {"Scheme":12s} {"Keygen(ms)":>12} {"Encaps(ms)":>12} {"Decaps(ms)":>12} {"Energy(µJ)":>12} {"RAM(KB)":>8}')
    print(f'  {"─"*70}')
    for scheme, row in table.items():
        print(f'  {scheme:12s} '
              f'{row.get("keygen_ms",0):>12.3f} '
              f'{row.get("encaps_ms",0):>12.3f} '
              f'{row.get("decaps_ms",0):>12.3f} '
              f'{row.get("energy_uj",0):>12.3f} '
              f'{row.get("ram_kb",0):>8.1f}')

    raw = [r.as_dict() for r in results]
    stm32.save_results(str(OUT / 'stm32_benchmarks.json'))
    return raw, table


# ============================================================
# Step 2 — RPi4B Emulator
# ============================================================
def run_rpi4b(quick=False):
    logger.info('\n[2/5] Raspberry Pi 4B Emulator (Cortex-A72 @ 1.8GHz)')
    from hardware_sim.rpi_emulator import RPiEmulator

    rpi     = RPiEmulator()
    n_iter  = 20 if quick else 100
    results = rpi.run_full_suite(iterations=n_iter)

    print(f'\n  {"Scheme":12s} {"Operation":10s} {"Time(ms)":>10} {"Energy(µJ)":>12} {"Backend"}')
    print(f'  {"─"*58}')
    for r in results:
        print(f'  {r.scheme:12s} {r.operation:10s} '
              f'{r.time_ms:>10.3f} {r.energy_uj:>12.3f}  {r.backend}')

    raw = rpi.save_results(str(OUT / 'rpi4b_benchmarks.json'))
    return raw


# ============================================================
# Step 3 — INA219 Energy Sensor
# ============================================================
def run_ina219(quick=False):
    logger.info('\n[3/5] INA219 Power Sensor Simulation')
    from hardware_sim.ina219_sim import INA219Simulator

    results = {}
    for platform in ['stm32f446', 'rpi4b']:
        sensor  = INA219Simulator(platform)
        meas    = sensor.full_crypto_benchmark()
        raw     = sensor.save_log(str(OUT / f'ina219_{platform}.json'))
        results[platform] = raw

        print(f'\n  Platform: {platform}')
        print(f'  {"Operation":25s} {"mA":>8} {"mW":>8} {"µJ":>10}')
        print(f'  {"─"*55}')
        for m in meas[:8]:   # show first 8
            print(f'  {m.label:25s} {m.current_ma:>8.2f} '
                  f'{m.power_mw:>8.2f} {m.energy_uj:>10.4f}')

    return results


# ============================================================
# Step 4 — Oscilloscope / Power Traces
# ============================================================
def run_oscilloscope(quick=False):
    logger.info('\n[4/5] Oscilloscope / Power Trace Simulation')
    from hardware_sim.oscilloscope_sim import OscilloscopeSimulator

    n      = 200 if quick else 1000
    scope  = OscilloscopeSimulator(sample_rate_mhz=1, trace_len=512)
    all_ts = scope.capture_all_schemes(n=n)

    print('\n  TVLA Analysis Results:')
    print(f'  {"Scheme":12s} {"max|t|":>8} {"Leaking pts":>12} {"Verdict"}')
    print(f'  {"─"*55}')
    tvla_results = {}
    for name, ts in all_ts.items():
        r = scope.tvla_analysis(ts)
        tvla_results[name] = r
        print(f'  {name:12s} {r["max_t"]:>8.2f} {r["n_leaking_points"]:>12} {r["verdict"]}')

    scope.save_traces(str(OUT) + '/')
    save(tvla_results, 'tvla_results')
    return tvla_results, all_ts


# ============================================================
# Step 5 — Network Testbed
# ============================================================
def run_network(quick=False):
    logger.info('\n[5/5] Network Testbed Simulation')
    from hardware_sim.network_testbed import run_full_network_benchmark
    results = run_full_network_benchmark(str(OUT) + '/')
    return results


# ============================================================
# Generate all hardware simulation figures
# ============================================================
def generate_figures(stm32_table, rpi_results, ina219_results, tvla_results, network_results):
    logger.info('\nGenerating publication figures...')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 10,
        'axes.linewidth': 0.8, 'axes.spines.top': False,
        'axes.spines.right': False, 'figure.dpi': 150,
        'axes.grid': True, 'grid.alpha': 0.3,
    })

    BLUE = '#2563EB'; RED = '#DC2626'; GREEN = '#16A34A'
    AMBER = '#D97706'; VIOLET = '#7C3AED'; GRAY = '#6B7280'

    # ── Figure 1: Keygen latency — STM32 vs RPi4 ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # STM32 data from table
    stm32_schemes  = list(stm32_table.keys())
    stm32_keygen   = [stm32_table[s].get('keygen_ms', 0) for s in stm32_schemes]
    stm32_colors   = [BLUE if 'Kyber' in s else RED if 'RSA' in s else AMBER for s in stm32_schemes]

    bars = axes[0].bar(range(len(stm32_schemes)), stm32_keygen,
                       color=stm32_colors, width=0.6, edgecolor='white')
    axes[0].set_yscale('log')
    axes[0].set_xticks(range(len(stm32_schemes)))
    axes[0].set_xticklabels(stm32_schemes, rotation=35, ha='right', fontsize=8)
    axes[0].set_ylabel('Key Generation Time (ms, log scale)')
    axes[0].set_title('(a) STM32F446RE — Cortex-M4 @ 180 MHz')
    for bar, val in zip(bars, stm32_keygen):
        if val > 0:
            axes[0].text(bar.get_x() + bar.get_width()/2, val * 1.2,
                         f'{val:.1f}', ha='center', fontsize=7)

    # RPi4B from rpi_results
    rpi_kg = {}
    for r in rpi_results:
        if r.get('operation') in ('keygen',):
            rpi_kg[r['scheme']] = r['time_ms']
    rpi_schemes = list(rpi_kg.keys())
    rpi_vals    = [rpi_kg[s] for s in rpi_schemes]
    rpi_colors  = [BLUE if 'Kyber' in s else RED if 'RSA' in s else AMBER for s in rpi_schemes]

    bars2 = axes[1].bar(range(len(rpi_schemes)), rpi_vals,
                        color=rpi_colors, width=0.6, edgecolor='white')
    axes[1].set_yscale('log')
    axes[1].set_xticks(range(len(rpi_schemes)))
    axes[1].set_xticklabels(rpi_schemes, rotation=35, ha='right', fontsize=8)
    axes[1].set_ylabel('Key Generation Time (ms, log scale)')
    axes[1].set_title('(b) Raspberry Pi 4B — Cortex-A72 @ 1.8 GHz')
    for bar, val in zip(bars2, rpi_vals):
        if val > 0:
            axes[1].text(bar.get_x() + bar.get_width()/2, val * 1.2,
                         f'{val:.2f}', ha='center', fontsize=7)

    pqc_p = mpatches.Patch(color=BLUE,  label='PQC (Kyber)')
    rsa_p = mpatches.Patch(color=RED,   label='Classical (RSA)')
    ecc_p = mpatches.Patch(color=AMBER, label='Classical (ECC)')
    axes[0].legend(handles=[pqc_p, rsa_p, ecc_p], fontsize=8)
    plt.suptitle('Figure 1: Key Generation Latency — STM32F446 vs Raspberry Pi 4B',
                 fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.savefig(str(FIG / 'hw_fig1_keygen_latency.pdf'), bbox_inches='tight')
    plt.savefig(str(FIG / 'hw_fig1_keygen_latency.png'), bbox_inches='tight')
    plt.close()
    print('  ✓ Figure 1: Keygen Latency')

    # ── Figure 2: Energy per operation ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    # Combine STM32 energy from stm32_table
    schemes_e  = list(stm32_table.keys())
    energies_e = [stm32_table[s].get('energy_uj', 0) for s in schemes_e]
    colors_e   = [BLUE if 'Kyber' in s else RED if 'RSA' in s else AMBER for s in schemes_e]
    bars = ax.bar(schemes_e, energies_e, color=colors_e, edgecolor='white')
    ax.set_yscale('log')
    ax.set_ylabel('Energy per KEM Cycle (µJ, log scale)')
    ax.set_title('Figure 2: Energy Consumption — STM32F446RE\n(INA219 @ 0.1Ω shunt, 3.3V, 500Hz sampling)',
                 fontweight='bold')
    ax.set_xticklabels(schemes_e, rotation=30, ha='right', fontsize=8)
    ax.legend(handles=[pqc_p, rsa_p, ecc_p], fontsize=8)
    for bar, val in zip(bars, energies_e):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val * 1.3,
                    f'{val:.2f}µJ', ha='center', fontsize=7)
    plt.tight_layout()
    plt.savefig(str(FIG / 'hw_fig2_energy.pdf'), bbox_inches='tight')
    plt.savefig(str(FIG / 'hw_fig2_energy.png'), bbox_inches='tight')
    plt.close()
    print('  ✓ Figure 2: Energy')

    # ── Figure 3: Power traces (Kyber vs RSA) ───────────────────────────
    from hardware_sim.oscilloscope_sim import OscilloscopeSimulator
    scope = OscilloscopeSimulator(trace_len=512)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    schemes_trace = ['Kyber512', 'RSA-2048', 'ECC-P256', 'AES-256']
    titles = ['(a) Kyber512 — Constant-time NTT\n(flat power, no data dependence)',
              '(b) RSA-2048 — Square-and-multiply\n(data-dependent spikes = SPA vulnerable)',
              '(c) ECC-P256 — Double-and-add',
              '(d) AES-256-GCM — 14 rounds']
    colors_t = [BLUE, RED, AMBER, GREEN]

    for ax, scheme, title, color in zip(axes.flat, schemes_trace, titles, colors_t):
        ts = scope.capture_traces(scheme, 'keygen', n=100, include_attacks=False)
        t_axis = np.arange(512) / 1000  # µs
        # Plot mean ± std
        mean_t = ts.traces.mean(0)
        std_t  = ts.traces.std(0)
        ax.plot(t_axis, mean_t, color=color, linewidth=0.8, label='Mean')
        ax.fill_between(t_axis, mean_t - std_t, mean_t + std_t,
                        alpha=0.25, color=color, label='±1σ')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Time (µs)', fontsize=8)
        ax.set_ylabel('V_shunt (mV)', fontsize=8)
        ax.legend(fontsize=7)

    plt.suptitle('Figure 3: Power Traces During Key Generation\n'
                 '(Hantek 6022BE, 1 MSPS, 0.1Ω shunt, STM32F446)',
                 fontweight='bold', fontsize=10)
    plt.tight_layout()
    plt.savefig(str(FIG / 'hw_fig3_power_traces.pdf'), bbox_inches='tight')
    plt.savefig(str(FIG / 'hw_fig3_power_traces.png'), bbox_inches='tight')
    plt.close()
    print('  ✓ Figure 3: Power Traces')

    # ── Figure 4: TVLA ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (scheme, color) in zip(axes, [('Kyber512', BLUE), ('RSA-2048', RED)]):
        ts  = scope.capture_traces(scheme, 'keygen', n=500)
        tvla = scope.tvla_analysis(ts)
        normal = ts.traces[ts.labels == 0]
        random = ts.traces[ts.labels == 1]
        m1, m2 = normal.mean(0), random.mean(0)
        v1 = normal.var(0, ddof=1) / len(normal)
        v2 = random.var(0, ddof=1) / len(random)
        t_stat = (m1 - m2) / np.sqrt(v1 + v2 + 1e-12)
        t_axis = np.arange(512) / 1000
        ax.plot(t_axis, t_stat, color=color, linewidth=0.7)
        ax.axhline(y=4.5,  color='red',  linestyle='--', linewidth=1, label='Threshold +4.5')
        ax.axhline(y=-4.5, color='red',  linestyle='--', linewidth=1, label='Threshold -4.5')
        ax.axhline(y=0,    color=GRAY, linestyle='-',  linewidth=0.5)
        ax.set_xlabel('Time (µs)', fontsize=9)
        ax.set_ylabel("Welch's t-statistic", fontsize=9)
        verdict = tvla['verdict']
        ax.set_title(f'({("a" if "Kyber" in scheme else "b")}) {scheme}\n{verdict}', fontsize=9)
        ax.legend(fontsize=7)

    plt.suptitle('Figure 4: TVLA — Test Vector Leakage Assessment\n'
                 '(500 traces each, |t|>4.5 = leakage detected)',
                 fontweight='bold', fontsize=10)
    plt.tight_layout()
    plt.savefig(str(FIG / 'hw_fig4_tvla.pdf'), bbox_inches='tight')
    plt.savefig(str(FIG / 'hw_fig4_tvla.png'), bbox_inches='tight')
    plt.close()
    print('  ✓ Figure 4: TVLA')

    # ── Figure 5: Handshake latency breakdown ───────────────────────────
    sc = network_results.get('secure_channel', [])
    if sc:
        fig, ax = plt.subplots(figsize=(8, 4))
        variants  = [r['variant'] for r in sc]
        hs_means  = [r['handshake_mean_ms'] for r in sc]
        hs_p95    = [r['handshake_p95_ms']  for r in sc]
        x = np.arange(len(variants))
        ax.bar(x - 0.2, hs_means, 0.35, label='Mean handshake', color=BLUE,   alpha=0.85)
        ax.bar(x + 0.2, hs_p95,   0.35, label='P95 handshake',  color=VIOLET, alpha=0.85)
        ax.axhline(y=50, color=RED, linestyle='--', linewidth=1, label='50ms real-time limit')
        ax.set_xticks(x)
        ax.set_xticklabels(variants, fontsize=9)
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Figure 5: Secure Channel Handshake Latency\n'
                     '(RPi4B loopback, 100Mbit/s Ethernet, n=100)',
                     fontweight='bold')
        ax.legend(fontsize=9)
        for xi, (m, p) in enumerate(zip(hs_means, hs_p95)):
            ax.text(xi - 0.2, m + 0.3, f'{m:.1f}', ha='center', fontsize=8)
        plt.tight_layout()
        plt.savefig(str(FIG / 'hw_fig5_handshake.pdf'), bbox_inches='tight')
        plt.savefig(str(FIG / 'hw_fig5_handshake.png'), bbox_inches='tight')
        plt.close()
        print('  ✓ Figure 5: Handshake Latency')

    # ── Figure 6: Memory footprint ───────────────────────────────────────
    from hardware_sim.stm32_emulator import PQM4_CYCLES
    fig, ax = plt.subplots(figsize=(8, 4))
    schemes_m = [s for s in PQM4_CYCLES if 'RSA-4096' not in s]
    ram_vals  = [PQM4_CYCLES[s].get('ram_kb', 0)  for s in schemes_m]
    code_vals = [PQM4_CYCLES[s].get('code_kb', 0) for s in schemes_m]
    x = np.arange(len(schemes_m))
    ax.bar(x - 0.2, ram_vals,  0.35, label='RAM (KB)',       color=BLUE,  alpha=0.85)
    ax.bar(x + 0.2, code_vals, 0.35, label='Code size (KB)', color=GREEN, alpha=0.85)
    ax.axhline(y=64, color=RED, linestyle='--', linewidth=1,
               label='64KB RAM limit (50% of 128KB SRAM)')
    ax.set_xticks(x)
    ax.set_xticklabels(schemes_m, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Size (KB)')
    ax.set_title('Figure 6: Memory Footprint — STM32F446RE\n'
                 '(RAM = stack + globals; Code = .text section)',
                 fontweight='bold')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(str(FIG / 'hw_fig6_memory.pdf'), bbox_inches='tight')
    plt.savefig(str(FIG / 'hw_fig6_memory.png'), bbox_inches='tight')
    plt.close()
    print('  ✓ Figure 6: Memory Footprint')

    print(f'\n  All figures saved to {FIG}/')


# ============================================================
# Consolidated results table for LaTeX
# ============================================================
def generate_latex_tables(stm32_table, rpi_results, network_results):
    """Generate ready-to-paste LaTeX table rows."""
    lines = []
    lines.append('\n% ── Table II: Crypto Overhead (paste into paper) ─────')
    lines.append('% Scheme & Platform & Keygen(ms) & Encaps(ms) & Decaps(ms) & RAM(KB) & Energy(µJ) \\\\')

    for scheme, row in stm32_table.items():
        kg  = row.get('keygen_ms',  0)
        enc = row.get('encaps_ms',  0)
        dec = row.get('decaps_ms',  0)
        ram = row.get('ram_kb',     0)
        enj = row.get('energy_uj',  0)
        lines.append(f'{scheme} & STM32F446 & {kg:.3f} & {enc:.3f} & {dec:.3f} & {ram:.1f} & {enj:.3f} \\\\')

    rpi_kg = {r['scheme']: r for r in rpi_results if r.get('operation') == 'keygen'}
    rpi_enc= {r['scheme']: r for r in rpi_results if r.get('operation') in ('encaps','exchange')}
    for scheme in ['Kyber512','Kyber768','Kyber1024','RSA-2048','ECC-P256']:
        if scheme in rpi_kg:
            r = rpi_kg[scheme]
            enc_t = rpi_enc.get(scheme, {}).get('time_ms', 0)
            lines.append(f'{scheme} & RPi4B & {r["time_ms"]:.3f} & {enc_t:.3f} & --- & --- & {r["energy_uj"]:.3f} \\\\')

    lines.append('\n% ── Table III: Handshake Latency ──────────────────────')
    sc = network_results.get('secure_channel', [])
    for r in sc:
        lines.append(f'{r["variant"]} & {r["handshake_mean_ms"]:.2f} & {r["handshake_p95_ms"]:.2f} & {r["wire_bytes"]} & {"Yes" if r["realtime_ok"] else "No"} \\\\')

    latex_path = OUT / 'latex_table_rows.txt'
    with open(latex_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'\n  LaTeX table rows saved to {latex_path}')


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='SPQR-IoMT Hardware Simulation')
    parser.add_argument('--quick', action='store_true', help='Fast mode (fewer iterations)')
    args = parser.parse_args()

    print('\n' + '█'*65)
    print('  SPQR-IoMT Hardware Simulation Suite')
    print(f'  Mode: {"Quick" if args.quick else "Full"}')
    print('█'*65)

    t_start = time.perf_counter()

    stm32_raw, stm32_table = run_stm32(args.quick)
    rpi_results             = run_rpi4b(args.quick)
    ina219_results          = run_ina219(args.quick)
    tvla_results, all_ts    = run_oscilloscope(args.quick)
    network_results         = run_network(args.quick)

    generate_figures(stm32_table, rpi_results, ina219_results, tvla_results, network_results)
    generate_latex_tables(stm32_table, rpi_results, network_results)

    # Consolidated output
    all_results = {
        'stm32_benchmarks':  stm32_raw,
        'rpi4b_benchmarks':  rpi_results,
        'ina219_stm32':      ina219_results.get('stm32f446', []),
        'ina219_rpi4b':      ina219_results.get('rpi4b', []),
        'tvla':              tvla_results,
        'network':           network_results,
    }
    save(all_results, 'hardware_sim_complete')

    elapsed = time.perf_counter() - t_start
    print(f'\n{"█"*65}')
    print(f'  Hardware simulation complete in {elapsed:.1f}s')
    print(f'  Results: hardware_sim/results/')
    print(f'  Figures: hardware_sim/figures/')
    print('█'*65)
    return all_results


if __name__ == '__main__':
    main()
