#!/usr/bin/env python3
"""
Generate Figures 1-5 for the GPU-accelerated fastp paper.
Reads all data from the collected CSV files — no hardcoded values.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from collections import defaultdict

ROOT = os.environ.get(
    'FASTP_D0BROMIR_ROOT',
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
OUT  = os.path.join(ROOT, 'docs/publication/figures')
os.makedirs(OUT, exist_ok=True)

BENCH_CSV = os.path.join(ROOT, 'benchmark_results/fastp-gpu_v1.3.3-d0bromir/vs_opengene_v1.3.3/galaxy_arm_a100/20260503_193146/full_benchmark_20260503_193146.csv')
FIG1_CSV  = os.path.join(OUT, 'fig1_breakdown.csv')
FIG2_CSV  = os.path.join(OUT, 'fig2_kernel_speedup.csv')
FIG3_CSV  = os.path.join(OUT, 'fig3_speculative_stats.csv')
FIG4_CSV  = os.path.join(OUT, 'fig4_transfer_overhead.csv')

# ──────────────────────────────────────────────────────────────
# Color palette (IEEE-friendly, colorblind-safe)
# ──────────────────────────────────────────────────────────────
C_OG     = '#4e79a7'   # blue   — OpenGene
C_CPU    = '#59a14f'   # green  — d0bromir CPU
C_GPU    = '#e15759'   # red    — d0bromir GPU
C_LOAD   = '#4e79a7'   # blue   — loading stage
C_FILT   = '#b07aa1'   # purple — filtering stage
C_WRITE  = '#76b7b2'   # teal   — writing stage
C_OTHER  = '#d3d3d3'   # gray   — unaccounted

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

# ──────────────────────────────────────────────────────────────
# CSV helpers
# ──────────────────────────────────────────────────────────────
def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def median_of(lst):
    s = sorted(lst)
    return s[len(s)//2] if s else float('nan')

def load_bench():
    """Returns dict: (dataset, tool, threads) -> median walltime_s"""
    raw = defaultdict(list)
    for row in load_csv(BENCH_CSV):
        key = (row['dataset'], row['tool'], int(row['threads']))
        raw[key].append(float(row['walltime_s']))
    return {k: median_of(v) for k, v in raw.items()}

# Canonical dataset order and display labels
DS_ORDER = [
    ('Panel_SE_148M',  'Panel SE\n(148 MB)'),
    ('Panel_PE_304M',  'Panel PE\n(304 MB)'),
    ('WGS_SE_6.3G',    'WGS SE\n(6.3 GB)'),
    ('WGS_PE_12.8G',   'WGS PE\n(12.8 GB)'),
    ('WGS_PE_18.2G',   'WGS PE\n(18.2 GB)'),
    ('WGS_PE_40G',     'WGS PE\n(40 GB)'),
]

# ══════════════════════════════════════════════════════════════
# FIGURE 1: Three-way wall-time comparison + stage breakdown
# ══════════════════════════════════════════════════════════════
def fig1_breakdown():
    """(a) Wall-time at T=8 for all 6 datasets, 3 tools.
       (b) Stage breakdown (loading/filtering/writing) from fig1_breakdown.csv."""
    bench = load_bench()
    breakdown = load_csv(FIG1_CSV)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={'width_ratios': [1.2, 1]})

    # ── Panel (a): 3-way wall-time at T=8 ─────────────────────
    T = 8
    ds_keys   = [k for k, _ in DS_ORDER]
    ds_labels = [l for _, l in DS_ORDER]

    og_vals  = [bench.get((k, 'opengene',     T), float('nan')) for k in ds_keys]
    cpu_vals = [bench.get((k, 'd0bromir_cpu', T), float('nan')) for k in ds_keys]
    gpu_vals = [bench.get((k, 'd0bromir_gpu', T), float('nan')) for k in ds_keys]

    x = np.arange(len(ds_keys))
    w = 0.24

    bars_og  = ax1.bar(x - w, og_vals,  w, label='fastp (OpenGene baseline)', color=C_OG,  edgecolor='white', linewidth=0.5)
    bars_cpu = ax1.bar(x,     cpu_vals, w, label='fastp-GPU (CPU mode)',       color=C_CPU, edgecolor='white', linewidth=0.5)
    bars_gpu = ax1.bar(x + w, gpu_vals, w, label='fastp-GPU (GPU mode)',       color=C_GPU, edgecolor='white', linewidth=0.5)

    # Speedup annotations (GPU vs OpenGene)
    for i in range(len(ds_keys)):
        og, gp = og_vals[i], gpu_vals[i]
        if og > 0 and gp > 0 and not (np.isnan(og) or np.isnan(gp)):
            spd = og / gp
            color = C_GPU if spd > 1.0 else 'gray'
            ax1.annotate(f'{spd:.2f}×',
                         xy=(x[i] + w, gp),
                         xytext=(0, 4), textcoords='offset points',
                         fontsize=6.5, color=color, ha='center', fontweight='bold')

    # Wall-time labels on bars
    for bars in [bars_og, bars_cpu, bars_gpu]:
        for bar in bars:
            h = bar.get_height()
            if np.isnan(h) or h == 0:
                continue
            label = f'{h:.0f}s' if h >= 10 else f'{h:.1f}s'
            ax1.text(bar.get_x() + bar.get_width()/2, h + h*0.01,
                     label, ha='center', va='bottom', fontsize=5.5)

    ax1.set_ylabel('Wall Time (s)')
    ax1.set_title('(a) End-to-end Performance at T=8', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ds_labels, fontsize=8)
    ax1.legend(fontsize=7, loc='upper left')
    ax1.grid(axis='y', alpha=0.2)

    # ── Panel (b): Stage breakdown at T=8 ─────────────────────
    # Index by (dataset, tool)
    bk = {}
    for row in breakdown:
        bk[(row['dataset'], row['tool'])] = row

    # Show both CPU and GPU side by side per dataset
    b_labels = []
    b_x = []
    b_load_cpu, b_filt_cpu, b_write_cpu, b_other_cpu = [], [], [], []
    b_load_gpu, b_filt_gpu, b_write_gpu, b_other_gpu = [], [], [], []

    xi = 0
    xticks = []
    xtick_labels = []
    cpu_positions = []
    gpu_positions = []

    for ds_key, ds_label in DS_ORDER:
        cpu_row = bk.get((ds_key, 'd0bromir_cpu'))
        gpu_row = bk.get((ds_key, 'd0bromir_gpu'))

        def pct(row, col):
            try:
                return float(row[col].rstrip('%'))
            except (TypeError, ValueError, KeyError):
                return 0.0

        if cpu_row:
            lp = pct(cpu_row, 'loading_pct')
            fp = pct(cpu_row, 'filtering_pct')
            wp = pct(cpu_row, 'writing_pct')
            op = max(0.0, 100 - lp - fp - wp)
            b_load_cpu.append(lp)
            b_filt_cpu.append(fp)
            b_write_cpu.append(wp)
            b_other_cpu.append(op)
            cpu_positions.append(xi)
            xi += 1

        if gpu_row:
            lp = pct(gpu_row, 'loading_pct')
            fp = pct(gpu_row, 'filtering_pct')
            wp = pct(gpu_row, 'writing_pct')
            op = max(0.0, 100 - lp - fp - wp)
            b_load_gpu.append(lp)
            b_filt_gpu.append(fp)
            b_write_gpu.append(wp)
            b_other_gpu.append(op)
            gpu_positions.append(xi)
            xi += 1

        # label goes between the two bars
        xticks.append(xi - 1.5 if (cpu_row and gpu_row) else xi - 1)
        xtick_labels.append(ds_label)
        xi += 0.5   # gap between datasets

    cpu_pos = np.array(cpu_positions)
    gpu_pos = np.array(gpu_positions)

    def stack_bars(ax, positions, loads, filts, writes, others, prefix):
        pos = np.array(positions)
        w = 0.8
        b = np.zeros(len(pos))
        ax.bar(pos, loads,  w, bottom=b, color=C_LOAD,  edgecolor='white', linewidth=0.3, label='Loading'   if prefix == 'CPU' else '_nolegend_')
        b += np.array(loads)
        ax.bar(pos, filts,  w, bottom=b, color=C_FILT,  edgecolor='white', linewidth=0.3, label='Filtering' if prefix == 'CPU' else '_nolegend_')
        b += np.array(filts)
        ax.bar(pos, writes, w, bottom=b, color=C_WRITE, edgecolor='white', linewidth=0.3, label='Writing'   if prefix == 'CPU' else '_nolegend_')
        b += np.array(writes)
        ax.bar(pos, others, w, bottom=b, color=C_OTHER, edgecolor='white', linewidth=0.3, label='Other'     if prefix == 'CPU' else '_nolegend_')

    stack_bars(ax2, cpu_pos, b_load_cpu, b_filt_cpu, b_write_cpu, b_other_cpu, 'CPU')
    stack_bars(ax2, gpu_pos, b_load_gpu, b_filt_gpu, b_write_gpu, b_other_gpu, 'GPU')

    # Annotate bars where loading > 100% (reader-thread spans pipeline stages)
    for pos, lp in zip(cpu_pos, b_load_cpu):
        if lp > 100:
            ax2.text(pos, lp + 0.5, f'{lp:.0f}%', ha='center', va='bottom',
                     fontsize=5.5, color=C_LOAD, fontweight='bold')
    for pos, lp in zip(gpu_pos, b_load_gpu):
        if lp > 100:
            ax2.text(pos, lp + 0.5, f'{lp:.0f}%', ha='center', va='bottom',
                     fontsize=5.5, color=C_LOAD, fontweight='bold')

    # CPU / GPU labels under bar groups
    for pos in cpu_pos:
        ax2.text(pos, -6, 'CPU', ha='center', va='top', fontsize=6, color=C_CPU)
    for pos in gpu_pos:
        ax2.text(pos, -6, 'GPU', ha='center', va='top', fontsize=6, color=C_GPU)

    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xtick_labels, fontsize=7)
    ax2.set_ylabel('Fraction of Wall Time (%)')
    ax2.set_title('(b) Pipeline Stage Breakdown at T=8', fontsize=10)
    ax2.set_ylim(-8, 122)
    ax2.legend(loc='upper right', fontsize=7)
    ax2.grid(axis='y', alpha=0.2)
    # Note about loading > 100%
    ax2.axhline(100, color='black', linewidth=0.5, linestyle='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig1_breakdown.png'))
    fig.savefig(os.path.join(OUT, 'fig1_breakdown.pdf'))
    plt.close(fig)
    print("Figure 1 saved")


# ══════════════════════════════════════════════════════════════
# FIGURE 2: Filtering-stage speedup GPU vs CPU
# ══════════════════════════════════════════════════════════════
def fig2_kernel_speedup():
    """Grouped bars: CPU vs GPU filtering time (ms) per dataset at available
       thread counts. Speedup annotation on top. Data from fig2_kernel_speedup.csv."""
    rows = load_csv(FIG2_CSV)

    # Datasets with T=8 data (for the main panel)
    t8 = {r['dataset']: r for r in rows if int(r['threads']) == 8}

    # For thread-scaling sub-panel: datasets with multi-thread data
    multi_ds = sorted({r['dataset'] for r in rows if r['dataset'] not in ('Panel_SE_148M', 'Panel_PE_304M')})
    thread_counts = sorted({int(r['threads']) for r in rows})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Panel (a): T=8 CPU vs GPU filtering ───────────────────
    ordered_ds = [k for k, _ in DS_ORDER if k in t8]
    labels_a = [l for k, l in DS_ORDER if k in t8]

    cpu_ms = [float(t8[k]['d0bromir_cpu_filtering_ms']) for k in ordered_ds]
    gpu_ms = [float(t8[k]['d0bromir_gpu_filtering_ms']) for k in ordered_ds]

    x = np.arange(len(ordered_ds))
    w = 0.35

    ax1.bar(x - w/2, cpu_ms, w, label='CPU (d0bromir)', color=C_CPU)
    ax1.bar(x + w/2, gpu_ms, w, label='GPU (d0bromir)', color=C_GPU)

    for i, (c, g) in enumerate(zip(cpu_ms, gpu_ms)):
        spd = c / g if g > 0 else 0
        color = C_GPU if spd > 1.0 else 'gray'
        ymax = max(c, g)
        ax1.text(i, ymax * 1.04, f'{spd:.1f}×', ha='center', fontsize=8,
                 fontweight='bold', color=color)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_a, fontsize=7.5)
    ax1.set_ylabel('Filtering Stage Time (ms)')
    ax1.set_title('(a) CPU vs GPU Filtering Time at T=8', fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.2)

    # ── Panel (b): speedup across thread counts for WGS datasets ─
    ds_plot = [(k, l) for k, l in DS_ORDER if k in multi_ds]
    colors_b = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']
    markers_b = ['o', 's', '^', 'D']

    for idx, (ds_key, ds_label) in enumerate(ds_plot):
        ds_rows = sorted([r for r in rows if r['dataset'] == ds_key], key=lambda r: int(r['threads']))
        ts = [int(r['threads']) for r in ds_rows]
        speedups = [float(r['d0bromir_cpu_filtering_ms']) / float(r['d0bromir_gpu_filtering_ms'])
                    for r in ds_rows if float(r['d0bromir_gpu_filtering_ms']) > 0]
        ts_valid = [int(r['threads']) for r in ds_rows if float(r['d0bromir_gpu_filtering_ms']) > 0]

        color = colors_b[idx % len(colors_b)]
        marker = markers_b[idx % len(markers_b)]
        ax2.plot(ts_valid, speedups, marker=marker, color=color,
                 label=ds_label.replace('\n', ' '), linewidth=1.5, markersize=5)

    ax2.axhline(1.0, color='black', linewidth=0.8, linestyle='--', alpha=0.5, label='No speedup (1×)')
    ax2.set_xlabel('Thread Count')
    ax2.set_ylabel('GPU/CPU Filtering Speedup (×)')
    ax2.set_title('(b) Filtering Speedup vs Thread Count', fontsize=10)
    ax2.set_xticks(thread_counts)
    ax2.legend(fontsize=7, loc='upper right')
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig2_kernel_speedup.png'))
    fig.savefig(os.path.join(OUT, 'fig2_kernel_speedup.pdf'))
    plt.close(fig)
    print("Figure 2 saved")


# ══════════════════════════════════════════════════════════════
# FIGURE 3: Speculative statistics — CPU speedup over OpenGene
# ══════════════════════════════════════════════════════════════
def fig3_speculative_stats():
    """(a) Wall-time: OpenGene vs d0bromir CPU across thread counts.
       (b) CPU speedup % vs thread count for all 6 datasets."""
    rows = load_csv(FIG3_CSV)

    thread_counts = sorted({int(r['threads']) for r in rows})
    ds_list = [(k, l) for k, l in DS_ORDER if k in {r['dataset'] for r in rows}]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    colors = ['#4e79a7', '#e15759', '#59a14f', '#f28e2b', '#b07aa1', '#76b7b2']
    markers = ['o', 's', '^', 'D', 'v', 'P']

    # ── Panel (a): wall times for T=8 ─────────────────────────
    t8_rows = [r for r in rows if int(r['threads']) == 8]
    t8_by_ds = {r['dataset']: r for r in t8_rows}

    ordered_ds = [k for k, _ in ds_list if k in t8_by_ds]
    labels_a = [l for k, l in ds_list if k in t8_by_ds]

    og_vals  = [float(t8_by_ds[k]['opengene_s'])     for k in ordered_ds]
    cpu_vals = [float(t8_by_ds[k]['d0bromir_cpu_s'])  for k in ordered_ds]

    x = np.arange(len(ordered_ds))
    w = 0.35
    ax1.bar(x - w/2, og_vals,  w, label='OpenGene', color=C_OG)
    ax1.bar(x + w/2, cpu_vals, w, label='d0bromir CPU', color=C_CPU)

    for i, (og, cp) in enumerate(zip(og_vals, cpu_vals)):
        spd = (og - cp) / og * 100
        color = C_CPU if spd > 0 else C_GPU
        sign = '+' if spd > 0 else ''
        ax1.text(i, max(og, cp) * 1.03, f'{sign}{spd:.1f}%',
                 ha='center', fontsize=7.5, fontweight='bold', color=color)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_a, fontsize=7.5)
    ax1.set_ylabel('Wall Time (s)')
    ax1.set_title('(a) OpenGene vs d0bromir CPU at T=8', fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.2)

    # ── Panel (b): speedup % vs thread count ──────────────────
    for idx, (ds_key, ds_label) in enumerate(ds_list):
        ds_rows = sorted([r for r in rows if r['dataset'] == ds_key], key=lambda r: int(r['threads']))
        ts = [int(r['threads']) for r in ds_rows]
        spds = [float(r['cpu_speedup_pct'].rstrip('%')) for r in ds_rows]

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax2.plot(ts, spds, marker=marker, color=color,
                 label=ds_label.replace('\n', ' '), linewidth=1.5, markersize=5)

    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax2.set_xlabel('Thread Count')
    ax2.set_ylabel('d0bromir CPU Speedup over OpenGene (%)')
    ax2.set_title('(b) CPU Speedup vs Thread Count', fontsize=10)
    ax2.set_xticks(thread_counts)
    ax2.legend(fontsize=7, loc='lower right')
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig3_speculative_stats.png'))
    fig.savefig(os.path.join(OUT, 'fig3_speculative_stats.pdf'))
    plt.close(fig)
    print("Figure 3 saved")


# ══════════════════════════════════════════════════════════════
# FIGURE 4: GPU transfer overhead
# ══════════════════════════════════════════════════════════════
def fig4_transfer_overhead():
    """(a) GPU overhead (gpu - cpu wall) at each thread count.
       (b) GPU overhead as % of CPU wall time."""
    rows = load_csv(FIG4_CSV)

    thread_counts = sorted({int(r['threads']) for r in rows})
    ds_list = [(k, l) for k, l in DS_ORDER if k in {r['dataset'] for r in rows}]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    colors = ['#4e79a7', '#e15759', '#59a14f', '#f28e2b', '#b07aa1', '#76b7b2']
    markers = ['o', 's', '^', 'D', 'v', 'P']

    for idx, (ds_key, ds_label) in enumerate(ds_list):
        ds_rows = sorted([r for r in rows if r['dataset'] == ds_key], key=lambda r: int(r['threads']))
        ts = [int(r['threads']) for r in ds_rows]
        overhead_s = [float(r['gpu_overhead_vs_cpu_s']) for r in ds_rows]
        overhead_pct = [float(r['gpu_overhead_vs_cpu_pct'].rstrip('%')) for r in ds_rows]

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        label = ds_label.replace('\n', ' ')

        ax1.plot(ts, overhead_s, marker=marker, color=color,
                 label=label, linewidth=1.5, markersize=5)
        ax2.plot(ts, overhead_pct, marker=marker, color=color,
                 label=label, linewidth=1.5, markersize=5)

    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax1.set_xlabel('Thread Count')
    ax1.set_ylabel('GPU Overhead vs CPU (s)')
    ax1.set_title('(a) Absolute GPU Overhead', fontsize=10)
    ax1.set_xticks(thread_counts)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.2)

    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax2.set_xlabel('Thread Count')
    ax2.set_ylabel('GPU Overhead vs CPU (%)')
    ax2.set_title('(b) Relative GPU Overhead', fontsize=10)
    ax2.set_xticks(thread_counts)
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig4_transfer_overhead.png'))
    fig.savefig(os.path.join(OUT, 'fig4_transfer_overhead.pdf'))
    plt.close(fig)
    print("Figure 4 saved")


# ══════════════════════════════════════════════════════════════
# FIGURE 5: Thread Scaling
# ══════════════════════════════════════════════════════════════
def fig5_thread_scaling():
    """Wall time vs thread count for all 6 datasets, 3 tools."""
    bench = load_bench()

    tools = [
        ('opengene',     'fastp (OpenGene)',  C_OG,  'o', '-'),
        ('d0bromir_cpu', 'fastp-GPU (CPU)',   C_CPU, '^', '--'),
        ('d0bromir_gpu', 'fastp-GPU (GPU)',   C_GPU, 's', '-'),
    ]
    thread_counts = sorted({t for (_, _, t) in bench.keys()})

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()

    for idx, (ds_key, ds_label) in enumerate(DS_ORDER):
        ax = axes[idx]
        for tool_key, tool_label, color, marker, ls in tools:
            times = [bench.get((ds_key, tool_key, t), float('nan')) for t in thread_counts]
            ax.plot(thread_counts, times, marker=marker, linestyle=ls,
                    color=color, label=tool_label, markersize=4, linewidth=1.5)

        ax.set_xlabel('Threads')
        ax.set_ylabel('Wall Time (s)')
        ax.set_title(ds_label.replace('\n', ' '), fontsize=9)
        ax.set_xticks(thread_counts)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.3)

        # Annotate best GPU vs OpenGene speedup
        best_spd, best_t = 0.0, 0
        for t in thread_counts:
            og = bench.get((ds_key, 'opengene', t), float('nan'))
            gp = bench.get((ds_key, 'd0bromir_gpu', t), float('nan'))
            if og > 0 and gp > 0 and not (np.isnan(og) or np.isnan(gp)):
                spd = og / gp
                if spd > best_spd:
                    best_spd, best_t = spd, t
        if best_spd > 1.05:
            gp_time = bench.get((ds_key, 'd0bromir_gpu', best_t), float('nan'))
            ax.annotate(f'{best_spd:.2f}×',
                        xy=(best_t, gp_time),
                        xytext=(5, -15), textcoords='offset points',
                        fontsize=7, color=C_GPU, fontweight='bold')

    fig.suptitle('Thread Scaling: 3-way Comparison', fontsize=11, fontweight='bold')
    for i in range(len(DS_ORDER), len(axes)):
        axes[i].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, 'fig5_thread_scaling.png'))
    fig.savefig(os.path.join(OUT, 'fig5_thread_scaling.pdf'))
    plt.close(fig)
    print("Figure 5 saved")


if __name__ == '__main__':
    fig1_breakdown()
    fig2_kernel_speedup()
    fig3_speculative_stats()
    fig4_transfer_overhead()
    fig5_thread_scaling()
    print(f"\nAll figures saved to {OUT}")
