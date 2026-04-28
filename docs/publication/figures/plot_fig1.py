#!/usr/bin/env python3
"""
Figure 1: End-to-end wall-time comparison (T=8) + pipeline stage breakdown.
Reads: fig1_breakdown.csv, fig5_thread_scaling.csv (for wall times at T=8)
Output: fig1_breakdown.png / fig1_breakdown.pdf
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))

C_OG    = '#4e79a7'   # blue   — OpenGene
C_CPU   = '#59a14f'   # green  — d0bromir CPU
C_GPU   = '#e15759'   # red    — d0bromir GPU
C_LOAD  = '#4e79a7'   # blue   — loading
C_FILT  = '#b07aa1'   # purple — filtering
C_WRITE = '#76b7b2'   # teal   — writing
C_OTHER = '#d3d3d3'   # gray   — unaccounted

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 9, 'legend.fontsize': 8,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

DS_ORDER = [
    ('Panel_SE_148M', 'Panel SE\n(148 MB)'),
    ('Panel_PE_304M', 'Panel PE\n(304 MB)'),
    ('WGS_SE_6.3G',   'WGS SE\n(6.3 GB)'),
    ('WGS_PE_12.8G',  'WGS PE\n(12.8 GB)'),
    ('WGS_PE_18.2G',  'WGS PE\n(18.2 GB)'),
    ('WGS_PE_40G',    'WGS PE\n(40 GB)'),
]

def load_csv(name):
    with open(os.path.join(HERE, name)) as f:
        return list(csv.DictReader(f))

def main():
    # ── wall times at T=8 from fig5_thread_scaling.csv ────────
    t8 = {}
    for row in load_csv('fig5_thread_scaling.csv'):
        if int(row['threads']) == 8:
            t8[(row['dataset'], row['tool'])] = float(row['walltime_s'])

    # ── stage breakdown from fig1_breakdown.csv ───────────────
    bk = {}
    for row in load_csv('fig1_breakdown.csv'):
        bk[(row['dataset'], row['tool'])] = row

    def pct(row, col):
        try:
            return float(row[col].rstrip('%'))
        except (TypeError, ValueError, KeyError):
            return 0.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={'width_ratios': [1.2, 1]})

    # ── Panel (a): 3-way wall-time at T=8 ─────────────────────
    ds_keys   = [k for k, _ in DS_ORDER]
    ds_labels = [l for _, l in DS_ORDER]

    og_vals  = [t8.get((k, 'opengene'),     float('nan')) for k in ds_keys]
    cpu_vals = [t8.get((k, 'd0bromir_cpu'), float('nan')) for k in ds_keys]
    gpu_vals = [t8.get((k, 'd0bromir_gpu'), float('nan')) for k in ds_keys]

    x = np.arange(len(ds_keys))
    w = 0.24
    bars_og  = ax1.bar(x - w, og_vals,  w, label='fastp (OpenGene baseline)', color=C_OG,  edgecolor='white', linewidth=0.5)
    bars_cpu = ax1.bar(x,     cpu_vals, w, label='fastp-GPU (CPU mode)',       color=C_CPU, edgecolor='white', linewidth=0.5)
    bars_gpu = ax1.bar(x + w, gpu_vals, w, label='fastp-GPU (GPU mode)',       color=C_GPU, edgecolor='white', linewidth=0.5)

    # ── black height labels first (needed for offset calculation) ──
    for bars in [bars_og, bars_cpu, bars_gpu]:
        for bar in bars:
            h = bar.get_height()
            if np.isnan(h) or h == 0:
                continue
            lbl = f'{h:.0f}s' if h >= 10 else f'{h:.1f}s'
            ax1.text(bar.get_x() + bar.get_width()/2, h + h * 0.03,
                     lbl, ha='center', va='bottom', fontsize=5.5)

    # ── red speedup labels above the GPU bar height label ────────
    for i in range(len(ds_keys)):
        og, gp = og_vals[i], gpu_vals[i]
        if og > 0 and gp > 0 and not (np.isnan(og) or np.isnan(gp)):
            spd = og / gp
            color = C_GPU if spd > 1.0 else 'gray'
            # place above the GPU bar's height label (which is at gp + gp*0.03)
            ax1.annotate(f'{spd:.2f}×', xy=(x[i] + w, gp + gp * 0.03),
                         xytext=(0, 9), textcoords='offset points',
                         fontsize=6.5, color=color, ha='center', fontweight='bold')

    # give enough headroom for bar labels + speedup annotations
    _top = max(v for v in og_vals + cpu_vals + gpu_vals if not np.isnan(v))
    ax1.set_ylim(0, _top * 1.22)
    ax1.set_ylabel('Wall Time (s)')
    ax1.set_title('(a) End-to-end Performance at T=8', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ds_labels, fontsize=8)
    ax1.legend(fontsize=7, loc='upper left')
    ax1.grid(axis='y', alpha=0.2)

    # ── Panel (b): stage breakdown ────────────────────────────
    xi = 0
    xticks, xtick_labels = [], []
    cpu_pos, gpu_pos = [], []
    b_load_cpu, b_filt_cpu, b_write_cpu, b_other_cpu = [], [], [], []
    b_load_gpu, b_filt_gpu, b_write_gpu, b_other_gpu = [], [], [], []

    for ds_key, ds_label in DS_ORDER:
        cpu_row = bk.get((ds_key, 'd0bromir_cpu'))
        gpu_row = bk.get((ds_key, 'd0bromir_gpu'))

        if cpu_row:
            lp = pct(cpu_row, 'loading_pct')
            fp = pct(cpu_row, 'filtering_pct')
            wp = pct(cpu_row, 'writing_pct')
            s = lp + fp + wp
            if s > 100:  # reader thread measured across parallel stages; normalize
                lp, fp, wp = lp * 100 / s, fp * 100 / s, wp * 100 / s
            b_load_cpu.append(lp); b_filt_cpu.append(fp)
            b_write_cpu.append(wp); b_other_cpu.append(max(0.0, 100 - lp - fp - wp))
            cpu_pos.append(xi); xi += 1
        if gpu_row:
            lp = pct(gpu_row, 'loading_pct')
            fp = pct(gpu_row, 'filtering_pct')
            wp = pct(gpu_row, 'writing_pct')
            s = lp + fp + wp
            if s > 100:  # reader thread measured across parallel stages; normalize
                lp, fp, wp = lp * 100 / s, fp * 100 / s, wp * 100 / s
            b_load_gpu.append(lp); b_filt_gpu.append(fp)
            b_write_gpu.append(wp); b_other_gpu.append(max(0.0, 100 - lp - fp - wp))
            gpu_pos.append(xi); xi += 1

        mid = xi - 1.5 if (cpu_row and gpu_row) else xi - 1
        xticks.append(mid); xtick_labels.append(ds_label)
        xi += 0.5

    cpu_pos = np.array(cpu_pos)
    gpu_pos = np.array(gpu_pos)
    bw = 0.8

    def stacked(ax, pos, loads, filts, writes, others, legend):
        b = np.zeros(len(pos))
        ax.bar(pos, loads,  bw, bottom=b, color=C_LOAD,  edgecolor='white', linewidth=0.3,
               label='Loading'   if legend else '_nolegend_'); b += np.array(loads)
        ax.bar(pos, filts,  bw, bottom=b, color=C_FILT,  edgecolor='white', linewidth=0.3,
               label='Filtering' if legend else '_nolegend_'); b += np.array(filts)
        ax.bar(pos, writes, bw, bottom=b, color=C_WRITE, edgecolor='white', linewidth=0.3,
               label='Writing'   if legend else '_nolegend_'); b += np.array(writes)
        ax.bar(pos, others, bw, bottom=b, color=C_OTHER, edgecolor='white', linewidth=0.3,
               label='Encoding + overhead' if legend else '_nolegend_')

    stacked(ax2, cpu_pos, b_load_cpu, b_filt_cpu, b_write_cpu, b_other_cpu, legend=True)
    stacked(ax2, gpu_pos, b_load_gpu, b_filt_gpu, b_write_gpu, b_other_gpu, legend=False)

    for pos in cpu_pos:
        ax2.text(pos, -6, 'CPU', ha='center', va='top', fontsize=6, color=C_CPU)
    for pos in gpu_pos:
        ax2.text(pos, -6, 'GPU', ha='center', va='top', fontsize=6, color=C_GPU)

    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xtick_labels, fontsize=7)
    ax2.set_ylabel('Fraction of Wall Time (%)')
    ax2.set_title('(b) Pipeline Stage Breakdown at T=8', fontsize=10)
    ax2.set_ylim(-8, 122)
    ax2.axhline(100, color='black', linewidth=0.5, linestyle='--', alpha=0.4)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=7)
    ax2.grid(axis='y', alpha=0.2)

    fig.tight_layout()
    png_path = os.path.join(HERE, 'fig1_breakdown.png')
    fig.savefig(png_path, facecolor='white', transparent=False)
    fig.savefig(os.path.join(HERE, 'fig1_breakdown.pdf'), facecolor='white', transparent=False)
    plt.close(fig)
    from PIL import Image as _Img
    _Img.open(png_path).convert('RGB').save(png_path, optimize=True)
    print("fig1_breakdown.png / .pdf saved")

if __name__ == '__main__':
    main()
