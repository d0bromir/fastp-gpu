#!/usr/bin/env python3
"""
Figure 5: Thread scaling — wall time vs thread count, all 6 datasets, 3 tools.
Reads: fig5_thread_scaling.csv
Output: fig5_thread_scaling.png / fig5_thread_scaling.pdf
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

C_OG  = '#4e79a7'
C_CPU = '#59a14f'
C_GPU = '#e15759'

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 9, 'legend.fontsize': 8,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

DS_ORDER = [
    ('Panel_SE_148M', 'Panel SE (148 MB)'),
    ('Panel_PE_304M', 'Panel PE (304 MB)'),
    ('WGS_SE_6.3G',   'WGS SE (6.3 GB)'),
    ('WGS_PE_12.8G',  'WGS PE (12.8 GB)'),
    ('WGS_PE_18.2G',  'WGS PE (18.2 GB)'),
    ('WGS_PE_40G',    'WGS PE (40 GB)'),
]

TOOLS = [
    ('opengene',     'fastp (OpenGene)',  C_OG,  'o', '-',  1, 1.5),
    ('d0bromir_gpu', 'fastp-GPU (GPU)',   C_GPU, 's', '-',  2, 1.5),
    ('d0bromir_cpu', 'fastp-GPU (CPU)',   C_CPU, '^', '--', 3, 2.2),
]

def load_csv(name):
    with open(os.path.join(HERE, name)) as f:
        return list(csv.DictReader(f))

def median_of(lst):
    s = sorted(lst)
    return s[len(s)//2] if s else float('nan')

def main():
    raw = defaultdict(list)
    for row in load_csv('fig5_thread_scaling.csv'):
        key = (row['dataset'], row['tool'], int(row['threads']))
        raw[key].append(float(row['walltime_s']))
    bench = {k: median_of(v) for k, v in raw.items()}

    thread_counts = sorted({t for (_, _, t) in bench.keys()})

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()

    for idx, (ds_key, ds_label) in enumerate(DS_ORDER):
        ax = axes[idx]
        for tool_key, tool_label, color, marker, ls, zord, lw in TOOLS:
            times = [bench.get((ds_key, tool_key, t), float('nan'))
                     for t in thread_counts]
            ax.plot(thread_counts, times, marker=marker, linestyle=ls,
                    color=color, label=tool_label, markersize=4,
                    linewidth=lw, zorder=zord)

        ax.set_xlabel('Threads')
        ax.set_ylabel('Wall Time (s)')
        ax.set_title(ds_label, fontsize=9)
        ax.set_xticks(thread_counts)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.3)

        # Best GPU vs OpenGene speedup annotation
        best_spd, best_t = 0.0, 0
        for t in thread_counts:
            og = bench.get((ds_key, 'opengene', t), float('nan'))
            gp = bench.get((ds_key, 'd0bromir_gpu', t), float('nan'))
            if og > 0 and gp > 0 and not (np.isnan(og) or np.isnan(gp)):
                spd = og / gp
                if spd > best_spd:
                    best_spd, best_t = spd, t
        if best_spd > 1.05:
            gp_t = bench.get((ds_key, 'd0bromir_gpu', best_t), float('nan'))
            ax.annotate(f'{best_spd:.2f}×', xy=(best_t, gp_t),
                        xytext=(5, -15), textcoords='offset points',
                        fontsize=7, color=C_GPU, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))

    fig.suptitle('Thread Scaling: 3-way Comparison', fontsize=11, fontweight='bold')
    for i in range(len(DS_ORDER), len(axes)):
        axes[i].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    png_path = os.path.join(HERE, 'fig5_thread_scaling.png')
    fig.savefig(png_path, facecolor='white', transparent=False)
    fig.savefig(os.path.join(HERE, 'fig5_thread_scaling.pdf'), facecolor='white', transparent=False)
    plt.close(fig)
    from PIL import Image as _Img
    _Img.open(png_path).convert('RGB').save(png_path, optimize=True)
    print("fig5_thread_scaling.png / .pdf saved")

if __name__ == '__main__':
    main()
