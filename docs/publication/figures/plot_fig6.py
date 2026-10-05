#!/usr/bin/env python3
"""
Figure 6: Thread scaling on x86 server (Intel Xeon Gold 5218, a2 node).
CPU-only platform (no GPU); GPU-mode binary runs CPU fallback path.

Reads:  fig6_a2_cpu_scaling.csv
Output: fig6_a2_cpu_scaling.png / .pdf
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
    ('WGS_PE_18.2G',      'WGS PE (18.2 GB)'),
    ('WGS_PE_40G',        'WGS PE (40 GB)'),
    ('BS_HiSeq_SE_26.8G', 'BS-Seq SE (26.8 GB)'),
]

TOOLS = [
    ('opengene',     'fastp (OpenGene)',     C_OG,  'o', '-',  1, 1.5),
    ('d0bromir_cpu', 'fastp-GPU (CPU mode)', C_CPU, '^', '--', 3, 2.2),
    ('d0bromir_gpu', 'fastp-GPU (GPU mode)', C_GPU, 's', '-',  2, 1.5),
]


def load_csv(name):
    with open(os.path.join(HERE, name)) as f:
        return list(csv.DictReader(f))


def median_of(lst):
    s = sorted(lst)
    return s[len(s) // 2] if s else float('nan')


def main():
    raw = defaultdict(list)
    for row in load_csv('fig6_a2_cpu_scaling.csv'):
        key = (row['dataset'], row['tool'], int(row['threads']))
        raw[key].append(float(row['walltime_s']))
    bench = {k: median_of(v) for k, v in raw.items()}

    all_threads = sorted({t for (_, _, t) in bench.keys()})

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for idx, (ds_key, ds_label) in enumerate(DS_ORDER):
        ax = axes[idx]
        ds_threads = sorted({t for (ds, _, t) in bench.keys() if ds == ds_key})

        for tool_key, tool_label, color, marker, ls, zord, lw in TOOLS:
            times = [bench.get((ds_key, tool_key, t), float('nan'))
                     for t in ds_threads]
            ax.plot(ds_threads, times, marker=marker, linestyle=ls,
                    color=color, label=tool_label, markersize=5,
                    linewidth=lw, zorder=zord)

        ax.set_xlabel('Threads')
        ax.set_ylabel('Wall Time (s)')
        ax.set_title(ds_label, fontsize=10)
        ax.set_xticks(ds_threads)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

        # Annotate peak CPU speedup.
        best_spd, best_t = 0.0, 0
        for t in ds_threads:
            og  = bench.get((ds_key, 'opengene',     t), float('nan'))
            cpu = bench.get((ds_key, 'd0bromir_cpu', t), float('nan'))
            if og > 0 and cpu > 0 and not (np.isnan(og) or np.isnan(cpu)):
                spd = og / cpu
                if spd > best_spd:
                    best_spd, best_t = spd, t
        if best_spd > 1.05:
            cpu_t = bench.get((ds_key, 'd0bromir_cpu', best_t), float('nan'))
            ax.annotate(f'{best_spd:.2f}×', xy=(best_t, cpu_t),
                        xytext=(5, -15), textcoords='offset points',
                        fontsize=8, color=C_CPU, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                  ec='none', alpha=0.85))

    fig.suptitle(
        'Cross-Platform Thread Scaling: x86 Server\n'
        '(Intel Xeon Gold 5218, 4-socket, 64 cores, CPU-only)',
        fontsize=10, fontweight='bold'
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    png_path = os.path.join(HERE, 'fig6_a2_cpu_scaling.png')
    fig.savefig(png_path, facecolor='white', transparent=False)
    fig.savefig(os.path.join(HERE, 'fig6_a2_cpu_scaling.pdf'),
                facecolor='white', transparent=False)
    plt.close(fig)
    try:
        from PIL import Image as _Img
        _Img.open(png_path).convert('RGB').save(png_path, optimize=True)
    except ImportError:
        pass
    print("fig6_a2_cpu_scaling.png / .pdf saved")


if __name__ == '__main__':
    main()
