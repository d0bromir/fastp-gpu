#!/usr/bin/env python3
"""
Figure 4: GPU transfer overhead vs CPU mode.
(a) Absolute GPU overhead in seconds vs thread count.
(b) Relative GPU overhead (%) vs thread count.
Reads: fig4_transfer_overhead.csv
Output: fig4_transfer_overhead.png / fig4_transfer_overhead.pdf
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 9, 'legend.fontsize': 8,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

DS_ORDER = [
    'Panel_SE_148M', 'Panel_PE_304M',
    'WGS_SE_6.3G', 'WGS_PE_12.8G', 'WGS_PE_18.2G', 'WGS_PE_40G',
]
DS_LABELS = {
    'Panel_SE_148M': 'Panel SE (148 MB)',
    'Panel_PE_304M': 'Panel PE (304 MB)',
    'WGS_SE_6.3G':   'WGS SE (6.3 GB)',
    'WGS_PE_12.8G':  'WGS PE (12.8 GB)',
    'WGS_PE_18.2G':  'WGS PE (18.2 GB)',
    'WGS_PE_40G':    'WGS PE (40 GB)',
}

COLORS  = ['#4e79a7', '#e15759', '#59a14f', '#f28e2b', '#b07aa1', '#76b7b2']
MARKERS = ['o', 's', '^', 'D', 'v', 'P']

def load_csv(name):
    with open(os.path.join(HERE, name)) as f:
        return list(csv.DictReader(f))

def main():
    rows = load_csv('fig4_transfer_overhead.csv')
    thread_counts = sorted({int(r['threads']) for r in rows})
    present_ds    = {r['dataset'] for r in rows}
    ds_list       = [k for k in DS_ORDER if k in present_ds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for idx, ds_key in enumerate(ds_list):
        ds_rows = sorted([r for r in rows if r['dataset'] == ds_key],
                         key=lambda r: int(r['threads']))
        ts          = [int(r['threads']) for r in ds_rows]
        overhead_s  = [float(r['gpu_overhead_vs_cpu_s']) for r in ds_rows]
        overhead_pct = [float(r['gpu_overhead_vs_cpu_pct'].rstrip('%')) for r in ds_rows]

        color  = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        label  = DS_LABELS.get(ds_key, ds_key)

        ax1.plot(ts, overhead_s,   marker=marker, color=color,
                 label=label, linewidth=1.5, markersize=5)
        ax2.plot(ts, overhead_pct, marker=marker, color=color,
                 label=label, linewidth=1.5, markersize=5)

    for ax, ylabel, title in [
        (ax1, 'GPU Overhead vs CPU Mode (s)',  '(a) Absolute GPU Overhead'),
        (ax2, 'GPU Overhead vs CPU Mode (%)',  '(b) Relative GPU Overhead'),
    ]:
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_xlabel('Thread Count')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(thread_counts)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    png_path = os.path.join(HERE, 'fig4_transfer_overhead.png')
    fig.savefig(png_path, facecolor='white', transparent=False)
    fig.savefig(os.path.join(HERE, 'fig4_transfer_overhead.pdf'), facecolor='white', transparent=False)
    plt.close(fig)
    from PIL import Image as _Img
    _Img.open(png_path).convert('RGB').save(png_path, optimize=True)
    print("fig4_transfer_overhead.png / .pdf saved")

if __name__ == '__main__':
    main()
