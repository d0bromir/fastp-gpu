#!/usr/bin/env python3
"""
Figure 2: Filtering-stage CPU vs GPU speedup.
(a) Grouped bars at T=8 for all datasets with profiling data.
(b) Speedup (×) vs thread count for WGS datasets.
Reads: fig2_kernel_speedup.csv
Output: fig2_kernel_speedup.png / fig2_kernel_speedup.pdf
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))

C_CPU = '#59a14f'
C_GPU = '#e15759'

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
    'Panel_SE_148M': 'Panel SE\n(148 MB)',
    'Panel_PE_304M': 'Panel PE\n(304 MB)',
    'WGS_SE_6.3G':   'WGS SE\n(6.3 GB)',
    'WGS_PE_12.8G':  'WGS PE\n(12.8 GB)',
    'WGS_PE_18.2G':  'WGS PE\n(18.2 GB)',
    'WGS_PE_40G':    'WGS PE\n(40 GB)',
}

COLORS_LINE = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#b07aa1', '#59a14f']
MARKERS     = ['o', 's', '^', 'D', 'v', 'P']

def load_csv(name):
    with open(os.path.join(HERE, name)) as f:
        return list(csv.DictReader(f))

def main():
    rows = load_csv('fig2_kernel_speedup.csv')

    # Datasets that have multi-thread data (WGS)
    multi_ds = sorted(
        {r['dataset'] for r in rows
         if r['dataset'] not in ('Panel_SE_148M', 'Panel_PE_304M')},
        key=lambda k: DS_ORDER.index(k) if k in DS_ORDER else 99,
    )

    t8_by_ds = {r['dataset']: r for r in rows if int(r['threads']) == 8}
    thread_counts = sorted({int(r['threads']) for r in rows})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Panel (a): CPU vs GPU filtering time at T=8 ───────────
    ordered = [k for k in DS_ORDER if k in t8_by_ds]
    labels_a = [DS_LABELS.get(k, k) for k in ordered]
    cpu_ms = [float(t8_by_ds[k]['d0bromir_cpu_filtering_ms']) for k in ordered]
    gpu_ms = [float(t8_by_ds[k]['d0bromir_gpu_filtering_ms']) for k in ordered]

    x = np.arange(len(ordered))
    w = 0.35
    ax1.bar(x - w/2, cpu_ms, w, label='CPU filter (d0bromir)', color=C_CPU)
    ax1.bar(x + w/2, gpu_ms, w, label='GPU filter (d0bromir)', color=C_GPU)

    for i, (c, g) in enumerate(zip(cpu_ms, gpu_ms)):
        spd = c / g if g > 0 else 0.0
        ax1.annotate(f'{spd:.1f}\u00d7', xy=(i, max(c, g)),
                     xytext=(0, 6), textcoords='offset points',
                     ha='center', va='bottom', fontsize=8, fontweight='bold', color='#222222',
                     bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))

    _top = max(max(cpu_ms), max(gpu_ms))
    ax1.set_ylim(0, _top * 1.22)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_a, fontsize=7.5)
    ax1.set_ylabel('Filtering Stage Time (ms)')
    ax1.set_title('(a) CPU vs GPU Filtering Time at T=8', fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.2)

    # ── Panel (b): speedup × vs thread count (WGS datasets) ───
    for idx, ds_key in enumerate(k for k in DS_ORDER if k in multi_ds):
        ds_rows = sorted(
            [r for r in rows if r['dataset'] == ds_key],
            key=lambda r: int(r['threads'])
        )
        ts_valid  = [int(r['threads'])  for r in ds_rows if float(r['d0bromir_gpu_filtering_ms']) > 0]
        speedups  = [float(r['d0bromir_cpu_filtering_ms']) / float(r['d0bromir_gpu_filtering_ms'])
                     for r in ds_rows if float(r['d0bromir_gpu_filtering_ms']) > 0]
        color  = COLORS_LINE[idx % len(COLORS_LINE)]
        marker = MARKERS[idx % len(MARKERS)]
        ax2.plot(ts_valid, speedups, marker=marker, color=color,
                 label=DS_LABELS.get(ds_key, ds_key).replace('\n', ' '),
                 linewidth=1.5, markersize=5)

    ax2.axhline(1.0, color='black', linewidth=0.8, linestyle='--', alpha=0.5,
                label='No speedup (1\u00d7)')
    ax2.set_xlabel('Thread Count')
    ax2.set_ylabel('GPU/CPU Filtering Speedup (\u00d7)')
    ax2.set_title('(b) Filtering Speedup vs Thread Count', fontsize=10)
    ax2.set_xticks(thread_counts)
    ax2.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.0, 0.98))
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    png_path = os.path.join(HERE, 'fig2_kernel_speedup.png')
    fig.savefig(png_path, facecolor='white', transparent=False)
    fig.savefig(os.path.join(HERE, 'fig2_kernel_speedup.pdf'), facecolor='white', transparent=False)
    plt.close(fig)
    from PIL import Image as _Img
    _Img.open(png_path).convert('RGB').save(png_path, optimize=True)
    print("fig2_kernel_speedup.png / .pdf saved")

if __name__ == '__main__':
    main()
