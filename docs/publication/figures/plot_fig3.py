#!/usr/bin/env python3
"""
Figure 3: Statistics with Reversible Updates — d0bromir CPU speedup over OpenGene.
(a) Wall-time grouped bars at T=8 (OpenGene vs d0bromir CPU).
(b) CPU speedup (%) vs thread count for all 6 datasets.
Reads: fig3_speculative_stats.csv
Output: fig3_speculative_stats.png / fig3_speculative_stats.pdf
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

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

COLORS  = ['#4e79a7', '#e15759', '#59a14f', '#f28e2b', '#b07aa1', '#76b7b2']
MARKERS = ['o', 's', '^', 'D', 'v', 'P']

def load_csv(name):
    with open(os.path.join(HERE, name)) as f:
        return list(csv.DictReader(f))

def main():
    rows = load_csv('fig3_speculative_stats.csv')
    thread_counts = sorted({int(r['threads']) for r in rows})
    present_ds    = {r['dataset'] for r in rows}
    ds_list       = [k for k in DS_ORDER if k in present_ds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Panel (a): wall-time at T=8 ───────────────────────────
    t8 = {r['dataset']: r for r in rows if int(r['threads']) == 8}
    ordered = [k for k in ds_list if k in t8]
    labels_a = [DS_LABELS.get(k, k) for k in ordered]

    og_vals  = [float(t8[k]['opengene_s'])    for k in ordered]
    cpu_vals = [float(t8[k]['d0bromir_cpu_s']) for k in ordered]

    x = np.arange(len(ordered))
    w = 0.35
    ax1.bar(x - w/2, og_vals,  w, label='OpenGene',     color=C_OG)
    ax1.bar(x + w/2, cpu_vals, w, label='d0bromir CPU', color=C_CPU)

    for i, (og, cp) in enumerate(zip(og_vals, cpu_vals)):
        spd  = (og - cp) / og * 100
        sign = '+' if spd > 0 else ''
        col  = C_CPU if spd > 0 else C_GPU
        ax1.annotate(f'{sign}{spd:.1f}%', xy=(i, max(og, cp)),
                     xytext=(0, 6), textcoords='offset points',
                     ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=col)

    _top = max(max(og_vals), max(cpu_vals))
    ax1.set_ylim(0, _top * 1.22)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_a, fontsize=7.5)
    ax1.set_ylabel('Wall Time (s)')
    ax1.set_title('(a) OpenGene vs d0bromir CPU at T=8', fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.2)

    # ── Panel (b): speedup % vs thread count ──────────────────
    for idx, ds_key in enumerate(ds_list):
        ds_rows = sorted([r for r in rows if r['dataset'] == ds_key],
                         key=lambda r: int(r['threads']))
        ts   = [int(r['threads']) for r in ds_rows]
        spds = [float(r['cpu_speedup_pct'].rstrip('%')) for r in ds_rows]
        ax2.plot(ts, spds,
                 marker=MARKERS[idx % len(MARKERS)],
                 color=COLORS[idx % len(COLORS)],
                 label=DS_LABELS.get(ds_key, ds_key).replace('\n', ' '),
                 linewidth=1.5, markersize=5)

    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax2.set_xlabel('Thread Count')
    ax2.set_ylabel('d0bromir CPU Speedup over OpenGene (%)')
    ax2.set_title('(b) CPU Speedup vs Thread Count', fontsize=10)
    ax2.set_xticks(thread_counts)
    ax2.legend(fontsize=7, loc='lower right')
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    png_path = os.path.join(HERE, 'fig3_speculative_stats.png')
    fig.savefig(png_path, facecolor='white', transparent=False)
    fig.savefig(os.path.join(HERE, 'fig3_speculative_stats.pdf'), facecolor='white', transparent=False)
    plt.close(fig)
    from PIL import Image as _Img
    _Img.open(png_path).convert('RGB').save(png_path, optimize=True)
    print("fig3_speculative_stats.png / .pdf saved")

if __name__ == '__main__':
    main()
