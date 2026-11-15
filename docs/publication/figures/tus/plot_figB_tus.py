#!/usr/bin/env python3
"""
Figure B (TUS paper): Speedup ratio of fastp-gpu over upstream fastp.

speedup = (upstream fastp wall-time) / (fastp-gpu CPU-mode wall-time)
at each thread count for the four WGS benchmark datasets.

Key take-aways the figure must convey:
  - At T=1, the async-decompressor thread overhead can reduce speedup below 1.
  - Speedup grows with thread count as the parallel output pool helps.
  - On WGS PE 38.5 GB the speedup reaches ~3× at T=32.

Reads:  figB_tus_speedup.csv  (same directory)
Output: figB_tus_speedup.png / .pdf

Style: grayscale, different line/marker styles per dataset, single-column width.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_IN  = os.path.join(HERE, 'figB_tus_speedup.csv')
PNG_OUT = os.path.join(HERE, 'figB_tus_speedup.png')
PDF_OUT = os.path.join(HERE, 'figB_tus_speedup.pdf')

# --- style ---------------------------------------------------------------
plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size':         9,
    'axes.titlesize':    9,
    'axes.labelsize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   7,
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
})

# Distinct grayscale line + marker styles
DS_STYLES = [
    dict(color='black',  linestyle='-',   marker='o', markersize=4),
    dict(color='0.30',   linestyle='--',  marker='s', markersize=4),
    dict(color='0.55',   linestyle='-.',  marker='^', markersize=4),
    dict(color='0.70',   linestyle=':',   marker='D', markersize=4),
]

THREAD_COLS = ['t1', 't2', 't4', 't8', 't16', 't32']
THREAD_VALS = [1,     2,    4,    8,    16,    32 ]

def load_csv(path):
    rows = []
    with open(path) as f:
        lines = [l for l in f if not l.startswith('#')]
    for row in csv.DictReader(lines):
        rows.append(row)
    return rows

def main():
    rows = load_csv(CSV_IN)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for row, style in zip(rows, DS_STYLES):
        xs, ys = [], []
        for col, t in zip(THREAD_COLS, THREAD_VALS):
            val = row.get(col, '').strip()
            if val:
                xs.append(t)
                ys.append(float(val))
        ax.plot(xs, ys, label=row['label'], linewidth=1.2, **style)

    # reference line at speedup = 1.0
    ax.axhline(1.0, color='black', linewidth=0.6, linestyle='--', alpha=0.5)
    ax.text(33, 1.02, '1×', fontsize=7, ha='right', va='bottom', alpha=0.7)

    ax.set_xscale('log', base=2)
    ax.set_xticks(THREAD_VALS)
    ax.set_xticklabels([str(t) for t in THREAD_VALS], fontsize=8)
    ax.set_xlabel('Thread count (T)', fontsize=8)
    ax.set_ylabel('Speedup over upstream fastp', fontsize=8)
    ax.set_title('fastp-gpu CPU mode vs. upstream fastp', fontsize=9, pad=4)

    ax.set_ylim(0.5, 3.4)
    ax.set_xlim(0.8, 40)

    # annotate the peak on WGS PE 40G (last row, T=32)
    last = rows[-1]
    peak_val = float(last['t32'])
    ax.annotate(f'{peak_val:.2f}×',
                xy=(32, peak_val), xytext=(20, peak_val - 0.5),
                fontsize=7, ha='center',
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    ax.legend(loc='upper left', fontsize=7, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(PNG_OUT)
    plt.savefig(PDF_OUT)
    print(f'Saved: {PNG_OUT}')
    print(f'Saved: {PDF_OUT}')

if __name__ == '__main__':
    main()
