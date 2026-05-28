#!/usr/bin/env python3
"""
Figure A (TUS paper): Upstream fastp v1.3.3 thread-scaling — bottleneck evidence.

Plots measured wall-clock time vs thread count for four WGS datasets, showing
the two structural bottlenecks discussed in Section V of the paper:

  B1 (decompression): The WGS SE curve is non-monotone — wall time at T=16
      exceeds T=4 because the single reader thread starves under worker
      contention (no async decompressor in upstream fastp).

  B3 (serial output gzip): The WGS PE curves flatten or invert above T=8/16
      because the single writer thread becomes the critical path regardless of
      how many worker threads are added.

Reads:  figA_tus_upstream_scaling.csv  (same directory)
Output: figA_tus_upstream_scaling.png / .pdf

NOTE on what this is NOT: the old figA_tus_stage_breakdown.csv contained stage
profiling durations from the fastp-gpu pipelined architecture where all three
thread groups (reader, workers, writers) run concurrently. The "loading" time
in that data is the time until reader.join(), which approximates total wall time
because the reader is on the critical path; the tiny "writing" tail is merely the
flush residual after workers drain, NOT total compression work. That framing was
misleading and has been replaced by this honest wall-clock comparison.

Style: grayscale, different line/marker styles, single-column width.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_IN  = os.path.join(HERE, 'figA_tus_upstream_scaling.csv')
PNG_OUT = os.path.join(HERE, 'figA_tus_upstream_scaling.png')
PDF_OUT = os.path.join(HERE, 'figA_tus_upstream_scaling.pdf')

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

    # annotate B1 on SE curve: non-monotone bump T=4→T=16
    # SE row: t4=137.915, t16=200.896  (T=16 is 45% slower than T=4)
    ax.annotate('B1: non-monotone\n(reader starved)',
                xy=(16, 200.896), xytext=(4, 320),
                fontsize=6.5, ha='center',
                arrowprops=dict(arrowstyle='->', color='black', lw=0.7))

    # annotate B3 on WGS PE 40G curve: T=32 (741s) is *slower* than T=8 (721s)
    # WGS_PE_40G: t8=721.188, t32=741.566
    ax.annotate('B3: T=32 slower\nthan T=8',
                xy=(32, 741.566), xytext=(14, 400),
                fontsize=6.5, ha='center',
                arrowprops=dict(arrowstyle='->', color='black', lw=0.7))

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(THREAD_VALS)
    ax.set_xticklabels([str(t) for t in THREAD_VALS], fontsize=8)
    ax.set_xlabel('Thread count (T)', fontsize=8)
    ax.set_ylabel('Wall time (s)', fontsize=8)
    ax.set_title('Upstream fastp v1.3.3 — WGS datasets', fontsize=9, pad=4)

    ax.set_xlim(0.8, 40)
    ax.set_ylim(80, 4000)

    # clean log-y tick labels
    from matplotlib.ticker import LogFormatter
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))

    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(PNG_OUT)
    plt.savefig(PDF_OUT)
    print(f'Saved: {PNG_OUT}')
    print(f'Saved: {PDF_OUT}')


if __name__ == '__main__':
    main()
