#!/usr/bin/env python3
"""
plot_fig_bioadv.py -- Figures 1 and 2 of the Bioinformatics Advances
Application Note, drawn at their printed size so that the fonts stay
legible (BIOADV R2, Reviewer 2).

Figure 1: CPU vs. GPU execution paths (one column wide, printed at
\\columnwidth). Labels follow the paper text and src/peprocessor.cpp:
workers call Filter::filterBatchGPU*() and receive pass flags and
statistics back before handing reads to the writer.

Figure 2: thread scaling, six datasets, three tools (printed at
0.98\\textwidth). Data: fig_bioadv_scaling.csv. It is the CSV the
paper's numbers were taken from (fig5_thread_scaling.csv before commit
20c57e5) minus one upstream cell (WGS_PE_40G, T=16, 1098.289 s). That
value was imported from the April OpenGene v1.1.0 file, not measured
with v1.3.3; the live v1.3.3 runs of that cell did not finish.

Saves fig_bioadv_arch.{pdf,png} and fig_bioadv_scaling.{pdf,png}.
Run from the repo root:
    python3 docs/publication/figures/plot_fig_bioadv.py
"""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 8,
    'axes.titlesize': 8.5, 'axes.labelsize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'savefig.dpi': 300,
})


def save(fig, stem):
    for ext in ('pdf', 'png'):
        path = os.path.join(HERE, f'{stem}.{ext}')
        fig.savefig(path, bbox_inches='tight', pad_inches=0.02,
                    facecolor='white')
        print('Saved:', path)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1
def fig_arch():
    C_IO, C_CPU, C_GPU = '#D8E8F8', '#EEF4E4', '#FEF0DC'
    C_EDGE, C_CA, C_GA = '#555555', '#335511', '#885500'

    fig, ax = plt.subplots(figsize=(3.4, 2.75))
    ax.set_xlim(0, 3.4)
    ax.set_ylim(0, 2.75)
    ax.axis('off')

    def box(cx, cy, w, h, lines, fc, ls='solid', ec=C_EDGE):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                     boxstyle='round,pad=0.02', facecolor=fc,
                     edgecolor=ec, linewidth=0.7, linestyle=ls))
        n = len(lines)
        for i, line in enumerate(lines):
            ty = cy + (n - 1) * 0.075 - i * 0.15
            ax.text(cx, ty, line, ha='center', va='center', fontsize=8,
                    fontweight='bold' if i == 0 else 'normal')

    def arrow(x1, y1, x2, y2, color=C_CA, style='->'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=0.9))

    cx, w = 0.92, 1.76
    rows = [
        (2.62, ['Input FASTQ.gz'], C_IO, 0.17),
        (2.18, ['Reader thread', 'async decompression'], C_CPU, 0.38),
        (1.55, ['Worker threads (\u00d7w)', 'trim, polyG/X, UMI;',
                'CPU mode: filter + stats'], C_CPU, 0.53),
        (0.83, ['Compressor pool', 'libdeflate, in order'], C_CPU, 0.38),
        (0.28, ['Output FASTQ.gz'], C_IO, 0.17),
    ]
    for y, lines, fc, h in rows:
        box(cx, y, w, h, lines, fc)
    for (y1, _, _, h1), (y2, _, _, h2) in zip(rows, rows[1:]):
        arrow(cx, y1 - h1 / 2 - 0.02, cx, y2 + h2 / 2 + 0.02)

    gx, gy, gw = 2.86, 1.55, 1.0
    box(gx, gy, gw, 0.53, ['GPU mode', 'filter + stats,', 'warp per read'],
        C_GPU, ls='dashed', ec=C_GA)
    x1, x2 = cx + w / 2 + 0.04, gx - gw / 2 - 0.04
    arrow(x1, gy + 0.10, x2, gy + 0.10, color=C_GA)
    arrow(x2, gy - 0.10, x1, gy - 0.10, color=C_GA)
    ax.text(gx, gy + 0.40, 'optional', ha='center', va='center',
            fontsize=7.5, color=C_GA, fontstyle='italic')
    save(fig, 'fig_bioadv_arch')


# ---------------------------------------------------------------- Figure 2
DS_ORDER = [
    ('Panel_SE_148M', 'Panel SE (148 MB)'),
    ('Panel_PE_304M', 'Panel PE (304 MB)'),
    ('WGS_SE_6.3G', 'WGS SE (6.3 GB)'),
    ('WGS_PE_12.8G', 'WGS PE (12.7 GB)'),
    ('WGS_PE_18.2G', 'WGS PE (18.1 GB)'),
    ('WGS_PE_40G', 'WGS PE (38.5 GB)'),
]
TOOLS = [
    ('opengene', 'fastp v1.3.3', '#4e79a7', 'o', '-', 1.4),
    ('d0bromir_gpu', 'fastp-gpu, GPU mode', '#e15759', 's', '-', 1.4),
    ('d0bromir_cpu', 'fastp-gpu, CPU mode', '#59a14f', '^', '--', 1.6),
]


def fig_scaling():
    bench = {}
    with open(os.path.join(HERE, 'fig_bioadv_scaling.csv')) as f:
        for row in csv.DictReader(f):
            bench[(row['dataset'], row['tool'], int(row['threads']))] = \
                float(row['walltime_s'])
    threads = sorted({t for (_, _, t) in bench})

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.0))
    for ax, (ds, label) in zip(axes.flat, DS_ORDER):
        for key, tlabel, color, marker, ls, lw in TOOLS:
            pts = [(t, bench[(ds, key, t)]) for t in threads
                   if (ds, key, t) in bench]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=marker,
                    linestyle=ls, color=color, label=tlabel, markersize=3.5,
                    linewidth=lw)
        ax.set_xscale('log', base=2)
        ax.set_xticks(threads)
        ax.set_xticklabels([str(t) for t in threads])
        ax.set_title(label)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel('Threads (-w)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Wall time (s)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.6, w_pad=0.6)
    save(fig, 'fig_bioadv_scaling')


if __name__ == '__main__':
    fig_arch()
    fig_scaling()
