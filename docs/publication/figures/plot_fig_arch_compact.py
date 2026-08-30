#!/usr/bin/env python3
"""
plot_fig_arch_compact.py -- Simplified CPU-vs-GPU flow diagram for the
Application Notes (Bioinformatics Advances) paper.

Reviewer 1 asked for "a very simplified flow diagram or similar that
compares the execution of the CPU vs GPU". This is a compact,
single-column-width version of figures/plot_fig_arch.py (which is
sized for JCB's single-column layout and too detailed/tall for a
4-page, two-column Application Note): fewer boxes, shorter labels,
landscape-ish proportions so it fits one column.

Saves:
    docs/publication/figures/fig_arch_compact.pdf
    docs/publication/figures/fig_arch_compact.png

Run from the repo root:
    python3 docs/publication/figures/plot_fig_arch_compact.py
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

FIG_W, FIG_H = 3.35, 3.9  # inches -- fits one column of a two-column page

C_IO   = '#D8E8F8'
C_CPU  = '#EEF4E4'
C_GPU  = '#FEF0DC'
C_EDGE = '#555555'
C_CA   = '#335511'
C_GA   = '#885500'

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')


def box(cx, cy, w, h, lines, fc, fs=6.3):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle='round,pad=0.012',
        facecolor=fc, edgecolor=C_EDGE, linewidth=0.7,
        transform=ax.transAxes, clip_on=False))
    if isinstance(lines, str):
        lines = [lines]
    n = len(lines)
    for i, line in enumerate(lines):
        ty = cy if n == 1 else cy + h * 0.32 - h * 0.64 * i / max(n - 1, 1)
        fw = 'bold' if (i == 0 and n > 1) else 'normal'
        ax.text(cx, ty, line, transform=ax.transAxes, ha='center',
                 va='center', fontsize=(fs if i == 0 else fs - 0.4),
                 fontweight=fw)


def varrow(cx, y1, y2, color=C_CA):
    ax.annotate('', xy=(cx, y2), xytext=(cx, y1),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color=color, lw=0.8))


def sarrow(x1, y1, x2, y2, color=C_CA):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color=color, lw=0.8))


CX = 0.32   # shared CPU-column x
GX = 0.78   # GPU-column x

# CPU column, top to bottom
Y_IN   = 0.955
Y_RD   = 0.855
Y_TRIM = 0.735   # Algorithm 4 + parse + trim/filter, combined for compactness
Y_WORK = 0.560   # worker threads, Algorithm 1
Y_WR   = 0.115
Y_OUT  = 0.03

box(CX, Y_IN,   0.50, 0.055, 'Input FASTQ.gz', C_IO, fs=6.5)
box(CX, Y_RD,   0.50, 0.055, 'Reader thread', C_CPU, fs=6.5)
box(CX, Y_TRIM, 0.50, 0.11,
    ['Alg. 4: async decompress', '+ parse, trim, filter'], C_CPU)
box(CX, Y_WORK, 0.50, 0.11,
    ['Worker threads (xT)', 'Alg. 1: reversible stats'], C_CPU)
box(CX, Y_WR,   0.50, 0.055, 'Writer + compressor pool', C_CPU, fs=6.5)
box(CX, Y_OUT,  0.50, 0.055, 'Output FASTQ.gz', C_IO, fs=6.5)

varrow(CX, Y_IN - 0.0275, Y_RD + 0.0275)
varrow(CX, Y_RD - 0.0275, Y_TRIM + 0.055)
varrow(CX, Y_TRIM - 0.055, Y_WORK + 0.055)
varrow(CX, Y_WORK - 0.055, Y_WR + 0.0275)
varrow(CX, Y_WR - 0.0275, Y_OUT + 0.0275)

ax.text(0.02, (Y_WORK + Y_WR) / 2, 'CPU\nmode', transform=ax.transAxes,
        ha='center', va='center', fontsize=5.5, color='#336600',
        fontweight='bold', rotation=90)

# GPU box (optional path, dashed container), between worker and writer
Y_GPU = (Y_WORK + Y_WR) / 2
gx, gw = GX - 0.20, 0.40
gtop, gbot = Y_GPU + 0.10, Y_GPU - 0.10
ax.add_patch(FancyBboxPatch(
    (gx, gbot), gw, gtop - gbot, boxstyle='round,pad=0.006',
    facecolor='#FBF5EE', edgecolor='#CCAA66', linewidth=0.8,
    linestyle='dashed', transform=ax.transAxes, clip_on=False, zorder=0))
ax.text(GX, gtop + 0.018, 'GPU (optional)', transform=ax.transAxes,
        ha='center', va='bottom', fontsize=6, color=C_GA, fontweight='bold')
box(GX, Y_GPU, 0.34, 0.13,
    ['Alg. 2/3: warp-per-read', 'kernel, multi-slot', 'PCIe scheduler'],
    C_GPU, fs=5.8)

sarrow(CX + 0.25, Y_WORK, GX - 0.17, Y_GPU + 0.05, color=C_GA)
sarrow(GX - 0.17, Y_GPU - 0.05, CX + 0.25, Y_WR, color=C_GA)
ax.text((CX + 0.25 + GX - 0.17) / 2, (Y_WORK + Y_GPU) / 2 + 0.01,
        'GPU\nmode', transform=ax.transAxes, ha='center', va='center',
        fontsize=5.5, color=C_GA, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                  edgecolor=C_GA, linewidth=0.4, alpha=0.9))

plt.tight_layout(pad=0.03)
for ext in ('pdf', 'png'):
    path = os.path.join(HERE, f'fig_arch_compact.{ext}')
    dpi = 300 if ext == 'png' else 150
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    print(f'Saved: {path}')
plt.close(fig)
