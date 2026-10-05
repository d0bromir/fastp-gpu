#!/usr/bin/env python3
"""
plot_fig_arch.py  –  Pipeline architecture diagram for the fastp-gpu paper.

Saves:
    docs/publication/figures/fig_arch.pdf
    docs/publication/figures/fig_arch.png

Run from the repo root:
    python3 docs/publication/figures/plot_fig_arch.py
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Figure size ───────────────────────────────────────────────────────────────
FIG_W, FIG_H = 6.0, 8.5   # inches

# ── Column layout (axes fraction 0–1) ────────────────────────────────────────
# CPU pipeline: centre-x, width
CX_CPU, BW_CPU = 0.27, 0.44   # right edge = 0.49
# GPU path: centre-x, width
CX_GPU, BW_GPU = 0.78, 0.38   # left edge  = 0.59

# ── Box heights (axes fraction) ───────────────────────────────────────────────
BH_1 = 0.060   # one line
BH_2 = 0.085   # two lines
BH_3 = 0.100   # three lines

# ── Box specs  (cy, bh) ───────────────────────────────────────────────────────
# CPU pipeline, top → bottom
Y_INPUT   = (0.955, BH_1)
Y_READER  = (0.855, BH_1)
Y_DECOMP  = (0.730, BH_2)
Y_PARSER  = (0.610, BH_1)
Y_WORKERS = (0.490, BH_3)
Y_WRITER  = (0.105, BH_1)
Y_OUTPUT  = (0.020, BH_1)

# GPU column — packed between workers.bot and writer.top
_GAP    = 0.020
_wbot   = Y_WORKERS[0] - Y_WORKERS[1] / 2          # 0.440
_bat_cy = _wbot  - _GAP - BH_2 / 2                 # 0.3775
_sch_cy = _bat_cy - BH_2 / 2 - _GAP - BH_3 / 2    # 0.2650
_gpu_cy = _sch_cy - BH_3 / 2 - _GAP - BH_2 / 2    # 0.1525
Y_BATCHER = (_bat_cy, BH_2)
Y_SCHED   = (_sch_cy, BH_3)
Y_GPUK    = (_gpu_cy, BH_2)

# ── Colours ───────────────────────────────────────────────────────────────────
C_IO    = '#D8E8F8'   # light blue    – I/O nodes
C_CPU   = '#EEF4E4'   # light green   – CPU-side stages
C_WORK  = '#D5EED5'   # medium green  – worker threads
C_GPUB  = '#FEF0DC'   # light amber   – GPU-side boxes
C_GPUBG = '#FBF5EE'   # pale amber    – GPU device background
C_EDGE  = '#555555'
C_CA    = '#335511'   # CPU-path arrow colour
C_GA    = '#885500'   # GPU-path arrow colour


# ── Helpers ───────────────────────────────────────────────────────────────────
def bot(spec):
    return spec[0] - spec[1] / 2


def top(spec):
    return spec[0] + spec[1] / 2


def draw_box(ax, cx, spec, bw, lines, fc, fs=7.5):
    """Rounded rectangle with centred multi-line text (first line bold)."""
    cy, bh = spec
    ax.add_patch(FancyBboxPatch(
        (cx - bw / 2, cy - bh / 2), bw, bh,
        boxstyle='round,pad=0.013',
        facecolor=fc, edgecolor=C_EDGE, linewidth=0.75,
        transform=ax.transAxes, clip_on=False))
    if isinstance(lines, str):
        lines = [lines]
    n = len(lines)
    for i, line in enumerate(lines):
        ty = cy if n == 1 else cy + bh * 0.70 / 2 - bh * 0.70 * i / (n - 1)
        fw = 'bold' if (i == 0 and n > 1) else 'normal'
        ax.text(cx, ty, line, transform=ax.transAxes,
                ha='center', va='center',
                fontsize=(fs if i == 0 else fs - 0.5),
                fontweight=fw)


def varrow(ax, cx, y1, y2, color=C_CA):
    """Vertical arrow (downward) from y1 to y2 at horizontal position cx."""
    ax.annotate('', xy=(cx, y2), xytext=(cx, y1),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color=color, lw=0.9))


def sarrow(ax, x1, y1, x2, y2, color=C_CA):
    """Straight diagonal arrow from (x1,y1) to (x2,y2)."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color=color, lw=0.9,
                                connectionstyle='arc3,rad=0.0'))


# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# GPU device background (dashed rounded rect behind the three GPU boxes)
gx   = CX_GPU - BW_GPU / 2 - 0.025
gw   = BW_GPU + 0.050
gtop = top(Y_BATCHER) + 0.020
gbot = bot(Y_GPUK)    - 0.020
ax.add_patch(FancyBboxPatch(
    (gx, gbot), gw, gtop - gbot,
    boxstyle='round,pad=0.005',
    facecolor=C_GPUBG, edgecolor='#CCAA66',
    linewidth=0.9, linestyle='dashed',
    transform=ax.transAxes, clip_on=False, zorder=0))
ax.text(CX_GPU, gtop + 0.013, 'GPU device',
        transform=ax.transAxes, ha='center', va='bottom',
        fontsize=6.5, color=C_GA, fontweight='bold')

# ── CPU pipeline boxes ────────────────────────────────────────────────────────
draw_box(ax, CX_CPU, Y_INPUT,   BW_CPU, 'Input: FASTQ.gz  /  BGZF', C_IO, fs=8)
draw_box(ax, CX_CPU, Y_READER,  BW_CPU, 'Reader thread', C_CPU, fs=8)
draw_box(ax, CX_CPU, Y_DECOMP,  BW_CPU,
         ['Algorithm 4 — Async decompressor',
          'ISA-L  isal_inflate  (background thread)'], C_CPU)
draw_box(ax, CX_CPU, Y_PARSER,  BW_CPU, 'FASTQ parser  →  read pool', C_CPU, fs=8)
draw_box(ax, CX_CPU, Y_WORKERS, BW_CPU,
         ['Worker threads  (×T)',
          'Algorithm 1 — Reversible statistics',
          'adapter trimming  ·  quality filter'], C_WORK)
draw_box(ax, CX_CPU, Y_WRITER,  BW_CPU, 'Writer  +  compressor pool', C_CPU, fs=8)
draw_box(ax, CX_CPU, Y_OUTPUT,  BW_CPU, 'Output: FASTQ.gz', C_IO, fs=8)

# ── GPU column boxes ──────────────────────────────────────────────────────────
draw_box(ax, CX_GPU, Y_BATCHER, BW_GPU,
         ['Batch packer',
          'pinned host buffers'], C_GPUB)
draw_box(ax, CX_GPU, Y_SCHED,   BW_GPU,
         ['Algorithm 3 — Multi-slot scheduler',
          'm = 8 slots,  CUDA streams',
          'H\u2192D  /  D\u2192H  (PCIe)'], C_GPUB)
draw_box(ax, CX_GPU, Y_GPUK,    BW_GPU,
         ['Algorithm 2 — Warp-per-read kernels',
          '1 warp (32 lanes) per read'], C_GPUB)

# ── CPU pipeline vertical arrows ──────────────────────────────────────────────
varrow(ax, CX_CPU, bot(Y_INPUT),   top(Y_READER))
varrow(ax, CX_CPU, bot(Y_READER),  top(Y_DECOMP))

# "double-buffered" annotation between Reader and Decomp
mid_rd = (bot(Y_READER) + top(Y_DECOMP)) / 2
ax.text(CX_CPU + BW_CPU / 2 + 0.013, mid_rd,
        'double-\nbuffered',
        transform=ax.transAxes, ha='left', va='center',
        fontsize=5.5, color='#446688', style='italic')

varrow(ax, CX_CPU, bot(Y_DECOMP),  top(Y_PARSER))
varrow(ax, CX_CPU, bot(Y_PARSER),  top(Y_WORKERS))

# CPU-path direct arrow Workers → Writer (bypasses GPU)
varrow(ax, CX_CPU, bot(Y_WORKERS), top(Y_WRITER), color=C_CA)

# CPU-path label (rotated, left of the arrow)
ax.text(0.025, (bot(Y_WORKERS) + top(Y_WRITER)) / 2,
        'CPU path', transform=ax.transAxes,
        ha='center', va='center',
        fontsize=5.5, color='#336600', fontweight='bold', rotation=90)

varrow(ax, CX_CPU, bot(Y_WRITER),  top(Y_OUTPUT))

# ── GPU branch arrows ─────────────────────────────────────────────────────────
# Workers right edge → Batcher left edge  (offload to GPU)
sarrow(ax, CX_CPU + BW_CPU / 2, Y_WORKERS[0],
       CX_GPU - BW_GPU / 2,     Y_BATCHER[0], color=C_GA)

# GPU vertical chain
varrow(ax, CX_GPU, bot(Y_BATCHER), top(Y_SCHED),  color=C_GA)
varrow(ax, CX_GPU, bot(Y_SCHED),   top(Y_GPUK),   color=C_GA)

# GPU_K left edge → Writer right edge  (results return)
sarrow(ax, CX_GPU - BW_GPU / 2, Y_GPUK[0],
       CX_CPU + BW_CPU / 2,     Y_WRITER[0], color=C_GA)

# "GPU path" label in the column gap
mid_gap = (Y_WORKERS[0] + Y_WRITER[0]) / 2
ax.text((CX_CPU + BW_CPU / 2 + CX_GPU - BW_GPU / 2) / 2, mid_gap,
        'GPU\npath',
        transform=ax.transAxes, ha='center', va='center',
        fontsize=6, color=C_GA, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                  edgecolor=C_GA, linewidth=0.5, alpha=0.85))

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.05)
for ext in ('pdf', 'png'):
    path = os.path.join(HERE, f'fig_arch.{ext}')
    dpi  = 300 if ext == 'png' else 150
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    print(f'Saved: {path}')
plt.close(fig)
