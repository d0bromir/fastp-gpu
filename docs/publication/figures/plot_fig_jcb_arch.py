#!/usr/bin/env python3
"""
plot_fig_jcb_arch.py  –  Pipeline architecture diagram for the JCB paper.

Same content as plot_fig_arch.py, drawn at the printed column width
(3.4 in) so that no text is scaled down. The smallest text is 6.5 pt.
fig_arch.* (TCBB, ACM BCB) is not touched.

Saves:
    docs/publication/figures/fig_jcb_arch.{pdf,eps,png}
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

C_IO    = '#D8E8F8'
C_CPU   = '#EEF4E4'
C_WORK  = '#D5EED5'
C_GPUB  = '#FEF0DC'
C_GPUBG = '#FBF5EE'
C_EDGE  = '#555555'
C_CA    = '#335511'
C_GA    = '#885500'

W = 3.4                      # printed column width, inches
X_CPU, W_CPU = 0.05, 1.62    # left edge, width
X_GPU, W_GPU = 1.88, 1.47
GAP = 0.13                   # vertical gap between stacked boxes
LINE = 0.135                 # line pitch inside a box
PAD = 0.07                   # top/bottom text padding inside a box
FS_B, FS_N = 7.5, 6.8        # bold first line, normal lines


def box_h(n):
    return 2 * PAD + n * LINE


class Box:
    def __init__(self, x, w, top, lines, fc):
        self.x, self.w, self.top, self.lines, self.fc = x, w, top, lines, fc
        self.h = box_h(len(lines))

    @property
    def bot(self):
        return self.top - self.h

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.top - self.h / 2

    def draw(self, ax):
        ax.add_patch(FancyBboxPatch((self.x, self.bot), self.w, self.h,
                                    boxstyle='round,pad=0,rounding_size=0.05',
                                    facecolor=self.fc, edgecolor=C_EDGE, linewidth=0.6))
        for i, (txt, bold) in enumerate(self.lines):
            ax.text(self.cx, self.top - PAD - LINE * (i + 0.5), txt, ha='center', va='center',
                    fontsize=FS_B if bold else FS_N, fontweight='bold' if bold else 'normal')


def stack(x, w, top, specs):
    out = []
    for lines, fc in specs:
        b = Box(x, w, top, lines, fc)
        out.append(b)
        top = b.bot - GAP
    return out


def arrow(ax, x1, y1, x2, y2, color):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=0.8,
                                shrinkA=0, shrinkB=0))


B, N = True, False
top0 = 10.0                  # provisional; the axes are cropped below
inp, rdr, dec, par, wrk = stack(X_CPU, W_CPU, top0, [
    ([('Input: FASTQ.gz / BGZF', N)], C_IO),
    ([('Reader thread', N)], C_CPU),
    ([('Algorithm 4', B), ('async double-buffered', N), ('decompression (ISA-L)', N)], C_CPU),
    ([('FASTQ parser → read pool', N)], C_CPU),
    ([('Worker threads (×T)', B), ('trimming, quality filter', N),
      ('Algorithm 1:', N), ('reversible statistics', N)], C_WORK),
])
pack, sch, ker = stack(X_GPU, W_GPU, wrk.cy + 0.02, [
    ([('Batch packer', B), ('pinned host buffers', N)], C_GPUB),
    ([('Algorithm 3', B), ('multi-slot scheduler', N), ('8 slots, CUDA streams', N),
      ('H→D / D→H over PCIe', N)], C_GPUB),
    ([('Algorithm 2', B), ('warp-per-read kernels', N), ('1 warp (32 lanes) per read', N)], C_GPUB),
])
wtr, out = stack(X_CPU, W_CPU, ker.cy + box_h(1) / 2, [
    ([('Writer + compressor pool', N)], C_CPU),
    ([('Output: FASTQ.gz', N)], C_IO),
])

ymax = inp.top + 0.03
ymin = out.bot - 0.03
H = ymax - ymin
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(ymin, ymax)
ax.axis('off')

# GPU offload frame (packer and scheduler run on the host, kernel on the device)
fx, fw = X_GPU - 0.05, W_GPU + 0.08
ftop, fbot = pack.top + 0.06, ker.bot - 0.06
ax.add_patch(FancyBboxPatch((fx, fbot), fw, ftop - fbot,
                            boxstyle='round,pad=0,rounding_size=0.05',
                            facecolor=C_GPUBG, edgecolor='#CCAA66', linewidth=0.8,
                            linestyle='dashed', zorder=0))
ax.text(fx + fw / 2, ftop + 0.035, 'GPU offload (host + device)', ha='center', va='bottom',
        fontsize=7, color=C_GA, fontweight='bold')

for b in (inp, rdr, dec, par, wrk, pack, sch, ker, wtr, out):
    b.draw(ax)

# CPU path
for a, b in ((inp, rdr), (rdr, dec), (dec, par), (par, wrk), (wrk, wtr), (wtr, out)):
    arrow(ax, a.cx, a.bot, a.cx, b.top, C_CA)
ax.text(rdr.cx + 0.05, (rdr.bot + dec.top) / 2, 'double-buffered', ha='left', va='center',
        fontsize=6.5, color='#446688', style='italic')
ax.text(wrk.cx - 0.05, (wrk.bot + wtr.top) / 2, 'CPU path', ha='right', va='center',
        fontsize=7, color=C_CA, fontweight='bold', rotation=90)

# GPU path
arrow(ax, wrk.x + wrk.w, wrk.cy - 0.10, pack.x, pack.cy, C_GA)
arrow(ax, pack.cx, pack.bot, sch.cx, sch.top, C_GA)
arrow(ax, sch.cx, sch.bot, ker.cx, ker.top, C_GA)
arrow(ax, ker.x, ker.cy, wtr.x + wtr.w, wtr.cy, C_GA)
ax.text((wrk.x + wrk.w + fx) / 2 - 0.015, sch.cy, 'GPU path', ha='center', va='center',
        fontsize=7, color=C_GA, fontweight='bold', rotation=90)

for ext in ('pdf', 'eps', 'png'):
    path = os.path.join(HERE, f'fig_jcb_arch.{ext}')
    fig.savefig(path, dpi=300 if ext == 'png' else 150)
    print(f'Saved: {path}')
plt.close(fig)
