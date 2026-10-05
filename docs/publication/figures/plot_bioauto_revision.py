#!/usr/bin/env python3
"""Figures for the BIOAUTOMATION revision, drawn from the raw measurement files.

Reads  ../supplementary/bioautomation-revision/raw/...  (copied from galaxy and a2)
Writes fig_bioauto_scaling.pdf, fig_bioauto_stage.pdf, fig_bioauto_ablation.pdf (when data exist).

Palette: validated categorical slots 1-3 (blue, orange, aqua) from the dataviz skill.
Identity is never colour alone: each tool has its own marker shape and a direct label.
"""
import csv, glob, os, re, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "supplementary", "bioautomation-revision", "raw")
C = {"opengene": "#2a78d6", "d0bromir_cpu": "#eb6834", "d0bromir_gpu": "#1baf7a"}
LAB = {"opengene": "fastp v1.3.3", "d0bromir_cpu": "fastp-gpu CPU", "d0bromir_gpu": "fastp-gpu GPU"}
MK = {"opengene": "o", "d0bromir_cpu": "s", "d0bromir_gpu": "^"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "sans-serif"})


def sweep_data():
    idx = os.path.join(RAW, "galaxy", "results", "sweep_index.txt")
    runs = defaultdict(lambda: defaultdict(list)); fails = defaultdict(lambda: defaultdict(int))
    if not os.path.exists(idx): return runs, fails
    seen = set()
    for line in open(idx):
        m = re.match(r"rep=(\d+) ds=(\S+) T=(\d+) dir=(\S+)", line)
        if not m: continue
        d = os.path.basename(m.group(4))
        if d in seen: continue
        seen.add(d)
        for f in glob.glob(os.path.join(RAW, "galaxy", "benchmark_results", d, "full_benchmark_*.csv")):
            for r in csv.DictReader(open(f)):
                key = (r["dataset"], int(r["threads"]))
                if r["validation"] in ("TIMEOUT", "IDLE_TIMEOUT", "CRASH", "FAIL", "crashed"):
                    fails[key][r["tool"]] += 1; continue
                try: runs[key][r["tool"]].append(float(r["walltime_s"]))
                except ValueError: pass
    return runs, fails


def scaling(runs, fails, ds, title, fname):
    Ts = sorted({k[1] for k in runs if k[0] == ds} | {k[1] for k in fails if k[0] == ds})
    if not Ts: return
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    shift = {"opengene": 0, "d0bromir_cpu": 9, "d0bromir_gpu": -9}   # label offsets (points) so labels never overlap
    for tool in ("opengene", "d0bromir_cpu", "d0bromir_gpu"):
        xs, ys, es = [], [], []
        for T in Ts:
            v = runs[(ds, T)].get(tool, [])
            nf = fails[(ds, T)].get(tool, 0)
            # A cell where most runs failed is drawn as a hollow point and is not joined to the line.
            if v and nf > len(v):
                ax.errorbar([T], [st.mean(v)], yerr=[0], color=C[tool], marker=MK[tool], ms=7, mfc="none", mew=1.5, ls="none")
                continue
            if v: xs.append(T); ys.append(st.mean(v)); es.append(st.stdev(v) if len(v) > 1 else 0)
        ax.errorbar(xs, ys, yerr=es, color=C[tool], marker=MK[tool], ms=6, lw=1.6, capsize=3, label=LAB[tool],
                    markeredgecolor="#fcfcfb", markeredgewidth=1)
        if xs: ax.annotate(LAB[tool], (xs[-1], ys[-1]), xytext=(8, shift[tool]), textcoords="offset points", color=INK, va="center", fontsize=8)
    ymax = ax.get_ylim()[1]
    for T in Ts:   # note cells where the baseline did not complete
        nf, nok = fails[(ds, T)].get("opengene", 0), len(runs[(ds, T)].get("opengene", []))
        if nf:
            if nok:
                y = st.mean(runs[(ds, T)]["opengene"])
                ax.annotate(f"fastp v1.3.3 hung in {nf} of {nf+nok} runs;\nhollow point = the {nok} run that finished", (T, y),
                            xytext=(10, 0), textcoords="offset points", ha="left", va="center", fontsize=7.5, color=C["opengene"])
            else:
                ax.annotate(f"fastp v1.3.3 hung in\nall {nf} runs", (T, ymax * 0.5), ha="center", fontsize=7.5, color=C["opengene"])
    ax.set_xscale("log", base=2); ax.set_xticks(Ts); ax.set_xticklabels([str(t) for t in Ts]); ax.minorticks_off()
    ax.set_xlabel("Worker threads (-w)"); ax.set_ylabel("Wall time (s), mean $\\pm$ SD"); ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.set_xlim(Ts[0] / 1.25, Ts[-1] * 2.6)
    ax.set_ylim(0, ymax * 1.12)
    fig.savefig(os.path.join(HERE, fname)); plt.close(fig)


def stage_split(runs):
    f = os.path.join(RAW, "galaxy", "results", "stage_split.log")
    if not os.path.exists(f): return
    d = defaultdict(lambda: defaultdict(list))
    for line in open(f):
        m = re.match(r"(\S+) rep=\d+ mode=(\d) seconds=([\d.]+)", line)
        if m: d[m.group(1)][int(m.group(2))].append(float(m.group(3)))
    key = "WGS/DRR216653_1"
    if key not in d or not d[key][2]: return
    r0, r1, r2 = (st.mean(d[key][k]) for k in (0, 1, 2))
    walls = {T: st.mean(runs[("WGS_PE_40G", T)]["d0bromir_cpu"]) for T in (8, 32) if runs[("WGS_PE_40G", T)].get("d0bromir_cpu")}
    fig, ax = plt.subplots(figsize=(5.6, 2.4))
    rows = [("One mate file, one thread,\nno overlap", [(r0, "#86b6ef", "storage read"), (r1 - r0, "#2a78d6", "ISA-L inflate"), (r2 - r1, "#184f95", "record scan")])]
    for T in sorted(walls, reverse=True):
        rows.append((f"fastp-gpu CPU,\n{T} threads (whole pipeline)", [(walls[T], "#eb6834", "")]))
    for i, (lab, segs) in enumerate(rows):
        left = 0
        for w, col, name in segs:
            ax.barh(i, w, left=left, color=col, edgecolor="#fcfcfb", linewidth=1.5, height=0.55)
            if name and w > 30: ax.text(left + w / 2, i, f"{name}\n{w:.0f} s", ha="center", va="center", fontsize=7.5, color="white")
            left += w
        ax.text(left + 6, i, f"{left:.0f} s", va="center", fontsize=8, color=INK)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#86b6ef", label="storage read"), Patch(color="#2a78d6", label="ISA-L inflate"), Patch(color="#184f95", label="record scan")],
              loc="lower right", fontsize=7.5, frameon=False)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("Seconds (41.3 GB paired-end dataset; R1 file for the stage split)"); ax.set_xlim(0, max(walls.values(), default=0) * 1.12 if walls else 650)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.savefig(os.path.join(HERE, "fig_bioauto_stage.pdf")); plt.close(fig)


def ablation():
    f = os.path.join(RAW, "galaxy", "results_ablation", "ablation.csv")
    if not os.path.exists(f): return
    d = defaultdict(list)
    for r in csv.DictReader(open(f)):
        if r["exit"] == "0": d[(r["config"], int(r["threads"]))].append(float(r["wall_s"]))
    order = ["og", "all_off", "+async", "full_cpu", "loo_async", "fast_prestats", "gpu"]
    lab = {"og": "fastp v1.3.3", "all_off": "fastp-gpu, switches off", "+async": "+ async decompression", "fast_prestats": "opt-in fast pre-filter\n(not output-identical)",
           "full_cpu": "+ parallel compression\n(full CPU)", "gpu": "full GPU build", "loo_async": "full CPU without\nasync decompression"}
    col = {"og": C["opengene"], "gpu": C["d0bromir_gpu"]}
    Ts = sorted({k[1] for k in d})
    if not Ts: return
    fig, axes = plt.subplots(1, len(Ts), figsize=(3.4 * len(Ts), 3.3), sharey=False)
    axes = [axes] if len(Ts) == 1 else axes
    for ax, T in zip(axes, Ts):
        for i, c in enumerate(order):
            v = d.get((c, T), [])
            if not v: continue
            ax.barh(i, st.mean(v), xerr=st.stdev(v) if len(v) > 1 else 0, color=col.get(c, C["d0bromir_cpu"]), edgecolor="#fcfcfb", height=0.6, capsize=2)
            ax.text(st.mean(v) + 8, i, f"{st.mean(v):.0f}", va="center", fontsize=7.5, color=INK)
        ax.set_yticks(range(len(order))); ax.set_yticklabels([lab[c] for c in order] if ax is axes[0] else [], fontsize=7.5)
        ax.invert_yaxis(); ax.set_title(f"{T} threads", loc="left", fontsize=9.5); ax.set_xlabel("Wall time (s)")
        ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.savefig(os.path.join(HERE, "fig_bioauto_ablation.pdf")); plt.close(fig)


if __name__ == "__main__":
    runs, fails = sweep_data()
    scaling(runs, fails, "WGS_PE_40G", "Paired-end 41.3 GB (ARM host)", "fig_bioauto_scaling.pdf")
    scaling(runs, fails, "WGS_SE_6.3G", "Single-end 6.7 GB (ARM host)", "fig_bioauto_scaling_se.pdf")
    stage_split(runs)
    ablation()
    print("figures written")
