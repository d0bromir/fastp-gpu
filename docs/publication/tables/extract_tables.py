#!/usr/bin/env python3
"""
extract_tables.py — Reproduce all paper tables as CSVs from raw benchmark data.

Usage:
    python3 extract_tables.py [--outdir DIR]

Output: one CSV per paper table in the output directory (default: same dir as this script).

Sources
-------
  benchmark_results/fastp-gpu_v1.2.2/vs_opengene_v1.1.0/galaxy_arm_a100/20260417/full_benchmark_20260417.csv
      columns: dataset,type,tool,threads,walltime_s,reads_in,reads_out,pct_vs_opengene,fastq_files
      covers: gzip-output runs, tools = opengene | d0bromir_cpu | d0bromir_gpu,
              T = 1,2,4,8,16 (all datasets) + T=32 (Panel_SE, Panel_PE, WGS_SE, WGS_PE_12.8G)

  benchmark_results/fastp-gpu_v1.2.2/vs_opengene_v1.1.0/galaxy_arm_a100/20260418_085816_no_gzip/paper_bench_20260418_085816_no_gzip.csv
      columns: dataset,type,tool,threads,walltime_s,reads_in,reads_out
      covers: SE datasets only, no-gzip control at T=8

  benchmark_results/fastp-gpu_v1.2.2/vs_opengene_v1.1.0/galaxy_arm_a100/20260418_085816_no_gzip/paper_bench_20260418_085816_pe_no_gzip.log
      plain-text log; contains [PROF_CSV] blocks for PE no-gzip runs at T=8.
      run order (9 fastp runs):
        WGS_PE_12.8G opengene, d0bromir_cpu, d0bromir_gpu
        WGS_PE_18.2G opengene, d0bromir_cpu, d0bromir_gpu
        WGS_PE_40G   opengene, d0bromir_cpu, d0bromir_gpu
      only d0bromir_cpu and d0bromir_gpu emit [PROF_CSV] → 6 data rows total

  docs/publication/figures/fig2_kernel_speedup.csv
      columns: dataset,threads,d0bromir_cpu_filtering_ms,d0bromir_gpu_filtering_ms,
               gpu_vs_cpu_filtering_speedup,note

  docs/publication/figures/fig4_transfer_overhead.csv
      columns: dataset,type,threads,opengene_s,d0bromir_cpu_s,d0bromir_gpu_s,
               gpu_overhead_vs_cpu_s,gpu_overhead_vs_cpu_pct,note
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


# ── Path resolution ───────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent.parent.parent   # docs/publication/tables -> repo root

CONSOLIDATED_CSV  = REPO_ROOT / "benchmark_results" / "fastp-gpu_v1.2.2" / "vs_opengene_v1.1.0" / "galaxy_arm_a100" / "20260417" / "full_benchmark_20260417.csv"
FIG2_CSV          = REPO_ROOT / "docs" / "publication" / "figures" / "fig2_kernel_speedup.csv"
FIG4_CSV          = REPO_ROOT / "docs" / "publication" / "figures" / "fig4_transfer_overhead.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path.name}  ({len(rows)} rows)")


def speedup_str(baseline: float, candidate: float) -> str:
    if baseline == 0:
        return "N/A"
    return f"{baseline / candidate:.2f}x"


# ── Dataset ordering (for consistent output) ──────────────────────────────────

DS_ORDER = [
    "Panel_SE_148M",
    "Panel_PE_304M",
    "WGS_SE_6.3G",
    "WGS_PE_12.8G",
    "WGS_PE_18.2G",
    "WGS_PE_40G",
]

THREAD_ORDER = [1, 2, 4, 8, 16, 32]


# ══════════════════════════════════════════════════════════════════════════════
# Table 1 — Hardware configuration
# Source: static metadata — transcribed from the test system
# Paper section: 6.1
# ══════════════════════════════════════════════════════════════════════════════

# Hardware rows are static (not derived from benchmark runs).
# They are defined here as the single source of truth so that the CSV
# can be regenerated consistently alongside the other tables.
HARDWARE_ROWS = [
    {"parameter": "CPU model",     "value": "ARM Neoverse N1"},
    {"parameter": "CPU cores",     "value": "128 (single socket, 1 thread/core)"},
    {"parameter": "CPU clock",     "value": "3.0 GHz (max)"},
    {"parameter": "CPU cache",     "value": "L1d 64 KB x128 / L1i 64 KB x128 / L2 1 MB x128"},
    {"parameter": "GPU",           "value": "2x NVIDIA A100 80 GB PCIe (Gen4 x16)"},
    {"parameter": "GPU driver",    "value": "590.48.01"},
    {"parameter": "RAM",           "value": "256 GB DDR4"},
    {"parameter": "Storage",       "value": "2x Samsung PM9A3 894 GB NVMe SSD"},
    {"parameter": "OS",            "value": "Ubuntu 25.10 (aarch64), Linux 6.17.0"},
    {"parameter": "CUDA Toolkit",  "value": "12.6"},
    {"parameter": "GCC",           "value": "15.2.0"},
]


def make_table1(outdir: Path) -> None:
    """Static hardware spec table — no external source file required."""
    write_csv(outdir / "table1_hardware.csv", ["parameter", "value"], HARDWARE_ROWS)


# ══════════════════════════════════════════════════════════════════════════════
# Table 3 — Benchmark datasets
# Source: static metadata — accession numbers and dataset properties
# Paper section: 6.2
# ══════════════════════════════════════════════════════════════════════════════

# Dataset rows match §6.2 of the paper.
# read_count_m = millions of reads (SE count; PE listed as read-pairs).
# size_gb reflects the compressed FASTQ size on disk.
DATASET_ROWS = [
    {"label": "Panel SE 148M", "accession": "S1A_S1_L001_R1",      "size":  "148 MB",             "reads": "1.38M",   "read_length": "151 bp", "type": "Panel, SE"},
    {"label": "Panel PE 304M", "accession": "S1A_S1_L001 (R1+R2)", "size":  "304 MB (148+156 MB)", "reads": "2.76M",   "read_length": "151 bp", "type": "Panel, PE"},
    {"label": "WGS SE 6.3G",   "accession": "ERR1044780",           "size":  "6.3 GB",              "reads": "80.3M",   "read_length": "100 bp", "type": "WGS, SE"},
    {"label": "WGS PE 12.7G",  "accession": "ERR1044780 (R1+R2)",  "size":  "12.7 GB",             "reads": "160.6M",  "read_length": "100 bp", "type": "WGS, PE"},
    {"label": "WGS PE 18.1G",  "accession": "ERR1044319",           "size":  "18.1 GB",             "reads": "234.2M",  "read_length": "100 bp", "type": "WGS, PE"},
    {"label": "WGS PE 38.5G",  "accession": "DRR216653",            "size":  "38.5 GB (19.2+19.3)", "reads": "722.6M",  "read_length": "150 bp", "type": "WGS, PE"},
]


def make_table3(outdir: Path) -> None:
    """Static dataset inventory — no external source file required."""
    fields = ["label", "accession", "size", "reads", "read_length", "type"]
    write_csv(outdir / "table3_datasets.csv", fields, DATASET_ROWS)


# ══════════════════════════════════════════════════════════════════════════════
# Table 2 — Wall-clock time with parallel output compression
# Source: full_benchmark_20260417.csv
# Paper section: 7.2
# ══════════════════════════════════════════════════════════════════════════════

def make_table2(outdir: Path) -> None:
    """
    Three-way wall time comparison: opengene vs d0bromir_cpu vs d0bromir_gpu,
    all gzip-output runs across all datasets and thread counts.
    Speedup = opengene_s / tool_s.
    """
    rows_raw = load_csv(CONSOLIDATED_CSV)

    # Index: (dataset, threads) -> {tool: walltime_s}
    index: dict[tuple, dict] = defaultdict(dict)
    for r in rows_raw:
        key = (r["dataset"], int(r["threads"]))
        index[key][r["tool"]] = float(r["walltime_s"])

    out_rows = []
    for ds in DS_ORDER:
        for t in THREAD_ORDER:
            key = (ds, t)
            if key not in index:
                continue
            d = index[key]
            og  = d.get("opengene")
            cpu = d.get("d0bromir_cpu")
            gpu = d.get("d0bromir_gpu")
            if og is None:
                continue
            row = {
                "dataset":           ds,
                "threads":           t,
                "opengene_s":        f"{og:.3f}",
                "d0bromir_cpu_s":    f"{cpu:.3f}" if cpu is not None else "",
                "d0bromir_gpu_s":    f"{gpu:.3f}" if gpu is not None else "",
                "cpu_speedup":       speedup_str(og, cpu) if cpu else "",
                "gpu_speedup":       speedup_str(og, gpu) if gpu else "",
            }
            out_rows.append(row)

    fields = ["dataset", "threads", "opengene_s", "d0bromir_cpu_s", "d0bromir_gpu_s",
              "cpu_speedup", "gpu_speedup"]
    write_csv(outdir / "table2_parallel_compression.csv", fields, out_rows)


# ══════════════════════════════════════════════════════════════════════════════
# Table 5 — Correctness verification
# Source: full_benchmark_20260417.csv (opengene rows → canonical read counts)
# Paper section: 7.5
# ══════════════════════════════════════════════════════════════════════════════

def make_table5(outdir: Path) -> None:
    """
    For each dataset: total_reads, passed_filter, pass_rate, thread counts tested,
    and match result (PASS/FAIL).  All rows in the benchmark must have identical
    reads_in/reads_out; any mismatch would set result=FAIL.
    """
    rows_raw = load_csv(CONSOLIDATED_CSV)

    # Group by dataset → collect all (reads_in, reads_out) values seen
    by_ds: dict[str, dict] = defaultdict(lambda: {
        "type": "", "reads_in_set": set(), "reads_out_set": set(),
        "threads_tested": set()
    })
    for r in rows_raw:
        ds = r["dataset"]
        by_ds[ds]["type"] = r["type"]
        by_ds[ds]["reads_in_set"].add(int(r["reads_in"]))
        by_ds[ds]["reads_out_set"].add(int(r["reads_out"]))
        by_ds[ds]["threads_tested"].add(int(r["threads"]))

    out_rows = []
    for ds in DS_ORDER:
        d = by_ds[ds]
        # Canonical values (should all be identical across tools/threads)
        reads_in_vals  = d["reads_in_set"]
        reads_out_vals = d["reads_out_set"]
        result = "PASS" if len(reads_in_vals) == 1 and len(reads_out_vals) == 1 else "FAIL"
        reads_in  = sorted(reads_in_vals)[0]
        reads_out = sorted(reads_out_vals)[0]
        threads_str = ",".join(str(t) for t in sorted(d["threads_tested"]))
        pass_rate = reads_out / reads_in * 100
        out_rows.append({
            "dataset":         ds,
            "type":            d["type"],
            "total_reads":     reads_in,
            "passed_filter":   reads_out,
            "pass_rate_pct":   f"{pass_rate:.2f}%",
            "threads_tested":  threads_str,
            "result":          result,
        })

    fields = ["dataset", "type", "total_reads", "passed_filter",
              "pass_rate_pct", "threads_tested", "result"]
    write_csv(outdir / "table5_correctness.csv", fields, out_rows)


# ══════════════════════════════════════════════════════════════════════════════
# Table 6 — Filtering-stage wall time: CPU vs GPU kernel
# Source: docs/publication/figures/fig2_kernel_speedup.csv
# Paper section: 7.3
# ══════════════════════════════════════════════════════════════════════════════

def make_table6(outdir: Path) -> None:
    """
    Filtering-stage time (ms) for d0bromir_cpu and d0bromir_gpu, measured via
    FASTP_PROFILING cpu_filter_ms counter.  Data collected from profiling logs
    and pre-processed into fig2_kernel_speedup.csv.
    """
    rows_raw = load_csv(FIG2_CSV)

    # DS × threads ordering
    ds_order_map = {ds: i for i, ds in enumerate(DS_ORDER)}
    rows_raw.sort(key=lambda r: (ds_order_map.get(r["dataset"], 99), int(r["threads"])))

    out_rows = []
    for r in rows_raw:
        cpu_ms = float(r["d0bromir_cpu_filtering_ms"])
        gpu_ms = float(r["d0bromir_gpu_filtering_ms"])
        out_rows.append({
            "dataset":        r["dataset"],
            "threads":        int(r["threads"]),
            "cpu_filter_ms":  int(cpu_ms),
            "gpu_filter_ms":  int(gpu_ms),
            "speedup":        speedup_str(cpu_ms, gpu_ms),
        })

    fields = ["dataset", "threads", "cpu_filter_ms", "gpu_filter_ms", "speedup"]
    write_csv(outdir / "table6_kernel_filtering.csv", fields, out_rows)


# ══════════════════════════════════════════════════════════════════════════════
# Table 7 — OpenGene v1.1.0 vs. d0bromir CPU wall-clock time and speedup
# Source: full_benchmark_20260417.csv (opengene + d0bromir_cpu rows)
# Paper section: 7.4
# ══════════════════════════════════════════════════════════════════════════════

def make_table7(outdir: Path) -> None:
    """
    End-to-end speedup of d0bromir CPU over OpenGene with gzip output enabled,
    across all datasets and thread counts.  Combined effect of speculative
    statistics, parallel output compression, and enlarged I/O buffers.
    Note: T=32 rows for WGS_PE_18.2G and WGS_PE_40G are absent from the
    benchmark CSV (those values are extrapolated in the paper with a † note).
    """
    rows_raw = load_csv(CONSOLIDATED_CSV)

    index: dict[tuple, dict] = defaultdict(dict)
    for r in rows_raw:
        key = (r["dataset"], int(r["threads"]))
        index[key][r["tool"]] = float(r["walltime_s"])

    out_rows = []
    for ds in DS_ORDER:
        for t in THREAD_ORDER:
            key = (ds, t)
            if key not in index:
                continue
            d = index[key]
            og  = d.get("opengene")
            cpu = d.get("d0bromir_cpu")
            if og is None or cpu is None:
                continue
            out_rows.append({
                "dataset":        ds,
                "threads":        t,
                "opengene_s":     f"{og:.3f}",
                "d0bromir_cpu_s": f"{cpu:.3f}",
                "speedup":        speedup_str(og, cpu),
            })

    fields = ["dataset", "threads", "opengene_s", "d0bromir_cpu_s", "speedup"]
    write_csv(outdir / "table7_og_vs_cpu.csv", fields, out_rows)


# ══════════════════════════════════════════════════════════════════════════════
# Table 8 — GPU overhead vs CPU
# Source: docs/publication/figures/fig4_transfer_overhead.csv
# Paper section: 8.2
# ══════════════════════════════════════════════════════════════════════════════

def make_table8(outdir: Path) -> None:
    """
    Absolute and relative GPU overhead (GPU_s − CPU_s) by dataset and thread
    count, derived from wall-clock times in full_benchmark_20260417.csv and
    pre-computed into fig4_transfer_overhead.csv.
    """
    rows_raw = load_csv(FIG4_CSV)

    ds_order_map = {ds: i for i, ds in enumerate(DS_ORDER)}
    rows_raw.sort(key=lambda r: (ds_order_map.get(r["dataset"], 99), int(r["threads"])))

    out_rows = []
    for r in rows_raw:
        overhead_s   = float(r["gpu_overhead_vs_cpu_s"])
        overhead_pct = r["gpu_overhead_vs_cpu_pct"].rstrip("%")
        sign_s   = "+" if overhead_s   >= 0 else ""
        sign_pct = "+" if float(overhead_pct) >= 0 else ""
        out_rows.append({
            "dataset":              r["dataset"],
            "threads":              int(r["threads"]),
            "d0bromir_cpu_s":       float(r["d0bromir_cpu_s"]),
            "d0bromir_gpu_s":       float(r["d0bromir_gpu_s"]),
            "overhead_s":           f"{sign_s}{overhead_s:.3f}",
            "overhead_pct":         f"{sign_pct}{float(overhead_pct):.1f}%",
        })

    fields = ["dataset", "threads", "d0bromir_cpu_s", "d0bromir_gpu_s",
              "overhead_s", "overhead_pct"]
    write_csv(outdir / "table8_gpu_overhead.csv", fields, out_rows)





# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default=str(SCRIPT_DIR),
                        help="Directory for output CSVs (default: same dir as this script)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Verify required inputs exist
    missing = [p for p in [CONSOLIDATED_CSV, FIG2_CSV, FIG4_CSV] if not p.exists()]
    if missing:
        print("ERROR: missing source files:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    print(f"Output directory: {outdir}")
    print()

    make_table1(outdir)
    make_table2(outdir)
    make_table3(outdir)
    make_table5(outdir)
    make_table6(outdir)
    make_table7(outdir)
    make_table8(outdir)

    print()
    print("All tables extracted.")


if __name__ == "__main__":
    main()
