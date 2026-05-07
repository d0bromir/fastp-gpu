#!/usr/bin/env python3
"""
apply_t32_results.py — Merge real T=32 measurements into the canonical
benchmark CSV (full_benchmark_20260417.csv),
figure source CSVs, and the LaTeX paper, replacing the Amdahl extrapolations.

Usage:
    python3 scripts/apply_t32_results.py benchmark_results/<run_dir>/full_benchmark_<TS>.csv

What it updates:
  1. benchmark_results/fastp-gpu_v1.2.2/vs_opengene_v1.1.0/galaxy_arm_a100/20260417/full_benchmark_20260417.csv
       - Removes old extrapolated rows (WGS_PE_18.2G T=32 and WGS_PE_40G T=32)
       - Appends the 6 new measured rows (with pct_vs_opengene computed)
  2. docs/publication/figures/fig3_speculative_stats.csv
       - Replaces T=32 opengene_s / d0bromir_cpu_s / cpu_speedup_x
  3. docs/publication/figures/fig5_thread_scaling.csv
       - Replaces T=32 walltime_s for all 3 tools (both datasets)
  4. docs/publication/GPU_ACCELERATED_FASTP_PAPER.tex
       - Removes the dagger (†) footnote from the two T=32 rows in Table 7
       - Updates the numeric values in those two rows
  5. Regenerates table7_og_vs_cpu.csv via extract_tables.py
  6. Regenerates all figure PNG/PDF via the plot_*.py scripts
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONSOL_CSV = REPO / "benchmark_results" / "fastp-gpu_v1.2.2" / "vs_opengene_v1.1.0" / "galaxy_arm_a100" / "20260417" / "full_benchmark_20260417.csv"
FIG3_CSV   = REPO / "docs" / "publication" / "figures" / "fig3_speculative_stats.csv"
FIG5_CSV   = REPO / "docs" / "publication" / "figures" / "fig5_thread_scaling.csv"
TEX_FILE   = REPO / "docs" / "publication" / "GPU_ACCELERATED_FASTP_PAPER.tex"
TABLES_DIR = REPO / "docs" / "publication" / "tables"
FIGS_DIR   = REPO / "docs" / "publication" / "figures"


def load_csv(p: Path) -> list[dict]:
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(p: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {p.relative_to(REPO)}  ({len(rows)} rows)")


def speedup_pct(og: float, tool: float) -> str:
    if og == 0:
        return ""
    pct = (tool - og) / og * 100
    return f"{pct:+.1f}%"


# ── 1. Update canonical benchmark CSV (full_benchmark_20260417.csv) ────────────────

def update_consolidated(new_rows: list[dict]) -> dict:
    """Remove old T=32 rows for WGS_PE_18.2G and WGS_PE_40G, append new ones.
    Returns {(dataset, tool): walltime_s} for the new rows."""
    TARGET_DS = {"WGS_PE_18.2G", "WGS_PE_40G"}
    TARGET_T  = 32

    existing = load_csv(CONSOL_CSV)
    fields = list(existing[0].keys()) if existing else [
        "dataset","type","tool","threads","walltime_s","reads_in","reads_out",
        "pct_vs_opengene","fastq_files"
    ]

    # Remove old extrapolated rows
    kept = [r for r in existing
            if not (r["dataset"] in TARGET_DS and int(r["threads"]) == TARGET_T)]
    removed = len(existing) - len(kept)
    print(f"  removed {removed} extrapolated T=32 rows from full_benchmark_20260417.csv")

    # Build opengene wall times (needed to compute pct_vs_opengene)
    og_times: dict[str, float] = {}
    for r in new_rows:
        if r["tool"] == "opengene":
            og_times[r["dataset"]] = float(r["walltime_s"])

    # Prepare new rows
    wall_map = {}
    append_rows = []
    for r in new_rows:
        og = og_times.get(r["dataset"], 0)
        wall = float(r["walltime_s"])
        pct = speedup_pct(og, wall) if r["tool"] != "opengene" else "baseline"
        row = {
            "dataset":          r["dataset"],
            "type":             r.get("type", "PE"),
            "tool":             r["tool"],
            "threads":          int(r["threads"]),
            "walltime_s":       f"{wall:.3f}",
            "reads_in":         r["reads_in"],
            "reads_out":        r["reads_out"],
            "pct_vs_opengene":  pct,
            "fastq_files":      r.get("fastq_files", ""),
        }
        append_rows.append(row)
        wall_map[(r["dataset"], r["tool"])] = wall

    write_csv(CONSOL_CSV, fields, kept + append_rows)
    print(f"  added {len(append_rows)} real T=32 rows")
    return wall_map


# ── 2. Update fig3_speculative_stats.csv ─────────────────────────────────────

def update_fig3(wall_map: dict) -> None:
    rows = load_csv(FIG3_CSV)
    fields = list(rows[0].keys())
    TARGET_DS = {"WGS_PE_18.2G", "WGS_PE_40G"}
    TARGET_T  = 32

    updated = 0
    for r in rows:
        if r["dataset"] in TARGET_DS and int(r["threads"]) == TARGET_T:
            og  = wall_map.get((r["dataset"], "opengene"))
            cpu = wall_map.get((r["dataset"], "d0bromir_cpu"))
            if og is not None:
                r["opengene_s"] = f"{og:.3f}"
            if cpu is not None:
                r["d0bromir_cpu_s"] = f"{cpu:.3f}"
            if og and cpu:
                spd_x = og / cpu
                spd_pct = (og - cpu) / og * 100
                r["cpu_speedup_x"]   = f"{spd_x:.2f}x"
                r["cpu_speedup_pct"] = f"{spd_pct:.1f}%"
            updated += 1

    write_csv(FIG3_CSV, fields, rows)
    print(f"  updated {updated} T=32 rows in fig3_speculative_stats.csv")


# ── 3. Update fig5_thread_scaling.csv ────────────────────────────────────────

def update_fig5(wall_map: dict) -> None:
    rows = load_csv(FIG5_CSV)
    fields = list(rows[0].keys())
    TARGET_DS = {"WGS_PE_18.2G", "WGS_PE_40G"}
    TARGET_T  = 32
    TOOLS     = {"opengene", "d0bromir_cpu", "d0bromir_gpu"}

    updated = 0
    for r in rows:
        if r["dataset"] in TARGET_DS and int(r["threads"]) == TARGET_T and r["tool"] in TOOLS:
            new_wall = wall_map.get((r["dataset"], r["tool"]))
            if new_wall is not None:
                r["walltime_s"] = f"{new_wall:.3f}"

                # Recompute pct_vs_opengene
                og_wall = wall_map.get((r["dataset"], "opengene"))
                if og_wall:
                    if r["tool"] == "opengene":
                        r["pct_vs_opengene"] = "baseline"
                    else:
                        r["pct_vs_opengene"] = speedup_pct(og_wall, new_wall)
                updated += 1

    write_csv(FIG5_CSV, fields, rows)
    print(f"  updated {updated} T=32 rows in fig5_thread_scaling.csv")


# ── 4. Update LaTeX paper ─────────────────────────────────────────────────────

def update_tex(wall_map: dict) -> None:
    """
    In Table 7, the two dagger rows look like:
      WGS PE (18.2 GB) & 32† & 398.5 & 103.0 & 3.87× \\
      WGS PE (40 GB) & 32† & 1098.3 & 220.0 & 4.99× \\

    Replace values and remove the † from the thread count.
    Also remove the dagger footnote line if both rows are now real.
    """
    tex = TEX_FILE.read_text()

    updates = [
        # (dataset label in tex, dataset key in wall_map)
        ("WGS PE (18.2 GB)", "WGS_PE_18.2G"),
        ("WGS PE (40 GB)",   "WGS_PE_40G"),
    ]

    for tex_label, ds_key in updates:
        og  = wall_map.get((ds_key, "opengene"))
        cpu = wall_map.get((ds_key, "d0bromir_cpu"))
        if og is None or cpu is None:
            print(f"  WARNING: no wall times for {ds_key} — skipping tex update")
            continue

        spd = og / cpu
        # Match the row: "WGS PE (18.2 GB) & 32† & ... & ... & ... \\"
        # The values could be integers or decimals
        pattern = (
            r'(' + re.escape(tex_label) + r'\s*&\s*32)[†\u2020]'
            r'(\s*&\s*)[\d,.]+'      # opengene_s
            r'(\s*&\s*)[\d,.]+'      # cpu_s
            r'(\s*&\s*)[\d.]+×\s*(\\\\)'
        )
        replacement = (
            r'\g<1>'                            # label & 32 (no dagger)
            + r'\g<2>' + f"{og:.1f}"            # opengene_s
            + r'\g<3>' + f"{cpu:.1f}"           # cpu_s
            + r'\g<4>' + f"{spd:.2f}" + r'× \5' # speedup
        )
        new_tex, n = re.subn(pattern, replacement, tex)
        if n == 0:
            # Try alternate pattern without × (in case it uses \times or ×)
            pattern2 = (
                r'(' + re.escape(tex_label) + r'\s*&\s*32)[†\u2020]'
                r'(\s*&\s*)[\d,.]+'
                r'(\s*&\s*)[\d,.]+'
                r'(\s*&\s*)[\d.]+[×\u00d7][^\\\n]*?(\\\\)'
            )
            new_tex, n = re.subn(pattern2, replacement, tex)

        if n > 0:
            tex = new_tex
            print(f"  updated {tex_label} T=32 row in .tex (og={og:.1f}s cpu={cpu:.1f}s spd={spd:.2f}×)")
        else:
            print(f"  WARNING: could not find/replace {tex_label} T=32 row in .tex")
            print(f"           Search for '32†' near '{tex_label}' manually")

    # Remove the dagger footnote line (it references the now-real measurements)
    dagger_patterns = [
        r'\n†T=32 values for WGS PE 18\.2G and WGS PE 40G are extrapolated.*?\n',
        r'\n\\textsuperscript\{†\}.*?extrapolated.*?\n',
        r'\n†.*?Amdahl.*?\n',
    ]
    for pat in dagger_patterns:
        new_tex, n = re.subn(pat, '\n', tex, flags=re.IGNORECASE | re.DOTALL)
        if n > 0:
            tex = new_tex
            print(f"  removed dagger footnote from .tex")
            break

    TEX_FILE.write_text(tex)
    print(f"  wrote {TEX_FILE.relative_to(REPO)}")


# ── 5. Regenerate table CSVs ─────────────────────────────────────────────────

def regen_tables() -> None:
    print("\nRegenerating table CSVs...")
    r = subprocess.run(
        [sys.executable, str(TABLES_DIR / "extract_tables.py")],
        cwd=str(REPO), capture_output=True, text=True
    )
    if r.returncode != 0:
        print("  ERROR:", r.stderr)
    else:
        print(r.stdout.strip())


# ── 6. Regenerate figures ─────────────────────────────────────────────────────

def regen_figures() -> None:
    print("\nRegenerating figures...")
    for script in sorted(FIGS_DIR.glob("plot_fig*.py")):
        r = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(FIGS_DIR), capture_output=True, text=True
        )
        status = "OK" if r.returncode == 0 else "FAIL"
        print(f"  {script.name}: {status}")
        if r.returncode != 0:
            print("   ", r.stderr.strip())


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("t32_csv", help="Path to the t32_bench_<timestamp>.csv output file")
    parser.add_argument("--no-regen", action="store_true",
                        help="Skip regenerating tables and figures")
    args = parser.parse_args()

    t32_path = Path(args.t32_csv)
    if not t32_path.exists():
        print(f"ERROR: {t32_path} not found", file=sys.stderr)
        sys.exit(1)

    new_rows = load_csv(t32_path)
    # Strip the header row if it accidentally ended up in data
    new_rows = [r for r in new_rows if r["dataset"] not in ("dataset",)]

    expected = {("WGS_PE_18.2G", "opengene"), ("WGS_PE_18.2G", "d0bromir_cpu"), ("WGS_PE_18.2G", "d0bromir_gpu"),
                ("WGS_PE_40G",   "opengene"), ("WGS_PE_40G",   "d0bromir_cpu"), ("WGS_PE_40G",   "d0bromir_gpu")}
    found = {(r["dataset"], r["tool"]) for r in new_rows}
    missing = expected - found
    if missing:
        print(f"WARNING: missing rows in t32 CSV: {missing}")

    print(f"\nLoaded {len(new_rows)} rows from {t32_path.name}")
    for r in new_rows:
        print(f"  {r['dataset']} {r['tool']} T={r['threads']} -> {r['walltime_s']}s")

    print("\n── 1. full_benchmark_20260417.csv ────────────────────────────────────")
    wall_map = update_consolidated(new_rows)

    print("\n── 2. fig3_speculative_stats.csv ─────────────────────────────────────")
    update_fig3(wall_map)

    print("\n── 3. fig5_thread_scaling.csv ────────────────────────────────────────")
    update_fig5(wall_map)

    print("\n── 4. GPU_ACCELERATED_FASTP_PAPER.tex ───────────────────────────────")
    update_tex(wall_map)

    if not args.no_regen:
        regen_tables()
        regen_figures()

    print("\nDone. Run verify_tables.py to confirm correctness.")
    print("Then run scripts/build_paper.sh to rebuild the PDF.")


if __name__ == "__main__":
    main()
