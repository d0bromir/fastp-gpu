#!/usr/bin/env python3
"""
Merge rerun_timeouts.csv into
fastp-gpu_v1.2.2/vs_opengene_v1.3.3/galaxy_arm_a100/20260426_114535/full_benchmark_20260426_114535.csv:
- Replace the 9 timeout rows' walltime_s, reads_in, reads_out with the rerun values.
- Recompute pct_vs_opengene for ALL rows (so any (dataset, type, threads, rep)
  group whose opengene baseline changed is updated consistently).

Formula: pct = (walltime - opengene_walltime_for_same_dataset_type_threads_rep) / opengene * 100
Format: signed, 1 decimal, '%' suffix (e.g. '-23.6%', '+0.1%').
opengene rows keep pct_vs_opengene = 'baseline'.
validation column is preserved as-is.
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import os
ROOT = Path(os.environ.get(
    'FASTP_D0BROMIR_ROOT',
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)) / "benchmark_results"
MAIN = ROOT / "fastp-gpu_v1.2.2" / "vs_opengene_v1.3.3" / "galaxy_arm_a100" / "20260426_114535" / "full_benchmark_20260426_114535.csv"
RERUN = ROOT / "rerun_timeouts.csv"
BACKUP = MAIN.with_suffix(".csv.bak")

def main() -> int:
    if not MAIN.exists():
        print(f"missing: {MAIN}", file=sys.stderr); return 1
    if not RERUN.exists():
        print(f"missing: {RERUN}", file=sys.stderr); return 1

    # Load rerun results keyed by (dataset, type, tool, threads, rep)
    rerun: dict[tuple, dict] = {}
    with RERUN.open() as f:
        for row in csv.DictReader(f):
            key = (row["dataset"], row["type"], row["tool"], row["threads"], row["rep"])
            rerun[key] = row
    print(f"loaded {len(rerun)} rerun rows")

    # Load main CSV
    with MAIN.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not fieldnames:
        print("no header", file=sys.stderr); return 1

    # Apply rerun overrides. Skip rerun rows where reads_in == 0 (DNF / killed).
    applied = 0
    skipped_dnf: list[tuple] = []
    for r in rows:
        key = (r["dataset"], r["type"], r["tool"], r["threads"], r["rep"])
        if key in rerun:
            new = rerun[key]
            try:
                if int(new.get("reads_in", "0") or "0") == 0:
                    skipped_dnf.append(key)
                    continue
            except ValueError:
                skipped_dnf.append(key)
                continue
            r["walltime_s"] = new["walltime_s"]
            r["reads_in"] = new["reads_in"]
            r["reads_out"] = new["reads_out"]
            applied += 1
    print(f"applied {applied} rerun overrides (rerun rows={len(rerun)}, DNF skipped={len(skipped_dnf)})")
    for k in skipped_dnf:
        print(f"  DNF (kept original): {k}")
    if applied + len(skipped_dnf) != len(rerun):
        print("WARNING: not all rerun rows matched", file=sys.stderr)

    # Build opengene baseline lookup (dataset, type, threads, rep) -> walltime
    baselines: dict[tuple, float] = {}
    for r in rows:
        if r["tool"] == "opengene":
            try:
                baselines[(r["dataset"], r["type"], r["threads"], r["rep"])] = float(r["walltime_s"])
            except ValueError:
                pass

    # Recompute pct_vs_opengene for all rows
    for r in rows:
        if r["tool"] == "opengene":
            r["pct_vs_opengene"] = "baseline"
            continue
        key = (r["dataset"], r["type"], r["threads"], r["rep"])
        base = baselines.get(key)
        try:
            wall = float(r["walltime_s"])
        except ValueError:
            wall = None
        if base is None or base == 0 or wall is None:
            r["pct_vs_opengene"] = "N/A"
        else:
            pct = (wall - base) / base * 100.0
            r["pct_vs_opengene"] = f"{pct:+.1f}%"

    # Backup and write
    if not BACKUP.exists():
        BACKUP.write_bytes(MAIN.read_bytes())
        print(f"backup written: {BACKUP}")
    with MAIN.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"updated {MAIN}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
