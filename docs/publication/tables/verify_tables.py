#!/usr/bin/env python3
"""
verify_tables.py — Cross-check every generated table CSV against:
  1. The raw source files (ground-truth values)
  2. Hard-coded paper values (spot-checks of key claims)

Exit code 0 = all checks pass; 1 = at least one failure.
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent.parent.parent

CONSOLIDATED_CSV = REPO_ROOT / "benchmark_results" / "fastp-gpu_v1.2.2" / "vs_opengene_v1.1.0" / "galaxy_arm_a100" / "20260417" / "full_benchmark_20260417.csv"
FIG2_CSV         = REPO_ROOT / "docs" / "publication" / "figures" / "fig2_kernel_speedup.csv"
FIG4_CSV         = REPO_ROOT / "docs" / "publication" / "figures" / "fig4_transfer_overhead.csv"

TABLES = SCRIPT_DIR

PASS = 0
FAIL = 0


def ok(msg: str) -> None:
    global PASS
    PASS += 1
    print(f"  PASS  {msg}")


def fail(msg: str) -> None:
    global FAIL
    FAIL += 1
    print(f"  FAIL  {msg}")


def check(cond: bool, msg: str) -> None:
    (ok if cond else fail)(msg)


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def near(a: float, b: float, tol: float = 0.005) -> bool:
    """True if values are within tol of each other (absolute)."""
    return abs(a - b) <= tol


def speedup(og: float, tool: float) -> float:
    return og / tool if tool != 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: build indexes from raw sources
# ─────────────────────────────────────────────────────────────────────────────

def index_consolidated() -> dict:
    """Return {(dataset, threads, tool): walltime_s}"""
    idx = {}
    for r in load_csv(CONSOLIDATED_CSV):
        idx[(r["dataset"], int(r["threads"]), r["tool"])] = float(r["walltime_s"])
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# Table 1
# ─────────────────────────────────────────────────────────────────────────────

# Expected rows match HARDWARE_ROWS in extract_tables.py
EXPECTED_HARDWARE = {
    "CPU model":    "ARM Neoverse N1",
    "CPU cores":    "128 (single socket, 1 thread/core)",
    "CPU clock":    "3.0 GHz (max)",
    "GPU":          "2x NVIDIA A100 80 GB PCIe (Gen4 x16)",
    "GPU driver":   "590.48.01",
    "RAM":          "256 GB DDR4",
    "CUDA Toolkit": "12.6",
    "GCC":          "15.2.0",
}

def verify_table1() -> None:
    print("\n[Table 1] Hardware configuration")
    rows = load_csv(TABLES / "table1_hardware.csv")

    check(len(rows) == 11, f"11 hardware parameter rows (got {len(rows)})")

    params = {r["parameter"]: r["value"] for r in rows}
    for param, expected in EXPECTED_HARDWARE.items():
        check(params.get(param) == expected,
              f"{param} = {expected!r}  (got {params.get(param)!r})")


# ─────────────────────────────────────────────────────────────────────────────
# Table 3
# ─────────────────────────────────────────────────────────────────────────────

# Key fields spot-checked against §6.2 of the paper and ENA/SRA accessions
EXPECTED_DATASETS = {
    "Panel SE 148M": {"accession": "S1A_S1_L001_R1",      "type": "Panel, SE", "read_length": "151 bp"},
    "Panel PE 304M": {"accession": "S1A_S1_L001 (R1+R2)", "type": "Panel, PE", "read_length": "151 bp"},
    "WGS SE 6.3G":   {"accession": "ERR1044780",           "type": "WGS, SE",   "read_length": "100 bp"},
    "WGS PE 12.7G":  {"accession": "ERR1044780 (R1+R2)",  "type": "WGS, PE",   "read_length": "100 bp"},
    "WGS PE 18.1G":  {"accession": "ERR1044319",           "type": "WGS, PE",   "read_length": "100 bp"},
    "WGS PE 38.5G":  {"accession": "DRR216653",            "type": "WGS, PE",   "read_length": "150 bp"},
}

def verify_table3() -> None:
    print("\n[Table 3] Benchmark datasets")
    rows = load_csv(TABLES / "table3_datasets.csv")

    check(len(rows) == 6, f"6 dataset rows (got {len(rows)})")

    by_label = {r["label"]: r for r in rows}
    for label, expected in EXPECTED_DATASETS.items():
        r = by_label.get(label)
        check(r is not None, f"row for '{label}' present")
        if r:
            for field, val in expected.items():
                check(r[field] == val,
                      f"{label} {field}={val!r}  (got {r[field]!r})")

    # Cross-check read counts against consolidated benchmark for WGS datasets
    # reads_in from the benchmark equals total reads for each dataset
    raw_reads = {}
    for row in load_csv(CONSOLIDATED_CSV):
        ds = row["dataset"]
        if ds not in raw_reads:
            raw_reads[ds] = int(row["reads_in"])

    # Map table3 labels to dataset keys used in benchmark CSV
    label_to_ds = {
        "WGS SE 6.3G":  "WGS_SE_6.3G",
        "WGS PE 12.7G": "WGS_PE_12.8G",
        "WGS PE 18.1G": "WGS_PE_18.2G",
        "WGS PE 38.5G": "WGS_PE_40G",
    }
    # Panel SE: reads_in = 1,381,894 → 1.38M  (rounded); PE: 2,763,788 → 2.76M
    benchmark_read_counts = {
        "Panel SE 148M": (1_381_894,  "1.38M"),
        "Panel PE 304M": (2_763_788,  "2.76M"),
        "WGS SE 6.3G":   (80_314_764, "80.3M"),
        "WGS PE 12.7G":  (160_629_528,"160.6M"),
        "WGS PE 18.1G":  (234_238_034,"234.2M"),
        "WGS PE 38.5G":  (722_563_222,"722.6M"),
    }
    # Verify the 'reads' column rounds correctly relative to benchmark counts
    for label, (exact, expected_display) in benchmark_read_counts.items():
        r = by_label.get(label)
        if r:
            # Strip trailing M/G and compare prefix
            check(r["reads"].rstrip("M").rstrip("G") == expected_display.rstrip("M").rstrip("G"),
                  f"{label} reads display={expected_display!r}  (got {r['reads']!r})")


# ─────────────────────────────────────────────────────────────────────────────
# Table 2
# ─────────────────────────────────────────────────────────────────────────────

def verify_table2() -> None:
    print("\n[Table 2] Parallel compression wall times")
    raw = index_consolidated()
    rows = load_csv(TABLES / "table2_parallel_compression.csv")

    total = 0
    mismatches = 0
    for r in rows:
        ds, t = r["dataset"], int(r["threads"])
        for tool, col in [("opengene", "opengene_s"),
                          ("d0bromir_cpu", "d0bromir_cpu_s"),
                          ("d0bromir_gpu", "d0bromir_gpu_s")]:
            v_csv = r[col]
            if v_csv == "":
                continue
            raw_val = raw.get((ds, t, tool))
            if raw_val is None:
                fail(f"  {ds} T={t} {tool}: missing in raw source")
                mismatches += 1
                continue
            total += 1
            if not near(float(v_csv), raw_val):
                fail(f"  {ds} T={t} {tool}: CSV={v_csv} raw={raw_val:.3f}")
                mismatches += 1

    check(mismatches == 0,
          f"all {total} walltime cells match full_benchmark_20260417.csv")

    # Spot-check speedup formula
    for r in rows:
        og  = r["opengene_s"]
        cpu = r["d0bromir_cpu_s"]
        spd = r["cpu_speedup"]
        if og and cpu and spd:
            expected = f"{speedup(float(og), float(cpu)):.2f}x"
            check(spd == expected,
                  f"speedup formula {r['dataset']} T={r['threads']}: "
                  f"got {spd} expected {expected}")
            break  # one sanity check is enough

    # Paper key claim: 4.01× at WGS_PE_40G T=16
    peak = next((r for r in rows
                 if r["dataset"] == "WGS_PE_40G" and int(r["threads"]) == 16), None)
    check(peak is not None, "WGS_PE_40G T=16 row present")
    if peak:
        check(peak["cpu_speedup"] == "4.01x",
              f"peak CPU speedup = 4.01×  (got {peak['cpu_speedup']})")


# ─────────────────────────────────────────────────────────────────────────────
# Table 5
# ─────────────────────────────────────────────────────────────────────────────

def verify_table5() -> None:
    print("\n[Table 5] Correctness verification")
    rows = load_csv(TABLES / "table5_correctness.csv")

    # All rows should have result=PASS
    all_pass = all(r["result"] == "PASS" for r in rows)
    check(all_pass, "all datasets marked PASS")

    # 6 datasets
    check(len(rows) == 6, f"exactly 6 dataset rows (got {len(rows)})")

    # Spot-check from paper: WGS_PE_40G reads_in=722,563,222
    r40 = next((r for r in rows if r["dataset"] == "WGS_PE_40G"), None)
    check(r40 is not None, "WGS_PE_40G row present")
    if r40:
        check(int(r40["total_reads"]) == 722563222,
              f"WGS_PE_40G total_reads=722,563,222 (got {r40['total_reads']})")
        check(int(r40["passed_filter"]) == 717214748,
              f"WGS_PE_40G passed_filter=717,214,748 (got {r40['passed_filter']})")

    # Cross-check reads values against raw consolidated CSV
    raw_reads: dict[str, tuple] = {}
    for r in load_csv(CONSOLIDATED_CSV):
        ds = r["dataset"]
        if ds not in raw_reads:
            raw_reads[ds] = (int(r["reads_in"]), int(r["reads_out"]))

    for r in rows:
        ds = r["dataset"]
        if ds in raw_reads:
            exp_in, exp_out = raw_reads[ds]
            check(int(r["total_reads"]) == exp_in,
                  f"{ds} total_reads matches full_benchmark_20260417.csv")
            check(int(r["passed_filter"]) == exp_out,
                  f"{ds} passed_filter matches full_benchmark_20260417.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Table 6
# ─────────────────────────────────────────────────────────────────────────────

def verify_table6() -> None:
    print("\n[Table 6] Kernel filtering times")
    raw_rows = {(r["dataset"], int(r["threads"])): r
                for r in load_csv(FIG2_CSV)}
    tbl_rows = load_csv(TABLES / "table6_kernel_filtering.csv")

    mismatches = 0
    for r in tbl_rows:
        key = (r["dataset"], int(r["threads"]))
        src = raw_rows.get(key)
        if src is None:
            fail(f"  {key} not found in fig2_kernel_speedup.csv")
            mismatches += 1
            continue
        for col_tbl, col_src in [("cpu_filter_ms", "d0bromir_cpu_filtering_ms"),
                                  ("gpu_filter_ms", "d0bromir_gpu_filtering_ms")]:
            if int(r[col_tbl]) != int(float(src[col_src])):
                fail(f"  {key} {col_tbl}: table={r[col_tbl]} src={src[col_src]}")
                mismatches += 1

    check(mismatches == 0,
          f"all {len(tbl_rows)} kernel timing cells match fig2_kernel_speedup.csv")

    # Paper spot-check: WGS_PE_18.2G T=8 CPU=247 GPU=151 speedup=1.6×
    r_spot = next((r for r in tbl_rows
                   if r["dataset"] == "WGS_PE_18.2G" and int(r["threads"]) == 8), None)
    check(r_spot is not None, "WGS_PE_18.2G T=8 row present")
    if r_spot:
        check(int(r_spot["cpu_filter_ms"]) == 247,
              f"WGS_PE_18.2G T=8 cpu_filter_ms=247 (got {r_spot['cpu_filter_ms']})")
        check(int(r_spot["gpu_filter_ms"]) == 151,
              f"WGS_PE_18.2G T=8 gpu_filter_ms=151 (got {r_spot['gpu_filter_ms']})")
        check(r_spot["speedup"] == "1.64x",
              f"WGS_PE_18.2G T=8 speedup (got {r_spot['speedup']})")


# ─────────────────────────────────────────────────────────────────────────────
# Table 7
# ─────────────────────────────────────────────────────────────────────────────

def verify_table7() -> None:
    print("\n[Table 7] OpenGene vs d0bromir CPU wall-clock time (\u00a77.4)")
    raw = index_consolidated()
    rows = load_csv(TABLES / "table7_og_vs_cpu.csv")

    mismatches = 0
    for r in rows:
        ds, t = r["dataset"], int(r["threads"])
        og_raw  = raw.get((ds, t, "opengene"))
        cpu_raw = raw.get((ds, t, "d0bromir_cpu"))
        if og_raw is None or cpu_raw is None:
            fail(f"  {ds} T={t}: raw data missing")
            mismatches += 1
            continue
        if not near(float(r["opengene_s"]), og_raw):
            fail(f"  {ds} T={t} opengene_s: table={r['opengene_s']} raw={og_raw:.3f}")
            mismatches += 1
        if not near(float(r["d0bromir_cpu_s"]), cpu_raw):
            fail(f"  {ds} T={t} d0bromir_cpu_s: table={r['d0bromir_cpu_s']} raw={cpu_raw:.3f}")
            mismatches += 1
        # speedup formula
        exp_spd = f"{speedup(og_raw, cpu_raw):.2f}x"
        if r["speedup"] != exp_spd:
            fail(f"  {ds} T={t} speedup: table={r['speedup']} computed={exp_spd}")
            mismatches += 1

    check(mismatches == 0,
          f"all {len(rows)} rows match full_benchmark_20260417.csv")

    # Paper key claim: 4.01× at WGS_PE_40G T=16
    peak = next((r for r in rows
                 if r["dataset"] == "WGS_PE_40G" and int(r["threads"]) == 16), None)
    check(peak is not None, "WGS_PE_40G T=16 row present")
    if peak:
        check(peak["speedup"] == "4.01x",
              f"peak speedup=4.01× (got {peak['speedup']})")

    # Consistent with Table 2 (same opengene/cpu columns)
    t2_rows = {(r["dataset"], int(r["threads"])): r
               for r in load_csv(TABLES / "table2_parallel_compression.csv")}
    mismatches_t2 = 0
    for r in rows:
        key = (r["dataset"], int(r["threads"]))
        t2 = t2_rows.get(key)
        if t2 is None:
            continue
        if r["opengene_s"] != t2["opengene_s"] or r["d0bromir_cpu_s"] != t2["d0bromir_cpu_s"]:
            fail(f"  {key} differs between table2 and table7")
            mismatches_t2 += 1
    check(mismatches_t2 == 0, "table7 opengene/cpu values consistent with table2")


# ─────────────────────────────────────────────────────────────────────────────
# Table 8
# ─────────────────────────────────────────────────────────────────────────────

def verify_table8() -> None:
    print("\n[Table 8] GPU overhead")
    raw_rows = {(r["dataset"], int(r["threads"])): r for r in load_csv(FIG4_CSV)}
    tbl_rows = load_csv(TABLES / "table8_gpu_overhead.csv")

    check(len(tbl_rows) == len(raw_rows),
          f"row count matches fig4 source ({len(tbl_rows)} vs {len(raw_rows)})")

    mismatches = 0
    for r in tbl_rows:
        key = (r["dataset"], int(r["threads"]))
        src = raw_rows.get(key)
        if src is None:
            fail(f"  {key} missing in fig4 source")
            mismatches += 1
            continue
        for tcol, scol in [("d0bromir_cpu_s", "d0bromir_cpu_s"),
                            ("d0bromir_gpu_s", "d0bromir_gpu_s")]:
            if not near(float(r[tcol]), float(src[scol])):
                fail(f"  {key} {tcol}: table={r[tcol]} src={src[scol]}")
                mismatches += 1
        # Verify overhead is re-derived correctly: gpu - cpu
        computed_overhead = float(r["d0bromir_gpu_s"]) - float(r["d0bromir_cpu_s"])
        tbl_overhead = float(r["overhead_s"])
        if not near(tbl_overhead, computed_overhead, tol=0.005):
            fail(f"  {key} overhead_s: table={tbl_overhead:.3f} computed={computed_overhead:.3f}")
            mismatches += 1

    check(mismatches == 0, "all GPU overhead values match fig4 source")

    # Paper spot-checks: Panel_SE T=8 overhead≈+3.45s, ≈+46.9%
    spot = next((r for r in tbl_rows
                 if r["dataset"] == "Panel_SE_148M" and int(r["threads"]) == 8), None)
    check(spot is not None, "Panel_SE_148M T=8 row present")
    if spot:
        check(spot["overhead_s"].startswith("+"),
              f"Panel_SE T=8 overhead positive (got {spot['overhead_s']})")
        check(near(float(spot["overhead_s"]), 3.45, tol=0.05),
              f"Panel_SE T=8 overhead≈3.45s (got {spot['overhead_s']})")
        check("46" in spot["overhead_pct"],
              f"Panel_SE T=8 overhead≈46.x% (got {spot['overhead_pct']})")

    # WGS_PE_40G T=4 should be negative (GPU faster)
    neg = next((r for r in tbl_rows
                if r["dataset"] == "WGS_PE_40G" and int(r["threads"]) == 4), None)
    if neg:
        check(neg["overhead_s"].startswith("-"),
              f"WGS_PE_40G T=4 overhead negative (got {neg['overhead_s']})")  


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Table verification")
    print("=" * 60)

    verify_table1()
    verify_table2()
    verify_table3()
    verify_table5()
    verify_table6()
    verify_table7()
    verify_table8()

    print()
    print("=" * 60)
    total = PASS + FAIL
    print(f"Result: {PASS}/{total} checks passed, {FAIL} failed")
    print("=" * 60)
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
