# Benchmark results layout

Layout designed so new runs can be added without breaking analysis.

```
benchmark_results/
├── README.md                          # this file
├── _tools/                            # benchmark runner scripts
│   ├── run_benchmark.sh               # generic runner (Linux/aarch64)
│   └── run_benchmark_wsl.sh           # laptop/WSL runner
└── fastp-gpu_<release>/               # release of THIS tool (fastp-gpu)
    └── vs_opengene_<baseline>/        # which OpenGene fastp version was the baseline
        └── <platform>/                # hardware: galaxy_arm_a100, laptop_x86_rtx3050, ...
            ├── hardware.txt           #   shared hardware description
            └── <YYYYMMDD_HHMMSS>[_suffix]/    # one run per timestamp (suffix optional, e.g. _no_gzip)
                ├── meta.txt           #   provenance: versions, datasets, threads, notes
                ├── full_benchmark_<TS>.csv
                ├── full_profiling_<TS>.log
                ├── full_validation_<TS>.log
                └── full_status_<TS>.txt   #  (when produced by run_benchmark.sh)
```

The paper's canonical analysis input lives at
`fastp-gpu_v1.2.2/vs_opengene_v1.1.0/galaxy_arm_a100/20260417/full_benchmark_20260417.csv`
(merged from raw runs spanning 2026-04-15 — 2026-04-17).

## Adding a new run

1. Pick the right `fastp-gpu_<release>/vs_opengene_<baseline>/<platform>/` path
   (create folders if it's a new combination).
2. Create a timestamped folder under it.
3. Drop the run's CSV/logs/status file inside.
4. Add a `meta.txt` describing: timestamp, platform, fastp-gpu version, opengene
   version, datasets, thread sweep, hard timeout, notable issues.
5. If the run replaces the canonical inputs used by figures/tables, update
   `consolidated/consolidated_benchmark.csv` (and rerun the table-generation
   scripts).

## Conventions

- **Versions** in folder names match the binary's reported version (e.g.
  `fastp-gpu_v1.2.2`, `vs_opengene_v1.1.0`). Do NOT mix baselines inside one
  CSV — file separately under a different `vs_opengene_X.Y.Z/` directory.
- **Crashed/timed-out** rows: `reads_in=0` and `reads_out=0`; baseline rows are
  marked `pct_vs_opengene=crashed`, `validation=crashed`. Tool rows in the same
  `(dataset, type, threads, rep)` group are marked `pct_vs_opengene=N/A`.
- **Hard timeout**: Some runs in `20260426_114535` hit a 1800 s wall-clock cap.
  See that folder's `meta.txt` for which cells are affected.

## Canonical analysis inputs

Figures and tables read from
`fastp-gpu_v1.2.2/vs_opengene_v1.1.0/galaxy_arm_a100/20260417/full_benchmark_20260417.csv`.
This is intentionally a separate, pinned file — re-running raw benchmarks does
not change paper outputs until someone explicitly updates it.

Scripts that consume this file:
- `scripts/generate_figures.py`
- `scripts/apply_t32_results.py`
- `docs/publication/tables/extract_tables.py`
- `docs/publication/tables/verify_tables.py`
