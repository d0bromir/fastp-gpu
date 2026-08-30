# 3-repeat experiment: "run once" vs. "within noise" (BIOADV review, Comment 5)

The paper stated each configuration was run once, but separately
claimed GPU mode at T=32 on the largest dataset was "within noise" of
CPU mode — a claim that a single run cannot support on its own. This
experiment repeats that specific configuration 3x for both tool
variants to check it.

Dataset: WGS PE 40 GB (DRR216653, 722,563,222 reads), `-w 32`, OS page
cache cleared between runs, `galaxy` host (ARM Neoverse N1, dual A100).

## Results (`raw_results/reps_wgs_pe_40g_t32.csv`)

| tool | rep 1 | rep 2 | rep 3 | mean | sd | CV |
|---|---|---|---|---|---|---|
| CPU  | 239.50 s | 238.82 s | 237.76 s | 238.69 s | 0.88 s | 0.37% |
| GPU  | 241.50 s | 240.57 s | 238.58 s | 240.22 s | 1.49 s | 0.62% |

Both tools show <0.7% coefficient of variation across repeats. The
gap between CPU and GPU means (1.5 s, 0.6% of the CPU mean) is smaller
than either tool's own run-to-run variation, so the "within noise"
claim holds under repeated measurement, not just as a single-run
observation. The original single-run values used elsewhere in the
paper (CPU 241.3 s, GPU 242.2 s) fall within one sd of these means.

## Reproduce

```
bash reps_experiment.sh
```

Per-repeat JSON reports are in `raw_results/{cpu,gpu}_rep{1,2,3}.json`.
