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

<span style="color: #B22222;">Both modes show a coefficient of
variation below 0.7%. The difference between means is approximately
1.5 s (0.6% of the CPU mean), comparable to the sample SDs of 0.88 s
and 1.49 s. It is not smaller than either SD. These three repeats show
similar observed runtimes, not statistical equivalence.</span>

<span style="color: #B22222;">The original single-run values
(CPU 241.3 s, GPU 242.2 s) are separate observations. They are
approximately 3.0 and 1.3 sample SDs above the respective repeat
means, not within one SD. The manuscript distinguishes the original
sweep from this repeat experiment.</span>

## Reproduce

```
bash reps_experiment.sh
```

Per-repeat JSON reports are in `raw_results/{cpu,gpu}_rep{1,2,3}.json`.
