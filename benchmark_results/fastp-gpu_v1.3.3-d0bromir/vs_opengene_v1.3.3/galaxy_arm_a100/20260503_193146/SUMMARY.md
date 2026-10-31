# Speedup summary vs. OpenGene fastp v1.3.3

Source: `full_benchmark_20260503_193146.csv` (n = 58 validated dataset×thread cells per build).

| Build | Arithmetic mean | Geometric mean | Median | Range |
|---|---|---|---|---|
| d0bromir GPU | 1.62× | 1.22× | 1.34× | 0.27× – 8.53× |
| d0bromir CPU | 1.65× | 1.39× | 1.34× | 0.50× – 6.91× |

Headline: **geomean 1.22× (GPU), 1.39× (CPU)**. The arithmetic mean is inflated
by large SE WGS/BS-Seq runs and dragged down by sub-second Panel datasets where
GPU launch/transfer overhead dominates.

Best case: `BS_HiSeq_SE_26.8G` at T=32 → **8.53× (GPU)**, **6.91× (CPU)**.
Worst case: `Panel_SE_148M` at T=32 → 0.27× (GPU), 0.50× (CPU); the panel
workloads are too small for the GPU pipeline to amortise.

Where the GPU pays off (multi-GB WGS PE / large SE): 1.3×–2.0× across thread
counts, scaling to 7–8× on the largest SE input.
