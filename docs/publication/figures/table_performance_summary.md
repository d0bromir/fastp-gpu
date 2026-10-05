# Performance Summary Table

All measurements at T=8 threads, gzip-compressed output, OS page cache cleared before each run.

CPU = d0bromir_cpu (fastp-cpu-profile, -march=native -flto, PROFILING=1)
GPU = d0bromir_gpu (fastp, WITH_CUDA=1 PROFILING=1)

## Table 1: End-to-end wall time and speedup vs. opengene (T=8)

| Dataset | Size | Mode | opengene (s) | d0bromir CPU (s) | d0bromir GPU (s) | CPU vs opengene | GPU vs opengene | GPU vs CPU |
|---------|------|------|-------------|-----------------|-----------------|-----------------|-----------------|------------|
| Panel_SE_148M | 148 MB | SE | 7.6 | 3.9 | 7.4 | +48.3% | +2.7% | +88.3% |
| Panel_PE_304M | 304 MB | PE | 6.7 | 3.4 | 6.9 | +49.3% | -3.1% | +103.5% |
| WGS_SE_6.3G | 6.3 GB | SE | 293.0 | 82.4 | 86.9 | +71.9% | +70.3% | +5.5% |
| WGS_PE_12.8G | 12.8 GB | PE | 269.0 | 76.5 | 79.5 | +71.6% | +70.5% | +3.9% |
| WGS_PE_18.2G | 18.2 GB | PE | 397.5 | 102.0 | 104.4 | +74.3% | +73.7% | +2.4% |
| WGS_PE_40G | 40 GB | PE | 1098.3 | 479.3 | 474.3 | +56.4% | +56.8% | -1.0% |

## Table 2: Stage breakdown (d0bromir CPU and GPU, T=8)

| Dataset | Tool | Wall (s) | Loading (ms) | Filtering (ms) | Writing (ms) | Loading% | Filtering% | Writing% |
|---------|------|----------|-------------|---------------|-------------|----------|------------|----------|
| Panel_SE_148M | d0bromir_cpu | 3.905 | 2055 | 47 | 152 | 52.6% | 1.2% | 3.9% |
| Panel_SE_148M | d0bromir_gpu | 7.355 | 4331 | 38 | 182 | 58.9% | 0.5% | 2.5% |
| Panel_PE_304M | d0bromir_cpu | 3.379 | 2135 | 122 | 111 | 63.2% | 3.6% | 3.3% |
| Panel_PE_304M | d0bromir_gpu | 6.875 | 4377 | 97 | 146 | 63.7% | 1.4% | 2.1% |
| WGS_SE_6.3G | d0bromir_cpu | 82.424 | 69486 | 38 | 550 | 84.3% | 0.0% | 0.7% |
| WGS_SE_6.3G | d0bromir_gpu | 86.945 | 72375 | 44 | 541 | 83.2% | 0.1% | 0.6% |
| WGS_PE_12.8G | d0bromir_cpu | 76.5 | 72454 | 815 | 70 | 94.7% | 1.1% | 0.1% |
| WGS_PE_12.8G | d0bromir_gpu | 79.486 | 75013 | 697 | 82 | 94.4% | 0.9% | 0.1% |
| WGS_PE_18.2G | d0bromir_cpu | 101.979 | 98427 | 247 | 343 | 96.5% | 0.2% | 0.3% |
| WGS_PE_18.2G | d0bromir_gpu | 104.383 | 102016 | 151 | 136 | 97.7% | 0.1% | 0.1% |
| WGS_PE_40G | d0bromir_cpu | 479.325 | 546275 | 1409 | 102 | 114.0% | 0.3% | 0.0% |
| WGS_PE_40G | d0bromir_gpu | 474.334 | 484890 | 1272 | 101 | 102.2% | 0.3% | 0.0% |
