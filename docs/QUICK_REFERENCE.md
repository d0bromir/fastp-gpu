# GPU-Accelerated fastp — Quick Reference

**Version:** 1.3.3-d0bromir | **CUDA:** 12.0+ | **Hardware:** ARM Neoverse N1 + A100-SXM4-80GB

## Build

```bash
# GPU build (requires CUDA toolkit + nvCOMP)
make clean && make WITH_CUDA=1 WITH_NVCOMP=1 -j$(nproc)

# GPU + GDS build (adds GPU-Direct Storage support)
make clean && make WITH_CUDA=1 WITH_NVCOMP=1 WITH_GDS=1 -j$(nproc)

# CPU-only build
make clean && make -j$(nproc)
```

## GDS Setup (GPU-Direct Storage)

```bash
# Install and load the nvidia-fs kernel module (required for GDS)
sudo apt install nvidia-fs-dkms
sudo modprobe nvidia_fs
lsmod | grep nvidia_fs   # verify

# Note: libcufile/gds-tools alone are NOT sufficient.
# The nvidia-fs-dkms kernel module is required for NVMe-to-GPU DMA.
```

## Run

```bash
# GPU is used automatically when available
fastp -i in.fq.gz -o out.fq -w 8

# Paired-end
fastp -i R1.fq.gz -I R2.fq.gz -o out1.fq -O out2.fq -w 8

# With GDS (requires nvidia-fs kernel module)
fastp -i in.fq.gz -o out.fq -w 8 --use_gds
```

## Performance (v1.3.3-d0bromir vs upstream fastp 1.3.3, A100 + ARM Neoverse N1)

Headline geometric-mean speedup over the published 8-dataset / 6-thread matrix:
**3.00× wall-clock** at the per-dataset best thread count.

For the full per-dataset / per-thread tables (and the canonical CSV that
generates them) see:

- [PERFORMANCE_SUMMARY.md](PERFORMANCE_SUMMARY.md) — tabulated headline numbers
- `benchmark_results/fastp-gpu_v1.3.3-d0bromir/vs_opengene_v1.3.3/` — raw CSV
- `docs/publication/GPU_ACCELERATED_FASTP_PAPER.pdf` — full evaluation

## Key Constants

```
NUM_SLOTS       = 8       # concurrent GPU dispatch slots
SLOT_MAX_READS  = 16384   # max reads per slot
SLOT_BUF_BYTES  = 16 MB   # pinned buffer per slot
PACK_SIZE       = 8192    # reads per CPU pack
BLOCK_SIZE      = 256     # CUDA threads per block
READS_PER_BLOCK = 8       # warps per block (32 threads/read)
```

## Architecture Overview

```
Read batch → filterBatchGPU() → GPU kernel (pass/fail per read)
                                        ↓
           SE/PE processor: speculative statRead() → unstatRead() for failures
```

- **Speculative stats**: `postStats->statRead()` runs for all reads in first pass (cache-hot).
  Only the ~5% that fail filtering are reversed via `unstatRead()`.
- **GPU kernel**: Warp-per-read design — 32 threads cooperatively process each read.
- **Slot pool**: 8-slot concurrent pool with pinned memory for overlap of CPU/GPU work.

## GPU Support Matrix

| GPU | Compute Capability | Status |
|---|---|---|
| A100 / H100 | 8.0 / 9.0 | Reference hardware |
| RTX 4090 | 8.9 | Supported |
| RTX 3080 | 8.6 | Supported |
| V100 | 7.0 | Supported |
| P100 | 6.0 | Supported |

## Troubleshooting

```bash
# Verify CUDA
nvcc --version && nvidia-smi

# Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# If "GPU not available" at runtime
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# If GDS falls back ("nvidia-fs module not loaded")
sudo apt install nvidia-fs-dkms && sudo modprobe nvidia_fs
```

## Documentation

See [INDEX.md](INDEX.md) for full documentation guide.
