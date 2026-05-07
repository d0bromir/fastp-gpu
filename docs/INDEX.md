# fastp-GPU Documentation Index

**fastp Version:** 1.3.3-d0bromir | **CUDA:** 12.0+ | **Last Updated:** 2026-05

## Documentation

| Document | Description |
|---|---|
| [../README.md](../README.md) | Top-level project README and full fastp user guide (all options, features, examples) |
| [BUILD_WITH_CUDA.md](BUILD_WITH_CUDA.md) | Build prerequisites, instructions, GPU architecture table, troubleshooting |
| [CUDA_ACCELERATION.md](CUDA_ACCELERATION.md) | GPU feature overview, usage guide, tuning, kernel design |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture diagrams, data flow, class relationships |
| [PERFORMANCE_SUMMARY.md](PERFORMANCE_SUMMARY.md) | Benchmark results, thread-scaling tables, optimisation details |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Quick-reference card: build, run, constants, troubleshooting |
| [VERSION_MANAGEMENT.md](VERSION_MANAGEMENT.md) | Auto-patch version system usage and configuration |
| [publication/](publication/) | LaTeX sources, figures, and built PDFs of the academic paper |

## Quick Start

```bash
# Build with GPU support
make clean && make WITH_CUDA=1 WITH_NVCOMP=1 -j$(nproc)

# Build with GPU + GDS (GPU-Direct Storage)
make clean && make WITH_CUDA=1 WITH_NVCOMP=1 WITH_GDS=1 -j$(nproc)

# Run (GPU is automatic when available)
./fastp -i input.fq.gz -o output.fq -w 8

# Run with GDS (requires nvidia-fs kernel module)
./fastp -i input.fq.gz -o output.fq -w 8 --use_gds

# CPU-only build
make clean && make -j$(nproc)
```

## Key Source Files

| File | Purpose |
|---|---|
| `src/cuda_stats.cu` | Warp-per-read GPU kernel for per-read statistics |
| `src/cuda_stats_wrapper.cpp` | 8-slot concurrent pool, pinned memory, batch dispatch |
| `src/cuda_gzip.cu` | BGZF-GPU decompression via nvCOMP |
| `src/gds_pipeline.cu` | GPU-Direct Storage NVMe-to-GPU pipeline |
| `src/gds_pipeline.h` | GDS pipeline class (stub when GDS disabled) |
| `src/filter.cpp` | `filterBatchGPU()` — GPU batch filtering integration |
| `src/seprocessor.cpp` | SE pipeline with speculative post-filter stats |
| `src/peprocessor.cpp` | PE pipeline with speculative post-filter stats |
| `src/stats.cpp` | `statRead()` / `unstatRead()` — per-base histogram computation |
