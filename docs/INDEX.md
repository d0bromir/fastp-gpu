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
| [SOFTWARE_QUALIFICATION_REPORT.md](SOFTWARE_QUALIFICATION_REPORT.md) | IEC 62304 Software Qualification Report — requirements, tests, evidence (SQR-FASTPGPU-001) |
| [RISK_MANAGEMENT.md](RISK_MANAGEMENT.md) | ISO 14971 Risk Management File — hazard analysis, controls, residual risk (RMF-FASTPGPU-001) |
| [SOUP_LIST.md](SOUP_LIST.md) | IEC 62304 §8.1.2 Software of Unknown Provenance list — versions, licenses, anomaly assessment (SOUP-FASTPGPU-001) |
| [TRACEABILITY_MATRIX.md](TRACEABILITY_MATRIX.md) | Regulatory Traceability Matrix — IEC 62304 / IVDR / ISO 15189 clauses → requirements → tests → evidence (RTM-FASTPGPU-001) |
| [EU_IVD_STANDARDS_REFERENCE.md](EU_IVD_STANDARDS_REFERENCE.md) | EU IVD standards reference — generic, product-agnostic overview of IVDR / IEC 62304 / ISO 14971 / ISO 15189 / EN ISO 13485 / MDCG guidance with hierarchy, key clauses, and applicability rationale (REF-IVDSW-001) |
| [EU_CLINICAL_STANDARDS_PLAN.md](EU_CLINICAL_STANDARDS_PLAN.md) | EU clinical standards — how each standard applies to fastp-gpu, clause-by-clause implementation checklist, gap analysis, and step-by-step deployment roadmap (PLAN-FASTPGPU-001) |
| [EU_CLINICAL_COMPLIANCE.md](EU_CLINICAL_COMPLIANCE.md) | EU clinical compliance evidence — clause-by-clause demonstration of how fastp-gpu meets every applicable requirement after 2026-05-10 fixes (COMP-FASTPGPU-001) |

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
