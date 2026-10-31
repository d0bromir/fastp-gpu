# fastp GPU CUDA Acceleration - Per-Read Statistics

**Last Updated:** January 29, 2026  
**fastp Version:** 1.3.3-d0bromir (rebased on upstream 1.3.3)  
**CUDA Version:** 12.0+  
**Status:** ✅ Production Ready

## Overview

This GPU acceleration enhancement enables CUDA-accelerated computation of per-read statistics in fastp, specifically:
- Base counting
- N-rate calculation  
- Average quality score computation

Per-read statistics are fundamental to quality filtering and reporting in fastp. By utilizing NVIDIA GPUs with CUDA support, these computations can be significantly accelerated, especially when processing large batches of reads.

## Performance Characteristics

The GPU acceleration provides:
- **Parallel Processing**: 1 warp (32 threads) per read, 8 reads per block, up to 8 concurrent slots
- **Batch Processing**: `PACK_SIZE=8192` reads per pack; slots hold up to 16384 reads
- **Automatic Fallback**: CPU processing used if GPU unavailable or on error

**Measured results (ARM Neoverse N1 + A100-SXM4-80GB, ~/benchmark/results/):**

| Dataset | fastp-opengene CPU | fastp-d0bromir GPU | Delta |
|---|---|---|---|
| Panel 148 MB (4T) | 3.8 s | 7.4 s | GPU +3.6 s |
| WGS-6G (8T) | 96.8 s | 103.0 s | GPU +6.2 s |
| WGS-6G (32T, GPU vs CPU-forced) | 87.1 s | 87.3 s | ≈00 s |

The bottleneck is gzip decompression + file I/O, which dominates >97% of runtime.
GPU statistics kernels complete in microseconds per batch; the overhead of CUDA
stream management slightly exceeds the per-batch savings on typical WGS workloads.
At high parallelism (32 threads) the 8-slot concurrent pool achieves near-parity.

## Architecture

### Files Added

1. **cuda_stats.h** - C header with CUDA function interfaces
2. **cuda_stats.cu** - CUDA kernels and GPU memory management
   - `compute_read_stats_warp_kernel()`: Unified warp-per-read kernel (production)
   - Legacy kernels (`compute_read_stats_kernel`, `quality_trim_kernel`, `detect_polyG_kernel`) retained but not called in production path
   - `cuda_compute_read_stats_device()`: device-side launcher used by wrapper
   - Error handling with CUDA API

3. **cuda_stats_wrapper.h/cpp** - C++ wrapper class
   - `CudaStatsWrapper`: concurrent 8-slot pool (NUM_SLOTS=8)
   - Each slot owns pinned host buffers + device mirrors + `cudaStream_t` + `std::mutex`
   - `processBatch()` uses try-lock round-robin across slots for concurrency
   - Automatic GPU detection and fallback to CPU

4. **filter.h/cpp** - Enhanced filtering with GPU support
   - `filterBatchGPU()`: New method for batch GPU-accelerated filtering
   - Backward compatible with existing `passFilter()` method
   - Low complexity filtering remains on CPU (sequential-dependent)

5. **Makefile** - Build configuration
   - `make WITH_CUDA=1` enables CUDA compilation
   - `make` (default) builds CPU-only via `cuda_stats_stub.cpp`

6. **gds_pipeline.h** - GDS pipeline header
   - `GdsPipeline` class: NVMe-to-GPU DMA + BGZF decompression + on-GPU stats
   - Stub no-op when `WITH_GDS` is not defined

7. **gds_pipeline.cu** - GDS pipeline CUDA implementation
   - `bgzf_strip_headers_kernel`, `bgzf_concat_kernel`, `gds_read_stats_kernel`
   - 9-step `readAndDecompress()` pipeline: cuFileRead → header parse → nvcomp decompress → stats
   - Runtime detection of `nvidia-fs` kernel module via `/proc/modules`

### Algorithm Details

**CUDA Kernel (compute_read_stats_warp_kernel)**

```
Config: BLOCK_SIZE=256, READS_PER_BLOCK=8 (32 threads per read)
Grid:   ceil(num_reads / 8) blocks

For each warp (processes one read):
  Phase 1 – all 32 lanes active:
    Each lane processes bases at positions lane, lane+32, lane+64, …
    Accumulates n_bases, low_qual_bases, total_quality per lane.
    5-round warp reduction via __shfl_down_sync → lane 0 holds totals.

  Phase 2 – lane 0 only:
    Write total_bases, n_bases, low_qual_bases, total_quality to d_stats.
    Forward scan  → trim_start (first position ≥ qual_threshold).
    Backward scan → trim_end   (last  position ≥ qual_threshold).
    Backward scan → polyG_trim_pos (G-run of ≥10 bases; -1 if none).
```

**Memory Layout**

- Sequence data: Concatenated in continuous GPU memory blocks
- Quality data: Separate continuous GPU memory blocks
- Pointer arrays: Device arrays holding pointers to sequence/quality regions
- Results: Pre-allocated output array for per-read statistics

## Building with CUDA Support

### Prerequisites

1. **NVIDIA CUDA Toolkit** (12.0 or later)
   ```bash
   # Ubuntu/Debian
   sudo apt-get install nvidia-cuda-toolkit
   
   # Or download from: https://developer.nvidia.com/cuda-downloads
   ```

2. **NVIDIA GPU Driver**
   ```bash
   nvidia-smi  # Verify installation
   ```

3. **libdeflate + isa-l** (required by fastp for BGZF decompression)

### Build Options

**Build with CUDA support (recommended):**
```bash
cd /home/mpiuser/tools/src/fastp_d0bromir
make WITH_CUDA=1 -j$(nproc)
```

**Build without CUDA (CPU-only fallback):**
```bash
make -j$(nproc)
```

**Check GPU compute capability:**
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

The Makefile compiles against the detected or default CUDA arch. On the
reference hardware (A100-SXM4-80GB) the correct capability is CC 8.0.

Compute capability reference:
- CC 7.0: Volta (Tesla V100)
- CC 7.5: Turing (RTX 2080, RTX 2070)
- CC 8.0: Ampere (A100, RTX 3080, RTX 3090)
- CC 8.9: Ada (RTX 4090, RTX 6000 Ada)

### Verify GPU Acceleration

The executable will print GPU status on startup:
```
CUDA GPU is available (Device ID: 0). Per-read statistics will be GPU-accelerated.
```

Or for CPU-only mode:
```
CUDA GPU not available. Per-read statistics will use CPU fallback.
```

## Usage

No changes to command-line interface. GPU acceleration is transparent:

```bash
# Standard fastp usage - GPU automatically used if available
fastp -i input.fq -o output.fq

# With paired-end reads
fastp -i R1.fq -I R2.fq -o R1_clean.fq -O R2_clean.fq

# Quality filtering (now GPU-accelerated)
fastp -i input.fq -o output.fq -q 20 -u 40
```

GPU acceleration is automatically:
1. **Enabled** when CUDA-capable GPU is detected
2. **Disabled** with fallback to CPU if GPU unavailable
3. **Applied** during batch processing of reads
4. **Transparent** to user (no behavior changes, only performance)

## Implementation Details

### Integration Points

1. **Filter::filterBatchGPU()** 
   - Called for batch filtering operations
   - Processes multiple reads in parallel on GPU
   - Falls back to CPU `passFilter()` for each read if GPU unavailable

2. **CudaStatsWrapper**
   - Singleton-like pattern for GPU device management
   - Lazy initialization of CUDA context
   - Automatic cleanup in destructor

3. **Memory Management**
   - Pinned (page-locked) host memory for zero-copy transfers
   - Continuous device memory allocation for efficient access
   - Automatic cleanup of GPU resources on error

### Error Handling

- CUDA errors are caught and logged to stderr
- Graceful fallback to CPU processing on any GPU error
- No program termination on GPU failures
- Compatibility maintained across all GPU generations

### Thread Safety

- GPU operations are serialized per Filter instance
- Multiple Filter instances can use the same GPU (with CUDA stream support possible in future)
- Quality filter remains thread-safe

## Performance Tuning

### Batch and Slot Sizing

The Makefile sets `PACK_SIZE=8192` reads per CPU pack (see `src/common.h`).
Each of the `NUM_SLOTS=8` concurrent GPU slots is sized for `SLOT_MAX_READS=16384`
reads (`SLOT_BUF_BYTES=16 MB` per slot). Total device memory committed: ~256 MB.

For best performance:
- **< 8192 reads / pack**: Some slots idle; performance still acceptable
- **8192 reads / pack**: Nominal operating point; 1024 blocks on A100
- **Increasing -w threads**: Up to 8 threads can saturate all 8 slots
  simultaneously; beyond 8 threads the bottleneck shifts to I/O.

### GPU Memory Limits (A100-SXM4-80GB)

80 GB device memory; 256 MB committed by the slot pool. Memory is not a
limitation for any realistic single-process run.

### When GPU Helps vs Does Not Help

- **Compute-bound workloads** (e.g., synthetic benchmarks with preloaded data):
  the warp kernel will outperform sequential CPU loops.
- **I/O-bound workloads** (real gzip FASTQ files): I/O+decompression >97% of
  runtime; GPU kernel time is negligible, adding slight overhead.

## Troubleshooting

### Issue: "CUDA GPU not available" message

**Solution 1: Check GPU Installation**
```bash
nvidia-smi
nvcc --version
```

**Solution 2: Verify CUDA Paths**
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Solution 3: Rebuild with Verbose Output**
```bash
make WITH_CUDA=1 VERBOSE=1
```

### Issue: CUDA Compute Capability Mismatch

If you get "compute capability is not supported" error:
1. Check your GPU's compute capability: `nvidia-smi --query-gpu=compute_cap`
2. Edit the `NVCC_FLAGS` in `Makefile` to match your GPU, then rebuild:
   ```bash
   make clean && make WITH_CUDA=1
   ```

### Issue: Out of GPU Memory

If processing large batches:
1. Reduce batch size (implement in calling code)
2. Use CPU fallback: Remove GPU code path
3. Process in multiple smaller batches

## GPU-Direct Storage (GDS)

### Overview

GPU-Direct Storage enables direct NVMe-to-GPU DMA transfers using NVIDIA's
`cuFile` API, bypassing CPU memory entirely. For BGZF-compressed FASTQ files,
the entire pipeline runs on-GPU:

```
NVMe → cuFileRead() → GPU buffer → BGZF header parse → nvCOMP decompress → per-read stats
```

### Requirements

| Component | Package | Purpose |
|---|---|---|
| Kernel module | `nvidia-fs-dkms` | NVMe-to-GPU DMA driver |
| Userspace library | `libcufile-dev` | cuFile API |
| Build flag | `WITH_GDS=1` | Compile GDS support |
| Runtime flag | `--use_gds` | Activate GDS I/O path |

**Critical:** The `nvidia-fs` kernel module must be loaded for GDS to function.
The userspace libraries (`libcufile`, `gds-tools`) alone are not sufficient.

```bash
# Install and load the kernel module
sudo apt install nvidia-fs-dkms
sudo modprobe nvidia_fs

# Verify
lsmod | grep nvidia_fs
```

If `modprobe` fails with "Module not found" after a kernel update, the DKMS
module needs to be rebuilt for the new kernel:
```bash
sudo apt install linux-headers-$(uname -r)
sudo dkms build nvidia-fs/2.28.2 -k $(uname -r) --force
sudo dkms install nvidia-fs/2.28.2 -k $(uname -r) --force
sudo modprobe nvidia_fs
```

### Runtime Behavior

fastp checks for GDS availability at startup:

1. **nvidia-fs loaded** → Full GDS pipeline active (NVMe → GPU DMA)
2. **nvidia-fs not loaded** → Graceful fallback to standard GPU I/O path
3. **Input not BGZF** → Fallback (GDS pipeline requires BGZF block structure)

The fallback is transparent and produces bit-identical output with no
performance penalty. There is no need to remove `--use_gds` when the
kernel module is unavailable.

### Build

```bash
make clean && make WITH_CUDA=1 WITH_NVCOMP=1 WITH_GDS=1 -j$(nproc)
```

### Usage

```bash
# Activate GDS for BGZF input files
fastp -i input.fq.gz -o output.fq --use_gds

# Paired-end with GDS
fastp -i R1.fq.gz -I R2.fq.gz -o out1.fq -O out2.fq --use_gds -w 8
```

### Validation Results

All outputs are bit-identical across CPU, GPU, and GPU+GDS modes:
- 18/18 JSON statistics match (total reads, bases, Q20/Q30, GC, adapters, etc.)
- FASTQ MD5 checksums identical
- Tested on BGZF panel files (148 MB) and plain gzip WGS files (6-9 GB)

## Future Enhancements

Potential optimizations for future releases:

1. **Multi-GPU Support**: Distribute slots across multiple GPU devices
2. **Kernel fusion**: Merge stats + adapter trimming into one kernel launch
3. **Profile Optimizations**: Auto-tuning READS_PER_BLOCK for different GPU generations
4. **GDS with nvidia-fs**: Full NVMe-to-GPU pipeline when kernel module is available

## References

- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- NVIDIA GPU Computing: https://developer.nvidia.com/cuda-zone/

## License

Same as fastp - MIT License
