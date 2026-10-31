# GPU CUDA Acceleration Build Instructions

**Last Updated:** January 29, 2026  
**fastp Version:** 1.3.3-d0bromir (rebased on upstream 1.3.3)  
**CUDA Version Required:** 12.0 or higher  
**Status:** ✅ Build Verified & Tested

## Quick Start

### Option 1: Build with CUDA (Recommended)

```bash
# Navigate to fastp directory
cd /home/mpiuser/tools/src/fastp_d0bromir

# Build with GPU support
make WITH_CUDA=1 -j$(nproc)

# Verify
./fastp --version
```

### Option 2: Build with CUDA + GDS (GPU-Direct Storage)

```bash
cd /home/mpiuser/tools/src/fastp_d0bromir

# Build with GPU stats, nvCOMP decompression, and GDS support
make WITH_CUDA=1 WITH_NVCOMP=1 WITH_GDS=1 -j$(nproc)
```

GDS enables NVMe-to-GPU DMA transfers, bypassing CPU memory for BGZF-compressed
input files. Requires the `nvidia-fs` kernel module (see [GDS Prerequisites](#gds-prerequisites-gpu-direct-storage) below).

### Option 3: Build without CUDA (CPU Only)

```bash
cd /home/mpiuser/tools/src/fastp_d0bromir
make -j$(nproc)
```

## Detailed Build Instructions

### Prerequisites Check

1. **Verify CUDA Installation**
   ```bash
   nvcc --version
   nvidia-smi
   ```

2. **Install Dependencies** (Ubuntu/Debian)
   ```bash
   sudo apt-get update
   sudo apt-get install -y \
       g++ \
       nvidia-cuda-toolkit \
       libdeflate-dev \
       libisal-dev
   ```

### Build Process

```bash
cd /home/mpiuser/tools/src/fastp_d0bromir
make clean
make WITH_CUDA=1 -j$(nproc)
```

### Testing the GPU Build

1. **Run Unit Tests**
   ```bash
   cd build
   
   # Build CUDA unit tests
   nvcc ../src/test_cuda_stats.cu ../src/cuda_stats.cu -o test_cuda_stats
   
   # Run tests
   ./test_cuda_stats
   ```

2. **Run fastp with GPU**
   ```bash
   # Process test data
   ./fastp -i ../testdata/R1.fq -o test_output.fq
   
   # Check output for GPU status (in stderr)
   # Should show: "CUDA GPU is available (Device ID: 0)"
   ```

## GPU Architecture Reference

| GPU | Compute Capability | Notes |
|---|---|---|
| A100-SXM4-80GB | 8.0 | Reference hardware |
| RTX 3080/3090 | 8.6 | |
| RTX 4090 | 8.9 | |
| V100 | 7.0 | |
| RTX 2080 | 7.5 | |

**Find your GPU's compute capability:**
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

The Makefile compiles for the default CC. Edit `NVCC_FLAGS` in `Makefile` to add
`-arch=sm_XX` for a specific target.

## Verify Build

```bash
./fastp --version
# Check stderr for: "[GPU] CUDA GPU available (Device 0)"
```

## Troubleshooting

## GDS Prerequisites (GPU-Direct Storage)

To use `--use_gds` for NVMe-to-GPU direct I/O, you need the `nvidia-fs` kernel
module in addition to the userspace `libcufile` libraries.

### 1. Install the nvidia-fs DKMS kernel module
```bash
# Install kernel headers for your running kernel
sudo apt install linux-headers-$(uname -r)

# Install the nvidia-fs DKMS package
sudo apt install nvidia-fs-dkms
```

### 2. Load the kernel module
```bash
sudo modprobe nvidia_fs

# Verify it loaded
lsmod | grep nvidia_fs
```

### 3. Install userspace GDS libraries (if not already present)
```bash
sudo apt install libcufile-dev-13-1 gds-tools-13-1
# Or use the meta-package: sudo apt install nvidia-gds-13-1
```

> **Note:** The userspace libraries (`libcufile`, `gds-tools`) alone are
> insufficient. The `nvidia-fs-dkms` package provides the kernel driver that
> enables actual NVMe-to-GPU DMA. Without it, `--use_gds` gracefully falls back
> to the standard GPU I/O path with no performance penalty.

**Common issue:** `modinfo nvidia-fs` returns "Module not found"
- **Cause:** Only `libcufile` / `gds-tools` were installed, not the kernel module.
- **Fix:** `sudo apt install nvidia-fs-dkms && sudo modprobe nvidia_fs`

**Common issue:** `modprobe: FATAL: Module nvidia_fs not found in directory /lib/modules/<kernel>`
- **Cause:** The kernel was updated but DKMS did not automatically rebuild
  `nvidia-fs` for the new kernel. This happens when DKMS detects a missing
  build dependency (`nvidia`).
- **Fix:** Force-rebuild and install for the running kernel:
  ```bash
  sudo apt install linux-headers-$(uname -r)
  sudo dkms build nvidia-fs/$(dkms status nvidia-fs | grep -oP '[\d.]+' | head -1) -k $(uname -r) --force
  sudo dkms install nvidia-fs/$(dkms status nvidia-fs | grep -oP '[\d.]+' | head -1) -k $(uname -r) --force
  sudo modprobe nvidia_fs
  ```
  The `--force` flag overrides the `nvidia` build-dependency check, which is
  safe because the NVIDIA driver is already loaded at runtime.

**Verify GDS readiness:**
```bash
# Check kernel module
lsmod | grep nvidia_fs

# Check cuFile library
ldconfig -p | grep cufile

# Runtime check (fastp will print on startup):
# "[GDS] nvidia-fs module loaded — GPU-Direct Storage active"
# or "[GDS] nvidia-fs module not loaded — falling back to standard GPU I/O"
```

## Build Variants

### Minimal Build (CPU Only)
```bash
make clean && make
# Binary: ./fastp  (uses cuda_stats_stub.cpp, no GPU)
```

### GPU Build (Recommended)
```bash
make clean && make WITH_CUDA=1
# Binary: ./fastp  (full GPU support)
```

### GPU + GDS Build (Full Pipeline)
```bash
make clean && make WITH_CUDA=1 WITH_NVCOMP=1 WITH_GDS=1
# Binary: ./fastp  (GPU stats + nvCOMP decompression + GDS direct I/O)
# Use --use_gds flag at runtime to activate GDS path
```

### GPU + GDS Build (Full Pipeline)
```bash
make clean && make WITH_CUDA=1 WITH_NVCOMP=1 WITH_GDS=1
# Binary: ./fastp  (GPU stats + nvCOMP decompression + GDS direct I/O)
# Use --use_gds flag at runtime to activate GDS path
```

### Debug Build with Symbols
```bash
make clean && make WITH_CUDA=1 DEBUG=1
```

### Specify GPU Architecture
```bash
make clean && make WITH_CUDA=1 CUDA_ARCH=80  # A100 CC 8.0
make clean && make WITH_CUDA=1 CUDA_ARCH=75  # RTX 2080 CC 7.5
```

## Installation

### System Installation
```bash
sudo cp fastp /usr/local/bin/
# Binary at: /usr/local/bin/fastp
```

### User Installation
```bash
cp fastp ~/local/bin/
# Binary at: ~/local/bin/fastp  (ensure ~/local/bin is in PATH)
```

## Verify Installation

```bash
# Check binary location
which fastp

# Check GPU availability
fastp --version 2>&1 | grep -i cuda

# Check linked libraries
ldd $(which fastp) | grep cuda
```

## Performance Verification

To verify GPU acceleration is working:

```bash
# Create test data
fastq_generate_test_data.sh > large_test.fq

# Run with GPU (check stderr for GPU message)
time fastp -i large_test.fq -o output.fq -q 20 -u 40 2>&1 | grep -i cuda

# Run CPU version for comparison
# Build separate CPU version and compare runtimes
```

## Uninstall

```bash
cd build
sudo make uninstall

# Or remove manually
sudo rm /usr/local/bin/fastp
```

## Environment Variables

### At Runtime

```bash
# Force CPU mode (ignore GPU)
export CUDA_VISIBLE_DEVICES=""
./fastp -i input.fq -o output.fq

# Verbose CUDA error checking
export CUDA_LAUNCH_BLOCKING=1
./fastp -i input.fq -o output.fq

# Limit GPU memory usage (MB)
export CUDA_VISIBLE_DEVICES=0  # Use only GPU 0
```

### At Build Time

```bash
# Verbose build output
make WITH_CUDA=1 V=1

# Extra NVCC flags
make WITH_CUDA=1 NVCCFLAGS="-keep"
```

## Advanced Configuration

### Multi-GPU Support (Future Enhancement)

Currently fastp uses one GPU per process. To support multiple GPUs:

1. Set CUDA device in code: `cudaSetDevice(device_id)`
2. Distribute batches across devices
3. Compile for multiple architectures: `make WITH_CUDA=1 CUDA_ARCH="60 70 80"`

### Custom CUDA Flags

```bash
make WITH_CUDA=1 NVCCFLAGS="-O3 --use_fast_math"
```

### Custom Compiler Flags

```bash
make WITH_CUDA=1 CXXFLAGS="-march=native -O3"
```

## Build System Information

The Makefile (`WITH_CUDA=1`) handles:
- Conditional CUDA compilation (nvcc for `.cu` files)
- Linking of `-lcudart -lcuda` when GPU enabled
- CPU-only stub (`cuda_stats_stub.cpp`) when GPU disabled
- Default GPU architecture (CC 8.0 for A100)

Generated build outputs:
- `fastp`: Main executable (GPU or CPU-only depending on build)
- `obj/*.o`, `obj/*.d`: Object and dependency files

## Support

For issues:
1. Check build output: `make WITH_CUDA=1 V=1`
2. Verify CUDA setup: `nvidia-smi && nvcc --version`
3. Check system requirements: `cat /proc/cpuinfo | grep avx`
4. Review logs: Check /var/log/cuda
