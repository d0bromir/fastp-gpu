/*
 * gds_pipeline.cu — GPU-Direct Storage end-to-end pipeline
 *
 * NVMe → GPU DMA → GPU decompress → GPU FASTQ parse → GPU stats
 *
 * See gds_pipeline.h for full documentation.
 */

#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP) && defined(HAVE_GDS)

#include "gds_pipeline.h"
#include <fcntl.h>        /* open(), O_RDONLY, O_DIRECT */
#include <unistd.h>       /* close(), lseek() */
#include <sys/stat.h>     /* fstat */
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <fstream>        /* ifstream for /proc/modules check */

/* Shared CUDA / nvCOMP error-check macros (see src/cuda_error_check.h). */
#define FASTP_CUDA_TAG "gds_pipeline"
#include "cuda_error_check.h"
#define GDS_CUDA_CHECK(call)   FASTP_CUDA_CHECK(call)
#define GDS_NVCOMP_CHECK(call) FASTP_NVCOMP_CHECK(call)

/* ─── BGZF constants ──────────────────────────────────────────────────── */
static const unsigned char BGZF_MAGIC_0 = 0x1f;
static const unsigned char BGZF_MAGIC_1 = 0x8b;

/* ═════════════════════════════════════════════════════════════════════════
 * GPU kernel: strip BGZF headers from contiguous raw blocks and produce
 * flat DEFLATE payloads for nvCOMP.
 *
 * Each thread processes one BGZF block:
 *   Input:  d_raw       contiguous raw BGZF blocks from GDS DMA
 *   Input:  d_offsets   byte offset of each block in d_raw
 *   Input:  d_sizes     total size of each block (including header/trailer)
 *   Output: d_deflate   flat buffer — each slot is GDS_BGZF_BLOCK_MAX wide
 *   Output: d_def_sizes actual DEFLATE payload size per block
 * ═════════════════════════════════════════════════════════════════════════*/
__global__ void bgzf_strip_headers_kernel(
    const unsigned char* __restrict__ d_raw,
    const uint32_t*      __restrict__ d_offsets,
    const uint32_t*      __restrict__ d_sizes,
    int                              num_chunks,
    unsigned char*                   d_deflate,
    size_t*                          d_def_sizes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_chunks) return;

    uint32_t off     = d_offsets[idx];
    uint32_t blk_sz  = d_sizes[idx];
    /* DEFLATE payload = skip 18-byte gzip header, drop 8-byte CRC+ISIZE */
    int hdr  = 18; /* GDS_BGZF_HDR */
    int trl  =  8; /* GDS_BGZF_TRL */
    int def_sz = (int)blk_sz - hdr - trl;
    if (def_sz < 0) def_sz = 0;

    d_def_sizes[idx] = (size_t)def_sz;

    /* Copy payload into uniform-stride output slot */
    const unsigned char* src = d_raw + off + hdr;
    unsigned char* dst = d_deflate + (size_t)idx * 65536; /* GDS_BGZF_BLOCK_MAX */
    for (int i = 0; i < def_sz; i++)
        dst[i] = src[i];
}

/* ═════════════════════════════════════════════════════════════════════════
 * GPU kernel: concatenate decompressed BGZF blocks into a contiguous
 * FASTQ text buffer.  Each thread handles one block.
 *
 * Input:  d_decomp_flat  — uniform-stride decompressed blocks
 * Input:  d_act_bytes    — actual decompressed size per block
 * Input:  d_out_offsets  — prefix-sum of actual sizes (exclusive scan)
 * Output: d_text         — tightly-packed FASTQ text
 * ═════════════════════════════════════════════════════════════════════════*/
__global__ void bgzf_concat_kernel(
    const unsigned char* __restrict__ d_decomp_flat,
    const size_t*        __restrict__ d_act_bytes,
    const size_t*        __restrict__ d_out_offsets,
    int                              num_chunks,
    char*                            d_text)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_chunks) return;

    size_t act  = d_act_bytes[idx];
    size_t ooff = d_out_offsets[idx];
    const unsigned char* src = d_decomp_flat + (size_t)idx * 65536;
    for (size_t i = 0; i < act; i++)
        d_text[ooff + i] = (char)src[i];
}

/* ═════════════════════════════════════════════════════════════════════════
 * Static helpers
 * ═════════════════════════════════════════════════════════════════════════*/

bool GdsPipeline::isAvailable() {
    /* Check that nvidia-fs kernel module is loaded.
     * Without it, cuFileDriverOpen() may return success but cuFileRead()
     * blocks indefinitely. */
    {
        std::ifstream modules("/proc/modules");
        if (modules.is_open()) {
            std::string line;
            bool found = false;
            while (std::getline(modules, line)) {
                if (line.compare(0, 9, "nvidia_fs") == 0 ||
                    line.compare(0, 9, "nvidia-fs") == 0) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                fprintf(stderr,
                        "[gds_pipeline] nvidia-fs kernel module not loaded; "
                        "GDS not available\n");
                return false;
            }
        }
    }

    CUfileError_t st = cuFileDriverOpen();
    if (st.err != CU_FILE_SUCCESS) return false;
    cuFileDriverClose();
    return true;
}

size_t GdsPipeline::parseBgzfBlockSize(const unsigned char* data,
                                       size_t avail) {
    if (avail < 20) return 0;
    if (data[0] != BGZF_MAGIC_0 || data[1] != BGZF_MAGIC_1) return 0;
    if (data[2] != 8) return 0;           /* DEFLATE method */
    if (!(data[3] & 0x04)) return 0;      /* FEXTRA flag    */
    uint16_t bsize = (uint16_t)(data[16]) |
                     ((uint16_t)(data[17]) << 8);
    return (size_t)bsize + 1;
}

/* ═════════════════════════════════════════════════════════════════════════
 * Constructor / Destructor
 * ═════════════════════════════════════════════════════════════════════════*/

GdsPipeline::GdsPipeline(int device)
    : mDevice(device), mValid(false), mEof(false), mStream(nullptr),
      mDriverOpen(false), mFd(-1), mFileOpen(false),
      mFileOffset(0), mFileSize(0),
      d_raw(nullptr), h_header_buf(nullptr),
      d_compressed_flat(nullptr), d_decompressed(nullptr),
      d_comp_ptrs(nullptr), d_decomp_ptrs(nullptr),
      d_comp_bytes(nullptr), d_decomp_buf_bytes(nullptr),
      d_decomp_act_bytes(nullptr), d_statuses(nullptr),
      d_temp(nullptr), mTempBytes(0), mDecompBytes(0),
      d_descs(nullptr), d_readCount(nullptr), mReadCount(0),
      d_stats(nullptr), h_stats(nullptr), h_pinned_text(nullptr)
{
    cudaError_t cerr = cudaSetDevice(mDevice);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "[gds_pipeline] cudaSetDevice(%d) failed: %s\n",
                mDevice, cudaGetErrorString(cerr));
        return;
    }

    CUfileError_t cst = cuFileDriverOpen();
    mDriverOpen = (cst.err == CU_FILE_SUCCESS);
    if (!mDriverOpen) {
        fprintf(stderr,
                "[gds_pipeline] cuFileDriverOpen failed (err=%d)\n",
                (int)cst.err);
        return;
    }

    allocate();
}

GdsPipeline::~GdsPipeline() {
    close();
    release();
    if (mDriverOpen) cuFileDriverClose();
}

/* ═════════════════════════════════════════════════════════════════════════
 * Resource allocation
 * ═════════════════════════════════════════════════════════════════════════*/

void GdsPipeline::allocate() {
    cudaError_t err;

    /* CUDA stream */
    err = cudaStreamCreate(&mStream);
    if (err != cudaSuccess) goto fail;

    /* ── GDS raw-read buffer (DMA target, registered with cuFile) ────── */
    err = cudaMalloc(&d_raw, GDS_RAW_BUF_BYTES);
    if (err != cudaSuccess) goto fail;
    {
        CUfileError_t cst = cuFileBufRegister(d_raw, GDS_RAW_BUF_BYTES, 0);
        if (cst.err != CU_FILE_SUCCESS) {
            fprintf(stderr,
                    "[gds_pipeline] cuFileBufRegister(d_raw) failed (err=%d)\n",
                    (int)cst.err);
            goto fail;
        }
    }

    /* ── Pinned host buffer for BGZF header scanning ───────────────── */
    err = cudaHostAlloc(&h_header_buf, HEADER_BUF_BYTES, cudaHostAllocDefault);
    if (err != cudaSuccess) goto fail;

    /* ── nvCOMP decompression buffers ──────────────────────────────── */
    err = cudaMalloc(&d_compressed_flat,
                     (size_t)GDS_MAX_CHUNKS * GDS_BGZF_BLOCK_MAX);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&d_decompressed,   GDS_DECOMP_BUF_BYTES);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(&d_comp_ptrs,       GDS_MAX_CHUNKS * sizeof(void*));
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&d_decomp_ptrs,     GDS_MAX_CHUNKS * sizeof(void*));
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&d_comp_bytes,      GDS_MAX_CHUNKS * sizeof(size_t));
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&d_decomp_buf_bytes, GDS_MAX_CHUNKS * sizeof(size_t));
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&d_decomp_act_bytes, GDS_MAX_CHUNKS * sizeof(size_t));
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&d_statuses,        GDS_MAX_CHUNKS * sizeof(nvcompStatus_t));
    if (err != cudaSuccess) goto fail;

    /* nvCOMP temp scratch */
    {
        nvcompBatchedDeflateDecompressOpts_t opts =
            nvcompBatchedDeflateDecompressDefaultOpts;
        nvcompStatus_t ns = nvcompBatchedDeflateDecompressGetTempSizeAsync(
            GDS_MAX_CHUNKS,
            GDS_BGZF_DECOMP_MAX,
            opts,
            &mTempBytes,
            (size_t)GDS_MAX_CHUNKS * GDS_BGZF_DECOMP_MAX);
        if (ns != nvcompSuccess) goto fail;
    }
    if (mTempBytes > 0) {
        err = cudaMalloc(&d_temp, mTempBytes);
        if (err != cudaSuccess) goto fail;
    }

    /* Pre-populate static pointer and buf-size arrays */
    {
        void*  tmp_comp[GDS_MAX_CHUNKS];
        void*  tmp_decomp[GDS_MAX_CHUNKS];
        size_t tmp_buf[GDS_MAX_CHUNKS];
        for (int i = 0; i < GDS_MAX_CHUNKS; i++) {
            tmp_comp[i]  = d_compressed_flat + (size_t)i * GDS_BGZF_BLOCK_MAX;
            tmp_decomp[i]= (unsigned char*)d_decompressed
                           + (size_t)i * GDS_BGZF_DECOMP_MAX;
            tmp_buf[i]   = GDS_BGZF_DECOMP_MAX;
        }
        cudaMemcpy(d_comp_ptrs,   tmp_comp,
                   GDS_MAX_CHUNKS * sizeof(void*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_decomp_ptrs, tmp_decomp,
                   GDS_MAX_CHUNKS * sizeof(void*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_decomp_buf_bytes, tmp_buf,
                   GDS_MAX_CHUNKS * sizeof(size_t), cudaMemcpyHostToDevice);
    }

    /* ── FASTQ parsing buffers ─────────────────────────────────────── */
    {
        size_t max_reads = GDS_DECOMP_BUF_BYTES / 8 + 1;
        err = cudaMalloc(&d_descs,     max_reads * sizeof(GpuReadDescriptor));
        if (err != cudaSuccess) goto fail;
        err = cudaMalloc(&d_readCount, sizeof(uint32_t));
        if (err != cudaSuccess) goto fail;
    }

    /* ── Per-read statistics ───────────────────────────────────────── */
    {
        size_t max_reads = GDS_DECOMP_BUF_BYTES / 8 + 1;
        err = cudaMalloc(&d_stats, max_reads * sizeof(ReadStatistics));
        if (err != cudaSuccess) goto fail;
        err = cudaHostAlloc(&h_stats,
                            max_reads * sizeof(ReadStatistics),
                            cudaHostAllocDefault);
        if (err != cudaSuccess) goto fail;
    }

    /* ── Pinned host buffer for text D2H ───────────────────────────── */
    err = cudaHostAlloc(&h_pinned_text, GDS_DECOMP_BUF_BYTES,
                        cudaHostAllocDefault);
    if (err != cudaSuccess) goto fail;

    mValid = true;
    return;

fail:
    fprintf(stderr, "[gds_pipeline] allocate() failed\n");
    release();
}

void GdsPipeline::release() {
    if (d_raw) {
        cuFileBufDeregister(d_raw);
        cudaFree(d_raw);    d_raw = nullptr;
    }
    if (h_header_buf)       { cudaFreeHost(h_header_buf);       h_header_buf = nullptr; }
    if (d_compressed_flat)  { cudaFree(d_compressed_flat);      d_compressed_flat = nullptr; }
    if (d_decompressed)     { cudaFree(d_decompressed);         d_decompressed = nullptr; }
    if (d_comp_ptrs)        { cudaFree(d_comp_ptrs);            d_comp_ptrs = nullptr; }
    if (d_decomp_ptrs)      { cudaFree(d_decomp_ptrs);          d_decomp_ptrs = nullptr; }
    if (d_comp_bytes)       { cudaFree(d_comp_bytes);            d_comp_bytes = nullptr; }
    if (d_decomp_buf_bytes) { cudaFree(d_decomp_buf_bytes);     d_decomp_buf_bytes = nullptr; }
    if (d_decomp_act_bytes) { cudaFree(d_decomp_act_bytes);     d_decomp_act_bytes = nullptr; }
    if (d_statuses)         { cudaFree(d_statuses);              d_statuses = nullptr; }
    if (d_temp)             { cudaFree(d_temp);                  d_temp = nullptr; }
    if (d_descs)            { cudaFree(d_descs);                 d_descs = nullptr; }
    if (d_readCount)        { cudaFree(d_readCount);             d_readCount = nullptr; }
    if (d_stats)            { cudaFree(d_stats);                 d_stats = nullptr; }
    if (h_stats)            { cudaFreeHost(h_stats);             h_stats = nullptr; }
    if (h_pinned_text)      { cudaFreeHost(h_pinned_text);       h_pinned_text = nullptr; }
    if (mStream)            { cudaStreamDestroy(mStream);        mStream = nullptr; }
}

/* ═════════════════════════════════════════════════════════════════════════
 * File management
 * ═════════════════════════════════════════════════════════════════════════*/

int GdsPipeline::open(const char* path) {
    if (!mValid || !mDriverOpen) return -1;
    if (mFileOpen) close();

    /* Try O_DIRECT first for true GDS DMA on supported NVMe devices.
     * Fall back to regular open if O_DIRECT fails (e.g., software RAID,
     * NFS, tmpfs). cuFileRead works in compat mode without O_DIRECT. */
    mFd = ::open(path, O_RDONLY | O_DIRECT);
    if (mFd < 0)
        mFd = ::open(path, O_RDONLY);
    if (mFd < 0) {
        fprintf(stderr, "[gds_pipeline] cannot open '%s'\n", path);
        return -1;
    }

    /* Get file size */
    struct stat st;
    if (fstat(mFd, &st) < 0) {
        fprintf(stderr, "[gds_pipeline] fstat failed on '%s'\n", path);
        ::close(mFd); mFd = -1;
        return -1;
    }
    mFileSize = st.st_size;

    /* Register with cuFile */
    CUfileDescr_t desc{};
    desc.handle.fd = mFd;
    desc.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    CUfileError_t cst = cuFileHandleRegister(&mCuFileHandle, &desc);
    if (cst.err != CU_FILE_SUCCESS) {
        fprintf(stderr,
                "[gds_pipeline] cuFileHandleRegister failed (err=%d)\n",
                (int)cst.err);
        ::close(mFd); mFd = -1;
        return -1;
    }

    mFileOpen   = true;
    mFileOffset = 0;
    mEof        = false;
    mReadCount  = 0;
    mDecompBytes = 0;

    return 0;
}

void GdsPipeline::close() {
    if (mFileOpen) {
        cuFileHandleDeregister(mCuFileHandle);
        ::close(mFd);
        mFd       = -1;
        mFileOpen = false;
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * Core pipeline: readAndDecompress
 *
 * Steps:
 *   1. GDS DMA: read raw BGZF blocks from NVMe directly into d_raw
 *   2. D2H: copy raw data to pinned host memory
 *   3. Parse BGZF block boundaries on CPU
 *   4. Strip BGZF headers on CPU, upload DEFLATE payloads to GPU
 *   5. nvCOMP batch DEFLATE decompression (in sub-batches on GPU)
 *   6. Concatenate decompressed blocks → contiguous FASTQ text on GPU
 * ═════════════════════════════════════════════════════════════════════════*/

/* Maximum chunks per nvCOMP decompression call.  Empirically, the nvCOMP
 * DEFLATE decompressor can hang with large batch counts (>384) when the
 * cuFile driver has been loaded.  256 is conservative and still efficient. */
static const int NVCOMP_SUB_BATCH = 256;

ssize_t GdsPipeline::readAndDecompress() {
    if (!mFileOpen || !mValid) return -1;
    if (mEof) return 0;

    cudaSetDevice(mDevice);

    /* ── 1. GDS DMA: NVMe → d_raw ─────────────────────────────────── */
    size_t bytes_remaining = (size_t)(mFileSize - mFileOffset);
    if (bytes_remaining == 0) { mEof = true; return 0; }

    /* Read at most 256 * 64 KB = 16 MB of compressed data per batch,
     * matching the caller's 16 MB buffer size. */
    static const size_t GDS_READ_BUDGET = (size_t)NVCOMP_SUB_BATCH * GDS_BGZF_BLOCK_MAX;
    size_t read_size = std::min(bytes_remaining, GDS_READ_BUDGET);

    ssize_t dma_bytes = cuFileRead(mCuFileHandle, d_raw, read_size,
                                   mFileOffset, (off_t)0);
    if (dma_bytes < 0) {
        fprintf(stderr, "[gds_pipeline] cuFileRead failed (ret=%zd)\n",
                dma_bytes);
        return -1;
    }
    if (dma_bytes == 0) { mEof = true; return 0; }

    /* ── 2. D2H: d_raw → pinned host buffer ───────────────────────── */
    unsigned char* h_raw = (unsigned char*)h_pinned_text;
    cudaMemcpy(h_raw, d_raw, (size_t)dma_bytes, cudaMemcpyDeviceToHost);

    /* ── 3. Parse BGZF block boundaries on CPU ────────────────────── */
    uint32_t h_offsets[GDS_MAX_CHUNKS];
    uint32_t h_sizes[GDS_MAX_CHUNKS];
    int      num_chunks = 0;
    size_t   pos = 0;

    /* Limit to NVCOMP_SUB_BATCH blocks per call so the decompressed output
     * fits within the caller's 16 MB buffer (FQ_BUF_SIZE).
     * 256 * 64 KB = 16 MB. */
    int max_chunks = NVCOMP_SUB_BATCH;

    while (pos < (size_t)dma_bytes && num_chunks < max_chunks) {
        size_t avail = (size_t)dma_bytes - pos;
        size_t block_size = parseBgzfBlockSize(h_raw + pos, avail);
        if (block_size == 0) break;
        if (pos + block_size > (size_t)dma_bytes) break;

        /* EOF marker: empty block (28 bytes) */
        if (block_size == 28) {
            pos += block_size;
            if (pos >= (size_t)dma_bytes &&
                mFileOffset + (off_t)pos >= mFileSize)
                mEof = true;
            break;
        }

        h_offsets[num_chunks] = (uint32_t)pos;
        h_sizes[num_chunks]   = (uint32_t)block_size;
        num_chunks++;
        pos += block_size;
    }

    if (num_chunks == 0) {
        mEof = true;
        return 0;
    }

    /* Advance file cursor to the end of consumed blocks */
    mFileOffset += (off_t)pos;

    /* ── 4. Strip BGZF headers on CPU, upload DEFLATE payloads ────── */
    {
        size_t deflate_sizes[GDS_MAX_CHUNKS];

        for (int i = 0; i < num_chunks; i++) {
            size_t gz_sz = h_sizes[i];
            int def_sz = (int)gz_sz - GDS_BGZF_HDR - GDS_BGZF_TRL;
            if (def_sz < 0) def_sz = 0;
            deflate_sizes[i] = (size_t)def_sz;

            unsigned char* src = h_raw + h_offsets[i] + GDS_BGZF_HDR;
            cudaMemcpy((unsigned char*)d_compressed_flat +
                           (size_t)i * GDS_BGZF_BLOCK_MAX,
                       src, (size_t)def_sz, cudaMemcpyHostToDevice);
        }

        cudaMemcpy(d_comp_bytes, deflate_sizes,
                   num_chunks * sizeof(size_t), cudaMemcpyHostToDevice);
    }

    /* ── 5. nvCOMP batch DEFLATE decompression ───────────────────────
     * Decompress all chunks in a single call.  With ≤256 chunks this
     * stays within the empirically safe limit for nvCOMP. */
    {
        nvcompBatchedDeflateDecompressOpts_t opts =
            nvcompBatchedDeflateDecompressDefaultOpts;

        nvcompStatus_t ns = nvcompBatchedDeflateDecompressAsync(
            (const void* const*)d_comp_ptrs,
            d_comp_bytes,
            d_decomp_buf_bytes,
            d_decomp_act_bytes,
            num_chunks,
            d_temp,
            mTempBytes,
            (void* const*)d_decomp_ptrs,
            opts,
            d_statuses,
            mStream);
        if (ns != nvcompSuccess) {
            fprintf(stderr,
                    "[gds_pipeline] nvCOMP decompress failed: %d\n", (int)ns);
            return -1;
        }
        cudaStreamSynchronize(mStream);
    }

    /* ── 6. Build prefix-sum and concatenate decompressed blocks ───── */
    {
        size_t h_act[GDS_MAX_CHUNKS];
        cudaMemcpy(h_act, d_decomp_act_bytes,
                   num_chunks * sizeof(size_t), cudaMemcpyDeviceToHost);

        size_t h_pfx[GDS_MAX_CHUNKS];
        size_t running = 0;
        for (int i = 0; i < num_chunks; i++) {
            h_pfx[i] = running;
            running += h_act[i];
        }
        mDecompBytes = running;

        size_t* d_out_offsets = nullptr;
        cudaMalloc(&d_out_offsets, num_chunks * sizeof(size_t));
        cudaMemcpy(d_out_offsets, h_pfx,
                   num_chunks * sizeof(size_t), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks  = (num_chunks + threads - 1) / threads;
        /* Concatenate into d_raw (no longer needed after step 2) to avoid
         * in-place overlap — d_decompressed has stride-65536 layout while
         * the concat output uses prefix-sum offsets. */
        bgzf_concat_kernel<<<blocks, threads, 0, mStream>>>(
            (const unsigned char*)d_decompressed,
            d_decomp_act_bytes,
            d_out_offsets,
            num_chunks,
            (char*)d_raw);
        cudaStreamSynchronize(mStream);

        /* Copy the tightly-packed result back to d_decompressed so the
         * accessor decompressedDevicePtr() returns valid data. */
        cudaMemcpy(d_decompressed, d_raw, mDecompBytes,
                   cudaMemcpyDeviceToDevice);

        cudaFree(d_out_offsets);
    }

    return (ssize_t)mDecompBytes;
}

/* ═════════════════════════════════════════════════════════════════════════
 * GPU kernel: descriptor-based per-read statistics
 *
 * Uses GpuReadDescriptor to index into the concatenated FASTQ text buffer.
 * Each warp processes one read cooperatively (same pattern as
 * compute_read_stats_warp_kernel but using offsets instead of pointers).
 * ═════════════════════════════════════════════════════════════════════════*/

#define STATS_BLOCK_SIZE 256
#define STATS_WARP_SIZE   32
#define STATS_RPB (STATS_BLOCK_SIZE / STATS_WARP_SIZE)

__global__ void gds_read_stats_kernel(
    const char*              __restrict__ d_text,
    const GpuReadDescriptor* __restrict__ d_descs,
    uint32_t                              num_reads,
    char                                  qual_threshold,
    int                                   trim_window_size,
    ReadStatistics*                       d_stats)
{
    int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int read_id       = global_thread / STATS_WARP_SIZE;
    int lane          = global_thread % STATS_WARP_SIZE;

    if ((uint32_t)read_id >= num_reads) return;

    const GpuReadDescriptor& desc = d_descs[read_id];
    const char* seq  = d_text + desc.seq_offset;
    const char* qual = d_text + desc.qual_offset;
    int read_len     = (int)desc.seq_len;

    /* Phase 1: parallel base stats (stride-32) */
    int n_bases    = 0;
    int low_qual   = 0;
    int total_qual = 0;

    for (int i = lane; i < read_len; i += STATS_WARP_SIZE) {
        char c = seq[i];
        char q = qual[i];
        if (c == 'N' || c == 'n') n_bases++;
        if (q < qual_threshold)   low_qual++;
        total_qual += (int)(unsigned char)(q - 33);
    }

    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        n_bases    += __shfl_down_sync(mask, n_bases,    offset);
        low_qual   += __shfl_down_sync(mask, low_qual,   offset);
        total_qual += __shfl_down_sync(mask, total_qual, offset);
    }

    /* Phase 2: trim + polyG (lane 0 only) */
    if (lane == 0) {
        d_stats[read_id].total_bases    = read_len;
        d_stats[read_id].n_bases        = n_bases;
        d_stats[read_id].low_qual_bases = low_qual;
        d_stats[read_id].total_quality  = total_qual;

        int trim_start = 0;
        int trim_end   = read_len;

        /* Single-base quality scan */
        for (int i = 0; i < read_len; i++) {
            if (qual[i] >= qual_threshold) { trim_start = i; break; }
        }
        for (int i = read_len - 1; i >= trim_start; i--) {
            if (qual[i] >= qual_threshold) { trim_end = i + 1; break; }
        }

        d_stats[read_id].trim_start = trim_start;
        d_stats[read_id].trim_end   = trim_end;

        /* PolyG detection */
        int run = 0, trim_pos = read_len;
        for (int i = read_len - 1; i >= 0; i--) {
            if (seq[i] == 'G' || seq[i] == 'g') {
                run++;
                if (run >= 10) trim_pos = i;
            } else {
                if (run >= 10) break;
                run = 0;
            }
        }
        d_stats[read_id].polyG_trim_pos = (trim_pos < read_len) ? trim_pos : -1;
    }
}

/* ═════════════════════════════════════════════════════════════════════════
 * computeStats
 * ═════════════════════════════════════════════════════════════════════════*/

int GdsPipeline::computeStats(char qual_threshold,
                              int  trim_window_size,
                              std::vector<ReadStatistics>& out_stats) {
    if (!mValid || mReadCount == 0) return -1;

    cudaSetDevice(mDevice);

    /* Zero-init stats on device */
    cudaMemsetAsync(d_stats, 0,
                    mReadCount * sizeof(ReadStatistics), mStream);

    /* Launch descriptor-based stats kernel */
    int grid = ((int)mReadCount * STATS_WARP_SIZE + STATS_BLOCK_SIZE - 1)
               / STATS_BLOCK_SIZE;

    gds_read_stats_kernel<<<grid, STATS_BLOCK_SIZE, 0, mStream>>>(
        (const char*)d_decompressed,
        d_descs,
        mReadCount,
        qual_threshold,
        trim_window_size,
        d_stats);

    /* D2H: copy only the stats (compact) */
    cudaMemcpyAsync(h_stats, d_stats,
                    mReadCount * sizeof(ReadStatistics),
                    cudaMemcpyDeviceToHost, mStream);
    cudaStreamSynchronize(mStream);

    out_stats.resize(mReadCount);
    memcpy(out_stats.data(), h_stats,
           mReadCount * sizeof(ReadStatistics));

    return 0;
}

/* ═════════════════════════════════════════════════════════════════════════
 * readBatch — convenience wrapper
 * ═════════════════════════════════════════════════════════════════════════*/

int GdsPipeline::readBatch(char qual_threshold,
                           int  trim_window_size,
                           std::vector<ReadStatistics>& out_stats,
                           char* out_text,
                           size_t out_text_capacity) {
    ssize_t decomp = readAndDecompress();
    if (decomp <= 0) return (int)decomp;

    /* Copy decompressed text to host if requested */
    if (out_text && out_text_capacity > 0) {
        size_t copy_sz = std::min(mDecompBytes, out_text_capacity);
        cudaMemcpyAsync(h_pinned_text, d_decompressed, copy_sz,
                        cudaMemcpyDeviceToHost, mStream);
        cudaStreamSynchronize(mStream);
        memcpy(out_text, h_pinned_text, copy_sz);
    }

    int rc = computeStats(qual_threshold, trim_window_size, out_stats);
    if (rc != 0) return -1;

    return (int)mReadCount;
}

#endif /* HAVE_CUDA && HAVE_NVCOMP && HAVE_GDS */
