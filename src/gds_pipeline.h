/*
 * gds_pipeline.h — GPU-Direct Storage end-to-end pipeline
 *
 * Implements a nearly CPU-free decompression-to-statistics pipeline:
 *
 *   NVMe ──DMA──▶ GPU memory (cuFile)
 *                    │
 *                    ▼
 *            BGZF block parsing (GPU kernel)
 *                    │
 *                    ▼
 *            DEFLATE decompress (nvCOMP on GPU)
 *                    │
 *                    ▼
 *            FASTQ record parsing (GPU kernel)
 *                    │
 *                    ▼
 *            Per-read statistics (GPU kernel)
 *                    │
 *                    ▼
 *             ReadStatistics[] ──D2H──▶ CPU
 *
 * CPU involvement is limited to orchestrating kernel launches and
 * consuming the final ReadStatistics array.  All intermediate data
 * lives entirely in GPU memory, eliminating H2D/D2H bounce buffers.
 *
 * Requirements:
 *   - CUDA 11.4+, nvCOMP, cuFile (libcufile.so)
 *   - GDS-capable driver and NVMe/Lustre/GPFS filesystem
 *   - Build with: make WITH_CUDA=1 WITH_NVCOMP=1 WITH_GDS=1
 *
 * When any prerequisite is missing, a no-op stub is compiled instead.
 */
#pragma once

#include "cuda_stats_wrapper.h"   /* ReadStatistics */

#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP) && defined(HAVE_GDS)

#include <cuda_runtime.h>
#include <cufile.h>
#include <nvcomp/deflate.h>
#include <cstddef>
#include <cstdint>
#include <sys/types.h>
#include "cuda_stats.h"           /* GpuReadDescriptor, cuda_fastq_parse_device */

/* ───────────────────────────────────────────────────────────────────────── */
/* Tuning constants                                                         */
/* ───────────────────────────────────────────────────────────────────────── */
static const int    GDS_MAX_CHUNKS      = 4096;   /* max BGZF blocks/batch  */
static const size_t GDS_BGZF_BLOCK_MAX  = 65536;  /* max compressed bytes   */
static const size_t GDS_BGZF_DECOMP_MAX = 65536;  /* max decompressed bytes */
static const size_t GDS_RAW_BUF_BYTES   = (size_t)GDS_MAX_CHUNKS * GDS_BGZF_BLOCK_MAX;
static const size_t GDS_DECOMP_BUF_BYTES = (size_t)GDS_MAX_CHUNKS * GDS_BGZF_DECOMP_MAX;
/* BGZF header/trailer sizes for DEFLATE payload extraction */
static const int    GDS_BGZF_HDR  = 18;
static const int    GDS_BGZF_TRL  = 8;

class GdsPipeline {
public:
    /**
     * @param device  CUDA device index (default 0).
     */
    explicit GdsPipeline(int device = 0);
    ~GdsPipeline();

    /* non-copyable */
    GdsPipeline(const GdsPipeline&) = delete;
    GdsPipeline& operator=(const GdsPipeline&) = delete;

    /** True if all resources (cuFile, nvCOMP, device buffers) are ready. */
    bool valid() const { return mValid; }

    /** True if GDS is supported on this system. */
    static bool isAvailable();

    /**
     * Open a BGZF-compressed FASTQ file for GDS reads.
     * @return 0 on success, -1 on error.
     */
    int open(const char* path);

    /** Close the file. */
    void close();

    /**
     * Read the next batch of BGZF blocks directly from NVMe into GPU
     * memory, decompress them on the GPU, and parse FASTQ records.
     *
     * After a successful call:
     *   - decompressedDevicePtr() points to the decompressed FASTQ text
     *     on the GPU
     *   - decompressedBytes() returns the total decompressed byte count
     *   - descriptorDevicePtr() / readCount() give the parsed read
     *     descriptors
     *
     * @return Number of decompressed bytes (>0), 0 on EOF, -1 on error.
     */
    ssize_t readAndDecompress();

    /**
     * Run per-read statistics on the most recently decompressed batch.
     * Results are transferred to host memory.
     *
     * @param qual_threshold   Phred quality threshold (ASCII, e.g. '0'+15)
     * @param trim_window_size Sliding-window width for trimming
     * @param out_stats        Caller-provided output vector; resized to
     *                         readCount().
     * @return 0 on success, -1 on error.
     */
    int computeStats(char qual_threshold, int trim_window_size,
                     std::vector<ReadStatistics>& out_stats);

    /**
     * Convenience: readAndDecompress + computeStats in a single call.
     * Fills |out_stats| and puts the decompressed FASTQ text into
     * |out_text| (host memory) for downstream record extraction.
     *
     * @return Number of reads processed (>0), 0 on EOF, -1 on error.
     */
    int readBatch(char qual_threshold, int trim_window_size,
                  std::vector<ReadStatistics>& out_stats,
                  char* out_text, size_t out_text_capacity);

    /* ── Accessors for advanced callers ─────────────────────────────── */

    /** Decompressed FASTQ text on the GPU (valid after readAndDecompress). */
    const char* decompressedDevicePtr() const { return d_decompressed; }

    /** Total decompressed bytes in the current batch. */
    size_t decompressedBytes() const { return mDecompBytes; }

    /** GPU-resident read descriptors (valid after readAndDecompress). */
    const GpuReadDescriptor* descriptorDevicePtr() const { return d_descs; }

    /** Number of FASTQ reads in the current batch. */
    uint32_t readCount() const { return mReadCount; }

    /** True once the file has been fully consumed. */
    bool eof() const { return mEof; }

private:
    void allocate();
    void release();

    /* Parse a BGZF block header at host memory; return total block size or 0. */
    static size_t parseBgzfBlockSize(const unsigned char* data, size_t avail);

    int             mDevice;
    bool            mValid;
    bool            mEof;
    cudaStream_t    mStream;

    /* ── cuFile state ────────────────────────────────────────────────── */
    bool            mDriverOpen;
    int             mFd;
    CUfileHandle_t  mCuFileHandle;
    bool            mFileOpen;
    off_t           mFileOffset;   /* current read cursor in the file */
    off_t           mFileSize;     /* total file size in bytes */

    /* ── GPU raw-read buffer (GDS target) ───────────────────────────── */
    unsigned char*  d_raw;         /* [GDS_RAW_BUF_BYTES] */

    /* ── Small pinned host buffer for BGZF header parsing ──────────── */
    /* We DMA a window of raw data to a pinned host buffer only for
     * scanning BGZF block boundaries (tiny fraction of total I/O). */
    unsigned char*  h_header_buf;
    static const size_t HEADER_BUF_BYTES = 256 * 1024; /* 256 KB */

    /* ── nvCOMP decompression resources ─────────────────────────────── */
    unsigned char*  d_compressed_flat;    /* [GDS_MAX_CHUNKS * GDS_BGZF_BLOCK_MAX] */
    char*           d_decompressed;       /* [GDS_DECOMP_BUF_BYTES] — also FASTQ text */
    void**          d_comp_ptrs;
    void**          d_decomp_ptrs;
    size_t*         d_comp_bytes;
    size_t*         d_decomp_buf_bytes;
    size_t*         d_decomp_act_bytes;
    nvcompStatus_t* d_statuses;
    void*           d_temp;
    size_t          mTempBytes;
    size_t          mDecompBytes;

    /* ── FASTQ parsing results ─────────────────────────────────────── */
    GpuReadDescriptor* d_descs;
    uint32_t*       d_readCount;
    uint32_t        mReadCount;

    /* ── per-read statistics ───────────────────────────────────────── */
    ReadStatistics* d_stats;
    ReadStatistics* h_stats;          /* pinned host mirror */

    /* ── Pinned host staging for decompressed text D2H ─────────────── */
    char*           h_pinned_text;
};

#else /* ── Stub when prerequisites are missing ──────────────────────── */

#include <cstddef>
#include <sys/types.h>
#include <vector>

class GdsPipeline {
public:
    explicit GdsPipeline(int = 0) {}
    ~GdsPipeline() = default;
    bool valid() const           { return false; }
    static bool isAvailable()    { return false; }
    int  open(const char*)       { return -1; }
    void close()                 {}
    ssize_t readAndDecompress()  { return -1; }
    int  computeStats(char, int, std::vector<ReadStatistics>&) { return -1; }
    int  readBatch(char, int, std::vector<ReadStatistics>&,
                   char*, size_t) { return -1; }
    bool eof() const             { return true; }
    uint32_t readCount() const   { return 0; }
    size_t decompressedBytes() const { return 0; }
};

#endif /* HAVE_CUDA && HAVE_NVCOMP && HAVE_GDS */
