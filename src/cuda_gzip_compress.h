/*
 * cuda_gzip_compress.h — GPU-accelerated BGZF compressor using NVIDIA nvCOMP.
 *
 * Companion of cuda_gzip.h (decompressor).  Each call to compressBgzf()
 * batches an entire ≤4 MiB writer buffer into a single nvCOMP DEFLATE batch
 * (~64 chunks of ≤65280 bytes each), then frames each compressed chunk with
 * a 18-byte BGZF gzip header, a host-computed CRC32, and a 4-byte ISIZE,
 * producing a byte-exact BGZF stream that decompresses to the input.
 *
 * Designed as a process-wide singleton: all writer compress workers share one
 * instance, serialised by an internal mutex.  Per-call cost is dominated by
 * GPU work, so the mutex is short and the GPU stays busy at high T.
 *
 * Opt-in by setting FASTP_GPU_GZIP=1 in the environment.  When the singleton
 * fails to initialise (no GPU, nvCOMP error), the caller falls back to the
 * CPU libdeflate path transparently.
 */
#pragma once
#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP)

#include <cuda_runtime.h>
#include <nvcomp/deflate.h>
#include <cstddef>
#include <cstdint>
#include <mutex>

// Match BGZF_BLOCK_PAYLOAD in bgzf_writer.h (≤ 65280 bytes per chunk).
static const size_t CUDA_BGZF_PAYLOAD    = 65280;
// Capacity of the device batch buffer.  64 chunks ≈ 4 MiB (one writer buffer).
// 128 chunks gives headroom for slightly larger inputs.
static const int    CUDA_BGZF_MAX_CHUNKS = 128;

class CudaGzipCompressor {
public:
    CudaGzipCompressor();
    ~CudaGzipCompressor();

    CudaGzipCompressor(const CudaGzipCompressor&) = delete;
    CudaGzipCompressor& operator=(const CudaGzipCompressor&) = delete;

    bool valid() const { return mValid; }

    // Compress src (any size ≤ CUDA_BGZF_MAX_CHUNKS*CUDA_BGZF_PAYLOAD) into a
    // BGZF stream written to dst.  Returns bytes written, or 0 on error.
    // Thread-safe: internally serialised with a mutex.
    size_t compressBgzf(const unsigned char* src, size_t srcLen,
                        unsigned char* dst, size_t dstCap);

    // Process-wide singleton.  Returns nullptr if construction failed.
    static CudaGzipCompressor* shared();

private:
    bool allocate();
    void freeAll();

    bool                                mValid;
    int                                 mDevice;
    cudaStream_t                        mStream;
    std::mutex                          mMutex;

    nvcompBatchedDeflateCompressOpts_t  mOpts;
    size_t                              mMaxOutChunk;
    size_t                              mTempBytes;

    // Device flat buffers
    unsigned char*  d_in_flat;        // [MAX_CHUNKS * CUDA_BGZF_PAYLOAD]
    unsigned char*  d_out_flat;       // [MAX_CHUNKS * mMaxOutChunk]
    void**          d_in_ptrs;
    void**          d_out_ptrs;
    size_t*         d_in_sizes;
    size_t*         d_out_sizes;
    nvcompStatus_t* d_statuses;
    void*           d_temp;

    // Pinned host staging
    unsigned char*  h_in_flat;
    unsigned char*  h_out_flat;
    size_t*         h_in_sizes;
    size_t*         h_out_sizes;
};

#endif // HAVE_CUDA && HAVE_NVCOMP
