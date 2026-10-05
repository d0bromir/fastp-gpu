/*
 * cuda_gzip_compress.cu — GPU-accelerated BGZF compressor.
 *
 * Splits @src into ≤65280-byte chunks, copies them to pinned host then to a
 * flat device buffer, dispatches a single nvcompBatchedDeflateCompressAsync
 * call, copies compressed chunks back, computes CRC32 per chunk on the host
 * (libdeflate), and emits a complete BGZF stream byte-compatible with
 * bgzf_write_block() in bgzf_writer.cpp.
 */
#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP)

#include "cuda_gzip_compress.h"
#include "libdeflate.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <new>

#define FASTP_CUDA_TAG "cuda_gzip_compress"
#include "cuda_error_check.h"
#define CUDA_CHECK(call)   FASTP_CUDA_CHECK(call)
#define NVCOMP_CHECK(call) FASTP_NVCOMP_CHECK(call)

// ─── BGZF framing constants (must match bgzf_writer.cpp) ─────────────────
static const int BGZF_HEADER_BYTES  = 18;
static const int BGZF_TRAILER_BYTES =  8;

static inline void put_u16le(unsigned char* p, uint16_t v) {
    p[0] = (unsigned char)(v       & 0xff);
    p[1] = (unsigned char)((v >> 8) & 0xff);
}
static inline void put_u32le(unsigned char* p, uint32_t v) {
    p[0] = (unsigned char)(v        & 0xff);
    p[1] = (unsigned char)((v >>  8) & 0xff);
    p[2] = (unsigned char)((v >> 16) & 0xff);
    p[3] = (unsigned char)((v >> 24) & 0xff);
}

// ─── Ctor / dtor ─────────────────────────────────────────────────────────
CudaGzipCompressor::CudaGzipCompressor()
    : mValid(false), mDevice(0), mStream(nullptr),
      mOpts(nvcompBatchedDeflateCompressDefaultOpts),
      mMaxOutChunk(0), mTempBytes(0),
      d_in_flat(nullptr), d_out_flat(nullptr),
      d_in_ptrs(nullptr), d_out_ptrs(nullptr),
      d_in_sizes(nullptr), d_out_sizes(nullptr),
      d_statuses(nullptr), d_temp(nullptr),
      h_in_flat(nullptr), h_out_flat(nullptr),
      h_in_sizes(nullptr), h_out_sizes(nullptr)
{
    if (const char* lvl = std::getenv("FASTP_GPU_GZIP_LEVEL")) {
        int v = std::atoi(lvl);
        if (v >= 0 && v <= 2) mOpts.algorithm = v;
    }
    if (cudaSetDevice(mDevice) != cudaSuccess) return;
    if (allocate()) mValid = true;
    else freeAll();
}

CudaGzipCompressor::~CudaGzipCompressor() {
    freeAll();
}

bool CudaGzipCompressor::allocate() {
    // 1. Query max output chunk size.
    nvcompStatus_t ns = nvcompBatchedDeflateCompressGetMaxOutputChunkSize(
        CUDA_BGZF_PAYLOAD, mOpts, &mMaxOutChunk);
    if (ns != nvcompSuccess) {
        fprintf(stderr, "[cuda_gzip_compress] GetMaxOutputChunkSize failed: %d\n", ns);
        return false;
    }
    // 2. Query temp scratch size.
    ns = nvcompBatchedDeflateCompressGetTempSizeAsync(
        CUDA_BGZF_MAX_CHUNKS, CUDA_BGZF_PAYLOAD, mOpts,
        &mTempBytes,
        (size_t)CUDA_BGZF_MAX_CHUNKS * CUDA_BGZF_PAYLOAD);
    if (ns != nvcompSuccess) {
        fprintf(stderr, "[cuda_gzip_compress] GetTempSizeAsync failed: %d\n", ns);
        return false;
    }

    // 3. Device buffers.
    if (cudaMalloc(&d_in_flat,   (size_t)CUDA_BGZF_MAX_CHUNKS * CUDA_BGZF_PAYLOAD) != cudaSuccess ||
        cudaMalloc(&d_out_flat,  (size_t)CUDA_BGZF_MAX_CHUNKS * mMaxOutChunk)      != cudaSuccess ||
        cudaMalloc(&d_in_ptrs,   CUDA_BGZF_MAX_CHUNKS * sizeof(void*))             != cudaSuccess ||
        cudaMalloc(&d_out_ptrs,  CUDA_BGZF_MAX_CHUNKS * sizeof(void*))             != cudaSuccess ||
        cudaMalloc(&d_in_sizes,  CUDA_BGZF_MAX_CHUNKS * sizeof(size_t))            != cudaSuccess ||
        cudaMalloc(&d_out_sizes, CUDA_BGZF_MAX_CHUNKS * sizeof(size_t))            != cudaSuccess ||
        cudaMalloc(&d_statuses,  CUDA_BGZF_MAX_CHUNKS * sizeof(nvcompStatus_t))    != cudaSuccess) {
        fprintf(stderr, "[cuda_gzip_compress] cudaMalloc failed\n");
        return false;
    }
    if (mTempBytes > 0 && cudaMalloc(&d_temp, mTempBytes) != cudaSuccess) {
        fprintf(stderr, "[cuda_gzip_compress] cudaMalloc temp failed\n");
        return false;
    }

    // 4. Pinned host buffers.
    if (cudaHostAlloc(&h_in_flat,
                      (size_t)CUDA_BGZF_MAX_CHUNKS * CUDA_BGZF_PAYLOAD,
                      cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&h_out_flat,
                      (size_t)CUDA_BGZF_MAX_CHUNKS * mMaxOutChunk,
                      cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&h_in_sizes,
                      CUDA_BGZF_MAX_CHUNKS * sizeof(size_t),
                      cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&h_out_sizes,
                      CUDA_BGZF_MAX_CHUNKS * sizeof(size_t),
                      cudaHostAllocDefault) != cudaSuccess) {
        fprintf(stderr, "[cuda_gzip_compress] cudaHostAlloc failed\n");
        return false;
    }

    // 5. Stream.
    if (cudaStreamCreate(&mStream) != cudaSuccess) {
        fprintf(stderr, "[cuda_gzip_compress] cudaStreamCreate failed\n");
        return false;
    }

    // 6. Pre-populate the static device pointer arrays.
    {
        void* tmp_in[CUDA_BGZF_MAX_CHUNKS];
        void* tmp_out[CUDA_BGZF_MAX_CHUNKS];
        for (int i = 0; i < CUDA_BGZF_MAX_CHUNKS; i++) {
            tmp_in[i]  = (void*)(d_in_flat  + (size_t)i * CUDA_BGZF_PAYLOAD);
            tmp_out[i] = (void*)(d_out_flat + (size_t)i * mMaxOutChunk);
        }
        if (cudaMemcpy(d_in_ptrs,  tmp_in,  CUDA_BGZF_MAX_CHUNKS * sizeof(void*),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_out_ptrs, tmp_out, CUDA_BGZF_MAX_CHUNKS * sizeof(void*),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            fprintf(stderr, "[cuda_gzip_compress] ptr array upload failed\n");
            return false;
        }
    }
    return true;
}

void CudaGzipCompressor::freeAll() {
    if (mStream)     { cudaStreamDestroy(mStream); mStream = nullptr; }
    if (d_in_flat)   { cudaFree(d_in_flat);   d_in_flat = nullptr; }
    if (d_out_flat)  { cudaFree(d_out_flat);  d_out_flat = nullptr; }
    if (d_in_ptrs)   { cudaFree(d_in_ptrs);   d_in_ptrs = nullptr; }
    if (d_out_ptrs)  { cudaFree(d_out_ptrs);  d_out_ptrs = nullptr; }
    if (d_in_sizes)  { cudaFree(d_in_sizes);  d_in_sizes = nullptr; }
    if (d_out_sizes) { cudaFree(d_out_sizes); d_out_sizes = nullptr; }
    if (d_statuses)  { cudaFree(d_statuses);  d_statuses = nullptr; }
    if (d_temp)      { cudaFree(d_temp);      d_temp = nullptr; }
    if (h_in_flat)   { cudaFreeHost(h_in_flat);   h_in_flat = nullptr; }
    if (h_out_flat)  { cudaFreeHost(h_out_flat);  h_out_flat = nullptr; }
    if (h_in_sizes)  { cudaFreeHost(h_in_sizes);  h_in_sizes = nullptr; }
    if (h_out_sizes) { cudaFreeHost(h_out_sizes); h_out_sizes = nullptr; }
    mValid = false;
}

// Compress one writer buffer (up to MAX_CHUNKS*PAYLOAD bytes).
// Output: concatenation of complete BGZF gzip members.
size_t CudaGzipCompressor::compressBgzf(const unsigned char* src, size_t srcLen,
                                        unsigned char* dst, size_t dstCap)
{
    if (!mValid || srcLen == 0) return 0;

    // Split into chunks of CUDA_BGZF_PAYLOAD bytes.
    size_t num_chunks = (srcLen + CUDA_BGZF_PAYLOAD - 1) / CUDA_BGZF_PAYLOAD;
    if (num_chunks == 0 || num_chunks > (size_t)CUDA_BGZF_MAX_CHUNKS) {
        // Caller should chunk larger inputs.
        return 0;
    }

    std::lock_guard<std::mutex> lk(mMutex);
    cudaSetDevice(mDevice);

    // ── 1. Stage chunks into pinned host buffer (uniform stride). ────────
    for (size_t i = 0; i < num_chunks; i++) {
        size_t off = i * CUDA_BGZF_PAYLOAD;
        size_t sz  = (i + 1 == num_chunks) ? (srcLen - off) : CUDA_BGZF_PAYLOAD;
        h_in_sizes[i] = sz;
        memcpy(h_in_flat + i * CUDA_BGZF_PAYLOAD, src + off, sz);
    }

    // ── 2. Upload input data + sizes. ────────────────────────────────────
    // Copy the whole packed flat buffer; nvCOMP only reads h_in_sizes[i] of each slot.
    if (cudaMemcpyAsync(d_in_flat, h_in_flat,
                        num_chunks * CUDA_BGZF_PAYLOAD,
                        cudaMemcpyHostToDevice, mStream) != cudaSuccess) {
        return 0;
    }
    if (cudaMemcpyAsync(d_in_sizes, h_in_sizes,
                        num_chunks * sizeof(size_t),
                        cudaMemcpyHostToDevice, mStream) != cudaSuccess) {
        return 0;
    }

    // ── 3. Compress. ─────────────────────────────────────────────────────
    nvcompStatus_t ns = nvcompBatchedDeflateCompressAsync(
        (const void* const*)d_in_ptrs,
        d_in_sizes,
        CUDA_BGZF_PAYLOAD,
        num_chunks,
        d_temp,
        mTempBytes,
        (void* const*)d_out_ptrs,
        d_out_sizes,
        mOpts,
        d_statuses,
        mStream);
    if (ns != nvcompSuccess) {
        fprintf(stderr, "[cuda_gzip_compress] CompressAsync failed: %d\n", ns);
        return 0;
    }

    // ── 4. Download compressed sizes, then compressed data. ──────────────
    if (cudaMemcpyAsync(h_out_sizes, d_out_sizes,
                        num_chunks * sizeof(size_t),
                        cudaMemcpyDeviceToHost, mStream) != cudaSuccess) {
        return 0;
    }
    // Copy the whole flat output buffer.  Sub-optimal vs strided copy but
    // simpler; for 64 chunks × ~32 KiB compressed ≈ 2 MiB, fine over PCIe.
    if (cudaMemcpyAsync(h_out_flat, d_out_flat,
                        num_chunks * mMaxOutChunk,
                        cudaMemcpyDeviceToHost, mStream) != cudaSuccess) {
        return 0;
    }
    if (cudaStreamSynchronize(mStream) != cudaSuccess) return 0;

    // ── 5. Frame each chunk into BGZF + write to dst. ────────────────────
    size_t off = 0;
    for (size_t i = 0; i < num_chunks; i++) {
        size_t in_sz   = h_in_sizes[i];
        size_t def_sz  = h_out_sizes[i];
        size_t total   = BGZF_HEADER_BYTES + def_sz + BGZF_TRAILER_BYTES;
        if (off + total > dstCap) return 0;
        if (total - 1 > 0xffff)  return 0;  // BSIZE overflow

        unsigned char* p = dst + off;
        p[0]  = 0x1f; p[1]  = 0x8b;
        p[2]  = 0x08;                       // method = DEFLATE
        p[3]  = 0x04;                       // FLG = FEXTRA
        p[4]  = p[5] = p[6] = p[7] = 0;     // MTIME
        p[8]  = 0x00;                       // XFL
        p[9]  = 0xff;                       // OS = unknown
        p[10] = 0x06; p[11] = 0x00;         // XLEN = 6
        p[12] = 'B';  p[13] = 'C';
        p[14] = 0x02; p[15] = 0x00;         // SLEN = 2

        // Copy compressed payload from staging.
        memcpy(p + BGZF_HEADER_BYTES,
               h_out_flat + i * mMaxOutChunk,
               def_sz);

        // CRC32 over the original uncompressed chunk (host-side).
        uint32_t crc = libdeflate_crc32(0, src + i * CUDA_BGZF_PAYLOAD, in_sz);
        put_u32le(p + BGZF_HEADER_BYTES + def_sz,     crc);
        put_u32le(p + BGZF_HEADER_BYTES + def_sz + 4, (uint32_t)in_sz);

        // BSIZE = total - 1
        put_u16le(p + 16, (uint16_t)(total - 1));

        off += total;
    }
    return off;
}

// ─── Singleton accessor ──────────────────────────────────────────────────
CudaGzipCompressor* CudaGzipCompressor::shared() {
    static std::once_flag once;
    static CudaGzipCompressor* inst = nullptr;
    std::call_once(once, []() {
        CudaGzipCompressor* c = new (std::nothrow) CudaGzipCompressor();
        if (c && c->valid()) inst = c;
        else {
            delete c;
            inst = nullptr;
            fprintf(stderr, "[cuda_gzip_compress] singleton init failed; "
                            "falling back to CPU libdeflate.\n");
        }
    });
    return inst;
}

#endif // HAVE_CUDA && HAVE_NVCOMP
