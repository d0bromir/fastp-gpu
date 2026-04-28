/*
 * cuda_gzip.cu — GPU-accelerated BGZF decompressor using NVIDIA nvCOMP.
 *
 * See cuda_gzip.h for full documentation.
 */
#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP)

#include "cuda_gzip.h"
#include "profiling.h"
#include <nvcomp/deflate.h>
#include <nvcomp/gzip.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

// Shared CUDA / nvCOMP error-check macros (see src/cuda_error_check.h).
#define FASTP_CUDA_TAG "cuda_gzip"
#include "cuda_error_check.h"
#define CUDA_CHECK(call)   FASTP_CUDA_CHECK(call)
#define NVCOMP_CHECK(call) FASTP_NVCOMP_CHECK(call)

// ─────────────────────────────────────────────────────────────────────────────
// BGZF block structure constants
//
// Byte offsets in a BGZF gzip member header:
//   0-1   1f 8b (gzip magic)
//   2     08    (DEFLATE method)
//   3     04    (FEXTRA flag)
//   4-7         mtime (ignored)
//   8           XFL  (ignored)
//   9           OS   (ignored)
//   10-11       XLEN (usually 6)
//   12    'B'   SI1
//   13    'C'   SI2
//   14-15 0x02 0x00  SLEN = 2
//   16-17       BSIZE (total block size minus 1, little-endian uint16)
//   18+         deflate payload
//   -8          CRC32 (4 bytes)
//   -4          ISIZE (4 bytes, uncompressed size mod 2^32)
// ─────────────────────────────────────────────────────────────────────────────
static const unsigned char BGZF_MAGIC[2] = {0x1f, 0x8b};
static const int BGZF_EXTRA_OFFSET = 12; // offset of extra sub-field
static const int BGZF_BSIZE_OFFSET  = 16; // offset of BSIZE field
static const int BGZF_HEADER_BYTES  = 18; // gzip header + XLEN + FEXTRA
static const int BGZF_TRAILER_BYTES =  8; // CRC32 (4) + ISIZE (4)

// ─────────────────────────────────────────────────────────────────────────────
// Static helpers
// ─────────────────────────────────────────────────────────────────────────────

bool CudaGzipDecompressor::isBgzf(const unsigned char* data, size_t nbytes) {
    if (nbytes < BGZF_SIGNATURE_BYTES) return false;
    if (data[0] != 0x1f || data[1] != 0x8b) return false;  // gzip magic
    if (data[2] != 8) return false;                          // DEFLATE method
    if (!(data[3] & 0x04)) return false;                     // FEXTRA flag
    // XLEN is at bytes 10-11 (little-endian)
    uint16_t xlen = (uint16_t)(data[10]) | ((uint16_t)(data[11]) << 8);
    if (nbytes < (size_t)(12 + xlen)) return false;
    // Scan the extra sub-fields for BC
    const unsigned char* p   = data + 12;
    const unsigned char* end = p + xlen;
    while (p + 4 <= end) {
        if (p[0] == 'B' && p[1] == 'C') return true;
        uint16_t slen = (uint16_t)(p[2]) | ((uint16_t)(p[3]) << 8);
        p += 4 + slen;
    }
    return false;
}

bool CudaGzipDecompressor::isGzip(const unsigned char* data, size_t nbytes) {
    if (nbytes < 2) return false;
    return data[0] == 0x1f && data[1] == 0x8b;
}

size_t CudaGzipDecompressor::bgzfBlockSize(const unsigned char* data,
                                            size_t avail) {
    if (avail < BGZF_SIGNATURE_BYTES + 2) return 0;
    if (data[0] != 0x1f || data[1] != 0x8b) return 0;
    if (!(data[3] & 0x04)) return 0;
    // BSIZE at bytes 16-17 (after SI1,SI2,SLEN)
    uint16_t bsize = (uint16_t)(data[BGZF_BSIZE_OFFSET]) |
                     ((uint16_t)(data[BGZF_BSIZE_OFFSET + 1]) << 8);
    return (size_t)bsize + 1;  // +1 because BSIZE = total_size - 1
}

// ─────────────────────────────────────────────────────────────────────────────
// Constructor / destructor
// ─────────────────────────────────────────────────────────────────────────────

CudaGzipDecompressor::CudaGzipDecompressor(int device)
    : mDevice(device), mValid(false), mStream(nullptr),
      d_compressed_flat(nullptr), d_decompressed_flat(nullptr),
      d_comp_ptrs(nullptr), d_decomp_ptrs(nullptr),
      d_comp_bytes(nullptr), d_decomp_buf_bytes(nullptr),
      d_decomp_act_bytes(nullptr), d_statuses(nullptr),
      d_temp(nullptr), mTempBytes(0),
      h_pinned_compressed(nullptr), h_pinned_output(nullptr),
      h_pinned_out_sizes(nullptr)
{
    cudaError_t err = cudaSetDevice(mDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[cuda_gzip] cudaSetDevice(%d) failed: %s\n",
                mDevice, cudaGetErrorString(err));
        return;
    }
    allocateBuffers();
}

CudaGzipDecompressor::~CudaGzipDecompressor() {
    freeBuffers();
}

// ─────────────────────────────────────────────────────────────────────────────
// Buffer allocation
// ─────────────────────────────────────────────────────────────────────────────

void CudaGzipDecompressor::allocateBuffers() {
    // Pick a representative temp size.  We call GetTempSizeAsync with our
    // maximum batch parameters so the temp buffer is always sufficient.
    nvcompBatchedDeflateDecompressOpts_t opts = nvcompBatchedDeflateDecompressDefaultOpts;

    size_t tempBytes = 0;
    nvcompStatus_t ns = nvcompBatchedDeflateDecompressGetTempSizeAsync(
        BGZF_MAX_CHUNKS,
        BGZF_DECOMP_BYTES,
        opts,
        &tempBytes,
        (size_t)BGZF_MAX_CHUNKS * BGZF_DECOMP_BYTES);
    if (ns != nvcompSuccess) {
        fprintf(stderr, "[cuda_gzip] Deflate GetTempSizeAsync failed: %d\n", ns);
        return;
    }
    mTempBytes = tempBytes;

    // Main flat buffers
    if (cudaMalloc(&d_compressed_flat,
                   (size_t)BGZF_MAX_CHUNKS * BGZF_BLOCK_BYTES) != cudaSuccess ||
        cudaMalloc(&d_decompressed_flat,
                   (size_t)BGZF_MAX_CHUNKS * BGZF_DECOMP_BYTES) != cudaSuccess)
        goto fail;

    // Metadata arrays on device
    if (cudaMalloc(&d_comp_ptrs,           BGZF_MAX_CHUNKS * sizeof(void*))   != cudaSuccess ||
        cudaMalloc(&d_decomp_ptrs,         BGZF_MAX_CHUNKS * sizeof(void*))   != cudaSuccess ||
        cudaMalloc(&d_comp_bytes,          BGZF_MAX_CHUNKS * sizeof(size_t))  != cudaSuccess ||
        cudaMalloc(&d_decomp_buf_bytes,    BGZF_MAX_CHUNKS * sizeof(size_t))  != cudaSuccess ||
        cudaMalloc(&d_decomp_act_bytes,    BGZF_MAX_CHUNKS * sizeof(size_t))  != cudaSuccess ||
        cudaMalloc(&d_statuses,            BGZF_MAX_CHUNKS * sizeof(nvcompStatus_t)) != cudaSuccess)
        goto fail;

    // Temp scratch
    if (mTempBytes > 0) {
        if (cudaMalloc(&d_temp, mTempBytes) != cudaSuccess) goto fail;
    }

    // Pinned host staging buffers
    if (cudaHostAlloc(&h_pinned_compressed,
                      (size_t)BGZF_MAX_CHUNKS * BGZF_BLOCK_BYTES,
                      cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&h_pinned_output,
                      (size_t)BGZF_MAX_CHUNKS * BGZF_DECOMP_BYTES,
                      cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&h_pinned_out_sizes,
                      BGZF_MAX_CHUNKS * sizeof(size_t),
                      cudaHostAllocDefault) != cudaSuccess)
        goto fail;

    // CUDA stream
    if (cudaStreamCreate(&mStream) != cudaSuccess) goto fail;

    // ── Pre-populate the static pointer arrays ────────────────────────────
    // d_comp_ptrs[i]   = d_compressed_flat   + i * BGZF_BLOCK_BYTES
    // d_decomp_ptrs[i] = d_decompressed_flat + i * BGZF_DECOMP_BYTES
    // d_decomp_buf_bytes[i] = BGZF_DECOMP_BYTES  (fixed)
    {
        void*   tmp_comp[BGZF_MAX_CHUNKS];
        void*   tmp_decomp[BGZF_MAX_CHUNKS];
        size_t  tmp_buf[BGZF_MAX_CHUNKS];
        for (int i = 0; i < BGZF_MAX_CHUNKS; i++) {
            tmp_comp[i]   = (void*)(d_compressed_flat   + (size_t)i * BGZF_BLOCK_BYTES);
            tmp_decomp[i] = (void*)(d_decompressed_flat + (size_t)i * BGZF_DECOMP_BYTES);
            tmp_buf[i]    = BGZF_DECOMP_BYTES;
        }
        if (cudaMemcpy(d_comp_ptrs,        tmp_comp,   BGZF_MAX_CHUNKS * sizeof(void*),
                       cudaMemcpyHostToDevice)                                          != cudaSuccess ||
            cudaMemcpy(d_decomp_ptrs,      tmp_decomp, BGZF_MAX_CHUNKS * sizeof(void*),
                       cudaMemcpyHostToDevice)                                          != cudaSuccess ||
            cudaMemcpy(d_decomp_buf_bytes, tmp_buf,    BGZF_MAX_CHUNKS * sizeof(size_t),
                       cudaMemcpyHostToDevice)                                          != cudaSuccess)
            goto fail;
    }

    mValid = true;
    return;

fail:
    fprintf(stderr, "[cuda_gzip] allocateBuffers() failed\n");
    freeBuffers();
}

void CudaGzipDecompressor::freeBuffers() {
    if (mStream)              { cudaStreamDestroy(mStream);            mStream = nullptr; }
    if (d_compressed_flat)    { cudaFree(d_compressed_flat);           d_compressed_flat = nullptr; }
    if (d_decompressed_flat)  { cudaFree(d_decompressed_flat);         d_decompressed_flat = nullptr; }
    if (d_comp_ptrs)          { cudaFree(d_comp_ptrs);                 d_comp_ptrs = nullptr; }
    if (d_decomp_ptrs)        { cudaFree(d_decomp_ptrs);               d_decomp_ptrs = nullptr; }
    if (d_comp_bytes)         { cudaFree(d_comp_bytes);                d_comp_bytes = nullptr; }
    if (d_decomp_buf_bytes)   { cudaFree(d_decomp_buf_bytes);          d_decomp_buf_bytes = nullptr; }
    if (d_decomp_act_bytes)   { cudaFree(d_decomp_act_bytes);          d_decomp_act_bytes = nullptr; }
    if (d_statuses)           { cudaFree(d_statuses);                  d_statuses = nullptr; }
    if (d_temp)               { cudaFree(d_temp);                      d_temp = nullptr; }
    if (h_pinned_compressed)  { cudaFreeHost(h_pinned_compressed);     h_pinned_compressed = nullptr; }
    if (h_pinned_output)      { cudaFreeHost(h_pinned_output);         h_pinned_output = nullptr; }
    if (h_pinned_out_sizes)   { cudaFreeHost(h_pinned_out_sizes);      h_pinned_out_sizes = nullptr; }
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch decompression
// ─────────────────────────────────────────────────────────────────────────────

int CudaGzipDecompressor::decompress(
    const unsigned char* h_compressed_flat,
    const size_t*        chunk_offsets,
    const size_t*        chunk_comp_bytes,
    size_t               num_chunks,
    unsigned char*       h_output_flat,
    size_t*              out_sizes)
{
    if (!mValid) return -1;
    if (num_chunks == 0) return 0;
    if (num_chunks > (size_t)BGZF_MAX_CHUNKS) {
        fprintf(stderr, "[cuda_gzip] num_chunks %zu > BGZF_MAX_CHUNKS %d\n",
                num_chunks, BGZF_MAX_CHUNKS);
        return -1;
    }

    cudaSetDevice(mDevice);

    PROF_START(_t_decomp);
    // BGZF blocks are gzip-wrapped DEFLATE. Strip the 18-byte gzip header and
    // 8-byte CRC32/ISIZE trailer and pass only the raw DEFLATE stream to nvCOMP.
    size_t  deflate_sizes[BGZF_MAX_CHUNKS];
    size_t total_comp = 0;
    for (size_t i = 0; i < num_chunks; i++) {
        size_t gz_sz = chunk_comp_bytes[i];
        if (gz_sz <= (size_t)(BGZF_HEADER_BYTES + BGZF_TRAILER_BYTES)) {
            fprintf(stderr, "[cuda_gzip] chunk %zu too small (%zu bytes)\n", i, gz_sz);
            return -1;
        }
        size_t def_sz = gz_sz - BGZF_HEADER_BYTES - BGZF_TRAILER_BYTES;
        deflate_sizes[i] = def_sz;
        memcpy(h_pinned_compressed + i * BGZF_BLOCK_BYTES,
               h_compressed_flat + chunk_offsets[i] + BGZF_HEADER_BYTES,
               def_sz);
        total_comp += def_sz;
    }

    // ── 2. Build compressed-size device array (host → device) ────────────
    // Use deflate_sizes (not chunk_comp_bytes which includes gzip header/trailer)
    CUDA_CHECK(cudaMemcpyAsync(d_comp_bytes,
                               deflate_sizes,
                               num_chunks * sizeof(size_t),
                               cudaMemcpyHostToDevice, mStream));

    // ── 3. Upload compressed data ─────────────────────────────────────────
    // Copy full uniform flat buffer (num_chunks * BGZF_BLOCK_BYTES)
    CUDA_CHECK(cudaMemcpyAsync(d_compressed_flat,
                               h_pinned_compressed,
                               num_chunks * BGZF_BLOCK_BYTES,
                               cudaMemcpyHostToDevice, mStream));

    // ── 4. GPU decompression (raw DEFLATE) ───────────────────────────────
    nvcompBatchedDeflateDecompressOpts_t opts = nvcompBatchedDeflateDecompressDefaultOpts;

    NVCOMP_CHECK(nvcompBatchedDeflateDecompressAsync(
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
        mStream));

    // ── 5. Download decompressed data and sizes ───────────────────────────
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_output,
                               d_decompressed_flat,
                               num_chunks * BGZF_DECOMP_BYTES,
                               cudaMemcpyDeviceToHost, mStream));
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_out_sizes,
                               d_decomp_act_bytes,
                               num_chunks * sizeof(size_t),
                               cudaMemcpyDeviceToHost, mStream));

    // ── 6. Wait for completion ────────────────────────────────────────────
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    // Debug: check nvCOMP statuses
    {
        nvcompStatus_t* h_statuses = new nvcompStatus_t[num_chunks];
        cudaMemcpy(h_statuses, d_statuses, num_chunks * sizeof(nvcompStatus_t),
                   cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < num_chunks; i++) {
            if (h_statuses[i] != nvcompSuccess) {
                fprintf(stderr, "[cuda_gzip] chunk %zu nvCOMP deflate status=%d\n",
                        i, (int)h_statuses[i]);
            }
        }
        delete[] h_statuses;
    }

    // ── 7. Copy from pinned staging to caller output ──────────────────────
    // Write at fixed BGZF_DECOMP_BYTES strides so readToBufBgzfGpu() can read
    // with h_output_flat[i * BGZF_DECOMP_BYTES]
    for (size_t i = 0; i < num_chunks; i++) {
        size_t act = h_pinned_out_sizes[i];
        out_sizes[i] = act;
        if (h_output_flat != nullptr) {
            memcpy(h_output_flat + (size_t)i * BGZF_DECOMP_BYTES,
                   h_pinned_output + (size_t)i * BGZF_DECOMP_BYTES,
                   act);
        }
    }

    PROF_END(_t_decomp, decompress_ns);

    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Whole-file standard gzip decompression on GPU
//
// Uses nvcompBatchedGzipDecompressAsync with a single chunk (the entire gzip
// file).  Steps:
//   1. Upload compressed gzip data to GPU.
//   2. Query decompressed size via nvcompBatchedGzipGetDecompressSizeAsync.
//   3. Allocate GPU output buffer.
//   4. Decompress.
//   5. Download decompressed data to host (new[]-allocated).
//   6. Free GPU temporaries.
// ─────────────────────────────────────────────────────────────────────────────

int CudaGzipDecompressor::decompressGzipWhole(
    const unsigned char* h_compressed,
    size_t               comp_bytes,
    unsigned char**      h_output,
    size_t*              decomp_bytes)
{
    if (!mValid) return -1;
    if (comp_bytes == 0) return -1;

    *h_output     = nullptr;
    *decomp_bytes = 0;

    cudaSetDevice(mDevice);

    PROF_START(_t_gzip);

    // ── 1. Upload compressed data to GPU ──────────────────────────────────
    void* d_comp = nullptr;
    cudaError_t ce = cudaMalloc(&d_comp, comp_bytes);
    if (ce != cudaSuccess) {
        fprintf(stderr, "[cuda_gzip] cudaMalloc(%zu) for compressed data failed: %s\n",
                comp_bytes, cudaGetErrorString(ce));
        return -1;
    }

    // Use pinned staging for faster H2D if file fits; otherwise direct copy
    unsigned char* h_pinned = nullptr;
    ce = cudaHostAlloc(&h_pinned, comp_bytes, cudaHostAllocDefault);
    if (ce == cudaSuccess) {
        memcpy(h_pinned, h_compressed, comp_bytes);
        ce = cudaMemcpyAsync(d_comp, h_pinned, comp_bytes,
                             cudaMemcpyHostToDevice, mStream);
    } else {
        // Fallback: unpinned copy
        h_pinned = nullptr;
        ce = cudaMemcpy(d_comp, h_compressed, comp_bytes,
                        cudaMemcpyHostToDevice);
    }
    if (ce != cudaSuccess) {
        fprintf(stderr, "[cuda_gzip] H2D copy failed: %s\n",
                cudaGetErrorString(ce));
        cudaFree(d_comp);
        if (h_pinned) cudaFreeHost(h_pinned);
        return -1;
    }

    // ── 2. Set up batch arrays (1 chunk) ──────────────────────────────────
    // Device arrays for the batch pointers/sizes
    void**          d_comp_ptrs_g   = nullptr;
    void**          d_decomp_ptrs_g = nullptr;
    size_t*         d_comp_sizes    = nullptr;
    size_t*         d_decomp_buf_sz = nullptr;
    size_t*         d_decomp_act_sz = nullptr;
    nvcompStatus_t* d_status        = nullptr;
    void*           d_temp_g        = nullptr;

    if (cudaMalloc(&d_comp_ptrs_g,   sizeof(void*))          != cudaSuccess ||
        cudaMalloc(&d_decomp_ptrs_g, sizeof(void*))          != cudaSuccess ||
        cudaMalloc(&d_comp_sizes,    sizeof(size_t))         != cudaSuccess ||
        cudaMalloc(&d_decomp_buf_sz, sizeof(size_t))         != cudaSuccess ||
        cudaMalloc(&d_decomp_act_sz, sizeof(size_t))         != cudaSuccess ||
        cudaMalloc(&d_status,        sizeof(nvcompStatus_t)) != cudaSuccess) {
        fprintf(stderr, "[cuda_gzip] Failed to allocate batch metadata\n");
        cudaFree(d_comp);
        if (h_pinned) cudaFreeHost(h_pinned);
        return -1;
    }

    // Copy compressed pointer and size to device
    void*  h_comp_ptr = d_comp;
    size_t h_comp_sz  = comp_bytes;
    cudaMemcpyAsync(d_comp_ptrs_g, &h_comp_ptr, sizeof(void*),
                    cudaMemcpyHostToDevice, mStream);
    cudaMemcpyAsync(d_comp_sizes, &h_comp_sz, sizeof(size_t),
                    cudaMemcpyHostToDevice, mStream);

    // ── 3. Query decompressed size ────────────────────────────────────────
    cudaStreamSynchronize(mStream);
    nvcompStatus_t ns = nvcompBatchedGzipGetDecompressSizeAsync(
        (const void* const*)d_comp_ptrs_g,
        d_comp_sizes,
        d_decomp_act_sz,
        1,  // num_chunks
        mStream);
    if (ns != nvcompSuccess) {
        fprintf(stderr, "[cuda_gzip] GzipGetDecompressSizeAsync failed: %d\n", ns);
        goto cleanup_fail;
    }
    cudaStreamSynchronize(mStream);

    size_t h_decomp_size;
    cudaMemcpy(&h_decomp_size, d_decomp_act_sz, sizeof(size_t),
               cudaMemcpyDeviceToHost);

    if (h_decomp_size == 0) {
        fprintf(stderr, "[cuda_gzip] Decompressed size reported as 0\n");
        goto cleanup_fail;
    }

    fprintf(stderr, "[cuda_gzip] Standard gzip: compressed=%zu  decompressed=%zu  ratio=%.1fx\n",
            comp_bytes, h_decomp_size, (double)h_decomp_size / comp_bytes);

    // ── 4. Allocate GPU output buffer ─────────────────────────────────────
    {
        void* d_decomp = nullptr;
        ce = cudaMalloc(&d_decomp, h_decomp_size);
        if (ce != cudaSuccess) {
            fprintf(stderr, "[cuda_gzip] cudaMalloc(%zu) for decompressed buffer failed: %s\n",
                    h_decomp_size, cudaGetErrorString(ce));
            goto cleanup_fail;
        }

        void*  h_decomp_ptr = d_decomp;
        size_t h_buf_sz     = h_decomp_size;
        cudaMemcpyAsync(d_decomp_ptrs_g, &h_decomp_ptr, sizeof(void*),
                        cudaMemcpyHostToDevice, mStream);
        cudaMemcpyAsync(d_decomp_buf_sz, &h_buf_sz, sizeof(size_t),
                        cudaMemcpyHostToDevice, mStream);

        // ── 5. Get temp size and allocate ─────────────────────────────────
        size_t tempBytes = 0;
        nvcompBatchedGzipDecompressOpts_t opts = nvcompBatchedGzipDecompressDefaultOpts;
        opts.algorithm = NVCOMP_GZIP_DECOMPRESS_ALGORITHM_LOOKAHEAD;
        ns = nvcompBatchedGzipDecompressGetTempSizeAsync(
            1, h_decomp_size, opts, &tempBytes, h_decomp_size);
        if (ns != nvcompSuccess) {
            fprintf(stderr, "[cuda_gzip] GzipDecompressGetTempSizeAsync failed: %d\n", ns);
            cudaFree(d_decomp);
            goto cleanup_fail;
        }
        if (tempBytes > 0) {
            if (cudaMalloc(&d_temp_g, tempBytes) != cudaSuccess) {
                fprintf(stderr, "[cuda_gzip] cudaMalloc temp (%zu) failed\n", tempBytes);
                cudaFree(d_decomp);
                goto cleanup_fail;
            }
        }

        // ── 6. Decompress ─────────────────────────────────────────────────
        cudaStreamSynchronize(mStream);
        ns = nvcompBatchedGzipDecompressAsync(
            (const void* const*)d_comp_ptrs_g,
            d_comp_sizes,
            d_decomp_buf_sz,
            d_decomp_act_sz,
            1,  // num_chunks
            d_temp_g,
            tempBytes,
            (void* const*)d_decomp_ptrs_g,
            opts,
            d_status,
            mStream);
        if (ns != nvcompSuccess) {
            fprintf(stderr, "[cuda_gzip] GzipDecompressAsync failed: %d\n", ns);
            cudaFree(d_decomp);
            goto cleanup_fail;
        }
        cudaStreamSynchronize(mStream);

        // Check status
        nvcompStatus_t h_status;
        cudaMemcpy(&h_status, d_status, sizeof(nvcompStatus_t),
                   cudaMemcpyDeviceToHost);
        if (h_status != nvcompSuccess) {
            fprintf(stderr, "[cuda_gzip] GPU gzip decompression status: %d\n",
                    (int)h_status);
            cudaFree(d_decomp);
            goto cleanup_fail;
        }

        // Read actual decompressed size
        cudaMemcpy(&h_decomp_size, d_decomp_act_sz, sizeof(size_t),
                   cudaMemcpyDeviceToHost);

        // ── 7. Download decompressed data to host ─────────────────────────
        unsigned char* host_out = new unsigned char[h_decomp_size];
        ce = cudaMemcpy(host_out, d_decomp, h_decomp_size,
                        cudaMemcpyDeviceToHost);
        if (ce != cudaSuccess) {
            fprintf(stderr, "[cuda_gzip] D2H copy failed: %s\n",
                    cudaGetErrorString(ce));
            delete[] host_out;
            cudaFree(d_decomp);
            goto cleanup_fail;
        }

        *h_output     = host_out;
        *decomp_bytes = h_decomp_size;

        // ── 8. Cleanup GPU memory ─────────────────────────────────────────
        cudaFree(d_decomp);
    }

    // Success path cleanup
    cudaFree(d_comp);
    cudaFree(d_comp_ptrs_g);
    cudaFree(d_decomp_ptrs_g);
    cudaFree(d_comp_sizes);
    cudaFree(d_decomp_buf_sz);
    cudaFree(d_decomp_act_sz);
    cudaFree(d_status);
    if (d_temp_g) cudaFree(d_temp_g);
    if (h_pinned) cudaFreeHost(h_pinned);

    PROF_END(_t_gzip, decompress_ns);
    return 0;

cleanup_fail:
    cudaFree(d_comp);
    cudaFree(d_comp_ptrs_g);
    cudaFree(d_decomp_ptrs_g);
    cudaFree(d_comp_sizes);
    cudaFree(d_decomp_buf_sz);
    cudaFree(d_decomp_act_sz);
    cudaFree(d_status);
    if (d_temp_g) cudaFree(d_temp_g);
    if (h_pinned) cudaFreeHost(h_pinned);
    return -1;
}

#endif // HAVE_CUDA && HAVE_NVCOMP
