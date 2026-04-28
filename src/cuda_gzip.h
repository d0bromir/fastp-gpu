/*
 * cuda_gzip.h — GPU-accelerated gzip/BGZF decompressor using NVIDIA nvCOMP.
 *
 * Supports two modes:
 *
 * 1. BGZF (Blocked GNU Zip Format) — a specialised multi-member gzip format
 *    where each block is independently compressed (used by htslib, bgzip, and
 *    compatible genomics tools).  Each BGZF block is at most 65535 bytes
 *    compressed and 65536 bytes uncompressed.  Batches of up to MAX_CHUNKS
 *    blocks are decompressed in parallel via nvCOMP batched deflate.
 *
 * 2. Standard gzip — a single-member gzip file (produced by `gzip`, pigz,
 *    etc.).  The entire compressed file is uploaded to the GPU and decompressed
 *    as a single batch item using nvCOMP's batched gzip API.  This requires
 *    enough GPU memory to hold both the compressed and decompressed data
 *    simultaneously.
 *
 * Design constraints:
 *  - MAX_CHUNKS  = 4096 blocks per batch  (~256 MB compressed, ~256 MB output)
 *  - Each slot uses uniform 65536-byte slots so no pointer arithmetic is needed.
 *  - Pinned host buffers for async H2D and D2H transfers.
 *  - Single CUDA stream per decompressor instance.
 *  - Thread-safe as long as each thread owns its own CudaGzipDecompressor.
 */
#pragma once
#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP)

#include <cuda_runtime.h>
#include <nvcomp/deflate.h>
#include <nvcomp/gzip.h>
#include <cstddef>

// ──────────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────────
static const int    BGZF_MAX_CHUNKS        = 4096;
static const size_t BGZF_BLOCK_BYTES       = 65536;  // max compressed per block
static const size_t BGZF_DECOMP_BYTES      = 65536;  // max decompressed per block
static const size_t BGZF_SIGNATURE_BYTES   = 18;     // minimum BGZF header size

// ──────────────────────────────────────────────────────────────────────────
// CudaGzipDecompressor
// ──────────────────────────────────────────────────────────────────────────
class CudaGzipDecompressor {
public:
    /**
     * @param device  CUDA device index to use.  The caller must ensure the
     *                device is valid.  cudaSetDevice is called in the ctor.
     */
    explicit CudaGzipDecompressor(int device = 0);
    ~CudaGzipDecompressor();

    // Non-copyable
    CudaGzipDecompressor(const CudaGzipDecompressor&) = delete;
    CudaGzipDecompressor& operator=(const CudaGzipDecompressor&) = delete;

    /**
     * Detect whether the first few bytes look like a gzip stream (any kind).
     * Checks for the 1f 8b magic bytes.
     *
     * @param data   Pointer to at least 2 bytes.
     * @param nbytes Number of bytes available at @data.
     * @return true if the data looks like a gzip stream.
     */
    static bool isGzip(const unsigned char* data, size_t nbytes);

    /**
     * Detect whether the first few bytes of a buffer look like a BGZF stream.
     * A valid BGZF block has gzip magic (1f 8b), method=8, FEXTRA flag set,
     * and a "BC" sub-field in the FEXTRA area.
     *
     * @param data   Pointer to at least BGZF_SIGNATURE_BYTES bytes.
     * @param nbytes Number of bytes available at @data.
     * @return true if the data looks like a BGZF stream.
     */
    static bool isBgzf(const unsigned char* data, size_t nbytes);

    /**
     * Parse the BGZF block size from the header at @data.
     * Returns the total block size in bytes (header + deflate data + CRC + ISIZE),
     * or 0 if the header is malformed.
     *
     * @param data   Pointer to the first byte of a BGZF block.
     * @param avail  Number of bytes available at @data.
     */
    static size_t bgzfBlockSize(const unsigned char* data, size_t avail);

    /**
     * Decompress a batch of BGZF blocks on the GPU.
     *
     * @param h_compressed_flat  Flat host buffer containing all compressed
     *                           blocks concatenated.
     * @param chunk_offsets      chunk_offsets[i] = byte offset of block i in
     *                           h_compressed_flat.
     * @param chunk_comp_bytes   chunk_comp_bytes[i] = compressed size of block i.
     * @param num_chunks         Number of blocks.  Must be ≤ BGZF_MAX_CHUNKS.
     * @param h_output_flat      Output flat host buffer.  Must be pre-allocated
     *                           with at least num_chunks * BGZF_DECOMP_BYTES bytes.
     * @param out_sizes          out_sizes[i] = actual decompressed bytes of block i.
     *                           Must be pre-allocated with num_chunks elements.
     *
     * @return 0 on success, -1 on error (details printed to stderr).
     */
    int decompress(const unsigned char* h_compressed_flat,
                   const size_t*        chunk_offsets,
                   const size_t*        chunk_comp_bytes,
                   size_t               num_chunks,
                   unsigned char*       h_output_flat,
                   size_t*              out_sizes);

    /**
     * Decompress an entire standard gzip file on the GPU.
     *
     * Uploads the complete compressed file to GPU memory, uses nvCOMP's gzip
     * batch API with a single chunk, downloads the decompressed result.
     * The caller must free *h_output with delete[].
     *
     * @param h_compressed   Pointer to the entire gzip file content in host memory.
     * @param comp_bytes     Size of the compressed file in bytes.
     * @param h_output       Output: pointer to new[]-allocated decompressed data.
     * @param decomp_bytes   Output: actual decompressed size in bytes.
     *
     * @return 0 on success, -1 on error (details printed to stderr).
     */
    int decompressGzipWhole(const unsigned char* h_compressed,
                            size_t               comp_bytes,
                            unsigned char**      h_output,
                            size_t*              decomp_bytes);

    /** Return the CUDA device this instance is bound to. */
    int device() const { return mDevice; }

    /** Return true if the object was successfully initialised. */
    bool valid() const { return mValid; }

private:
    void allocateBuffers();
    void freeBuffers();

    int             mDevice;
    bool            mValid;
    cudaStream_t    mStream;

    // ── Flat device buffers ───────────────────────────────────────────────
    // Each block slot is BGZF_BLOCK_BYTES / BGZF_DECOMP_BYTES wide.
    // Uniform sizing avoids variable-stride pointer arrays.
    unsigned char*  d_compressed_flat;    // [MAX_CHUNKS * BGZF_BLOCK_BYTES]
    unsigned char*  d_decompressed_flat;  // [MAX_CHUNKS * BGZF_DECOMP_BYTES]

    // ── Device metadata arrays ────────────────────────────────────────────
    void**          d_comp_ptrs;          // ptrs into d_compressed_flat
    void**          d_decomp_ptrs;        // ptrs into d_decompressed_flat
    size_t*         d_comp_bytes;         // compressed size per chunk
    size_t*         d_decomp_buf_bytes;   // allocated output bytes per chunk
    size_t*         d_decomp_act_bytes;   // actual output bytes (written by nvCOMP)
    nvcompStatus_t* d_statuses;           // per-chunk status

    // ── Temporary decompression scratch space ─────────────────────────────
    void*           d_temp;
    size_t          mTempBytes;

    // ── Pinned host buffers for async transfers ───────────────────────────
    unsigned char*  h_pinned_compressed;  // staging: compressed input
    unsigned char*  h_pinned_output;      // staging: decompressed output
    size_t*         h_pinned_out_sizes;   // staging: actual output sizes
};

#endif // HAVE_CUDA && HAVE_NVCOMP
