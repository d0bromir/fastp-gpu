#include "cuda_stats.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Define block and thread configuration
// BLOCK_SIZE must be a multiple of 32 (warp size).
// With 1 warp (32 threads) per read: BLOCK_SIZE=256 → 8 reads per block.
// For 8192 reads → 1024 blocks → covers all 108 A100 SMs ~9.5× → high utilisation.
#define BLOCK_SIZE 256
#define READS_PER_BLOCK (BLOCK_SIZE / 32)   // 8 reads per block
#define MAX_READ_LENGTH 512

/**
 * CUDA kernel for PolyG trimming detection
 * Each thread detects the longest homopolymer G run in its read
 */
__global__ void detect_polyG_kernel(
    const char** d_sequences,
    const int* d_read_lengths,
    int min_len,
    struct ReadStatistics* d_stats
) {
    int read_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (read_id >= d_read_lengths[0] && read_id >= 0) return;  // Safe boundary
    
    const char* seq = d_sequences[read_id];
    int read_len = d_read_lengths[read_id];
    
    // Find longest polyG run from the end of read
    int run_len = 0;
    int trim_pos = read_len;  // Default: trim nothing
    
    for (int i = read_len - 1; i >= 0; i--) {
        if (seq[i] == 'G' || seq[i] == 'g') {
            run_len++;
            if (run_len >= min_len) {
                trim_pos = i;
            }
        } else {
            if (run_len >= min_len) break;
            run_len = 0;
        }
    }
    
    d_stats[read_id].polyG_trim_pos = (trim_pos < read_len) ? trim_pos : -1;
}

/**
 * CUDA kernel for quality-based trimming with sliding window
 * Simplified: find start and end positions with sufficient quality
 */
__global__ void quality_trim_kernel(
    const char** d_qualities,
    const int* d_read_lengths,
    char qual_threshold,
    struct ReadStatistics* d_stats
) {
    int read_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (read_id >= d_read_lengths[0] && read_id >= 0) return;
    
    const char* qual = d_qualities[read_id];
    int read_len = d_read_lengths[read_id];
    
    // Find first position with adequate quality
    int trim_start = 0;
    for (int i = 0; i < read_len; i++) {
        if (qual[i] >= qual_threshold) {
            trim_start = i;
            break;
        }
    }
    
    // Find last position with adequate quality
    int trim_end = read_len;
    for (int i = read_len - 1; i >= trim_start; i--) {
        if (qual[i] >= qual_threshold) {
            trim_end = i + 1;
            break;
        }
    }
    
    d_stats[read_id].trim_start = trim_start;
    d_stats[read_id].trim_end = trim_end;
}

/**
 * CUDA kernel for applying trim positions to sequences
 * Strategy 1: Keep Data on GPU Longer
 * 
 * Extracts trimmed sequences and qualities based on computed trim positions,
 * reducing D2H transfer size and eliminating CPU substring operations.
 * 
 * Output format: packed array with variable-length reads
 * [read0_len(4B)][read0_seq(len)][read0_qual(len)][read1_len(4B)]...
 */
__global__ void apply_trim_to_sequences_kernel(
    const char** d_sequences,
    const char** d_qualities,
    const int* d_read_lengths,
    const struct ReadStatistics* d_stats,
    char* d_trimmed_output,
    int* d_output_offsets,  // Prefix sum of output sizes
    int num_reads
) {
    int read_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (read_id >= num_reads) return;
    
    const char* seq = d_sequences[read_id];
    const char* qual = d_qualities[read_id];
    const struct ReadStatistics* stat = &d_stats[read_id];
    
    // Get trim positions
    int trim_start = stat->trim_start;
    int trim_end = stat->trim_end;
    int polyG_pos = stat->polyG_trim_pos;
    
    // Apply PolyG trim if detected
    if (polyG_pos >= 0 && polyG_pos < trim_end) {
        trim_end = polyG_pos;
    }
    
    // Ensure valid range
    if (trim_start >= trim_end) {
        trim_start = 0;
        trim_end = 0;
    }
    
    int trimmed_len = trim_end - trim_start;
    
    // Get output offset for this read
    int out_offset = d_output_offsets[read_id];
    
    // Write trimmed length
    *(int*)(d_trimmed_output + out_offset) = trimmed_len;
    out_offset += sizeof(int);
    
    // Write trimmed sequence
    for (int i = 0; i < trimmed_len; i++) {
        d_trimmed_output[out_offset + i] = seq[trim_start + i];
    }
    out_offset += trimmed_len;
    
    // Write trimmed quality
    for (int i = 0; i < trimmed_len; i++) {
        d_trimmed_output[out_offset + i] = qual[trim_start + i];
    }
}

/**
 * Warp-per-read CUDA kernel for per-read statistics (REPLACES unified kernel).
 *
 * Each warp (32 threads) processes exactly one read cooperatively:
 *
 *   Phase 1 – parallel stats (ALL 32 lanes active):
 *     Each lane handles every 32nd base (stride-32 access pattern).
 *     n_bases, low_qual_bases, total_quality are accumulated per-lane then
 *     reduced to lane 0 using __shfl_down_sync in 5 rounds (log2(32)).
 *
 *   Phase 2 – trim positions + PolyG (lane 0 only, ~read_len iterations):
 *     Sequential scan; at 150 bp this is ~150 memory loads executed while
 *     the other 31 lanes are implicitly stalled but the warp stays in flight,
 *     overlapping with other warps on the same SM.
 *
 * Grid sizing: ceil(num_reads / READS_PER_BLOCK) blocks of BLOCK_SIZE threads.
 *   8 192 reads → 1 024 blocks → fills all 108 A100 SMs with ~9.5 blocks each
 *   → near-peak occupancy vs. the old 2 blocks for 512 reads (1 thread/read).
 */
__global__ void compute_read_stats_warp_kernel(
    const char** d_sequences,
    const char** d_qualities,
    const int* d_read_lengths,
    int num_reads,
    char qual_threshold,
    struct ReadStatistics* d_stats,
    int trim_window_size
) {
    const int WARP_SIZE = 32;
    int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int read_id       = global_thread / WARP_SIZE;
    int lane          = global_thread % WARP_SIZE;
    int warp_in_block = threadIdx.x / WARP_SIZE;

    if (read_id >= num_reads) return;

    const char* seq   = d_sequences[read_id];
    const char* qual  = d_qualities[read_id];
    int read_len      = d_read_lengths[read_id];

    /* Shared memory for sliding-window prefix sum (Phase 2a).
     * s_prefix[w][i+1] = sum of Phred[0..i]  (s_prefix[w][0] = 0)
     * Allocated statically; populated only when trim_window_size > 1.
     * 8 warps x 513 int32s x 4 B = 16.4 KB per block (A100 limit: 48-96 KB). */
    __shared__ int s_prefix[READS_PER_BLOCK][MAX_READ_LENGTH + 1];

    /* ------------------------------------------------------------------ */
    /* Phase 1: per-base stats – all 32 lanes work in parallel             */
    /* ------------------------------------------------------------------ */
    int n_bases    = 0;
    int low_qual   = 0;
    int total_qual = 0;

    for (int i = lane; i < read_len; i += WARP_SIZE) {
        char c = seq[i];
        char q = qual[i];
        if (c == 'N' || c == 'n') n_bases++;
        if (q < qual_threshold)   low_qual++;
        total_qual += (int)(unsigned char)(q - 33);
    }

    /* Warp-level reduction (5 rounds, log2(32)) via shuffle */
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        n_bases    += __shfl_down_sync(mask, n_bases,    offset);
        low_qual   += __shfl_down_sync(mask, low_qual,   offset);
        total_qual += __shfl_down_sync(mask, total_qual, offset);
    }
    /* Lane 0 now holds the warp-reduced totals. */

    /* ------------------------------------------------------------------ */
    /* Phase 2a: cooperatively build Phred prefix-sum (all 32 lanes).     */
    /* Skipped when trim_window_size <= 1 (fast path uses single-base scan)*/
    /* After this: s_prefix[warp_in_block][i+1] = sum(Phred[0 .. i])      */
    /* ------------------------------------------------------------------ */
    if (trim_window_size > 1) {
        /* Cap at MAX_READ_LENGTH to stay within shared-memory array bounds */
        int plen = (read_len < MAX_READ_LENGTH) ? read_len : MAX_READ_LENGTH;

        if (lane == 0) s_prefix[warp_in_block][0] = 0;
        __syncwarp(mask);

        int running = 0;
        for (int tile_start = 0; tile_start < plen; tile_start += WARP_SIZE) {
            int pos = tile_start + lane;
            /* Phred score (0-based); 0 for out-of-bounds padding */
            int q = (pos < plen) ? (int)(unsigned char)(qual[pos] - 33) : 0;
            /* Warp-level inclusive prefix scan via shuffle-up (5 rounds) */
            for (int step = 1; step < WARP_SIZE; step <<= 1) {
                int t = __shfl_up_sync(mask, q, step);
                if (lane >= step) q += t;
            }
            /* q[lane] = sum(Phred[tile_start .. tile_start+lane]).
             * Add accumulated sum from all previous tiles. */
            q += running;
            if (pos < plen)
                s_prefix[warp_in_block][pos + 1] = q;
            /* Broadcast from the last valid lane to update running total */
            int valid = ((plen - tile_start) < WARP_SIZE) ? (plen - tile_start) : WARP_SIZE;
            running = __shfl_sync(mask, q, valid - 1);
        }
        /* Ensure all stores are visible before lane 0 reads in Phase 2b */
        __syncwarp(mask);
    }

    /* ------------------------------------------------------------------ */
    /* Phase 2b: trim positions + PolyG – lane 0 only                     */
    /* ------------------------------------------------------------------ */
    if (lane == 0) {
        d_stats[read_id].total_bases    = read_len;
        d_stats[read_id].n_bases        = n_bases;
        d_stats[read_id].low_qual_bases = low_qual;
        d_stats[read_id].total_quality  = total_qual;

        int trim_start = 0;
        int trim_end   = read_len;

        if (trim_window_size <= 1) {
            /* W=1: single-base quality scan (original behaviour) */
            for (int i = 0; i < read_len; i++) {
                if (qual[i] >= qual_threshold) { trim_start = i; break; }
            }
            for (int i = read_len - 1; i >= trim_start; i--) {
                if (qual[i] >= qual_threshold) { trim_end = i + 1; break; }
            }
        } else {
            /* W>1: sliding-window trim using prefix sum from Phase 2a.
             * qual_threshold is ASCII-encoded (Phred + 33 offset). */
            int phred_thresh = (int)(unsigned char)(qual_threshold - 33);
            int win_thresh   = trim_window_size * phred_thresh;
            int W            = trim_window_size;
            int eff_len      = (read_len < MAX_READ_LENGTH) ? read_len : MAX_READ_LENGTH;

            /* trim_start: leftmost window p s.t. sum(Phred[p..p+W-1]) >= win_thresh */
            for (int p = 0; p + W <= eff_len; p++) {
                if (s_prefix[warp_in_block][p + W] - s_prefix[warp_in_block][p]
                        >= win_thresh) {
                    trim_start = p;
                    break;
                }
            }
            /* trim_end: end index of the rightmost passing window */
            for (int p = eff_len - W; p >= 0; p--) {
                if (s_prefix[warp_in_block][p + W] - s_prefix[warp_in_block][p]
                        >= win_thresh) {
                    trim_end = p + W;
                    break;
                }
            }
        }

        d_stats[read_id].trim_start = trim_start;
        d_stats[read_id].trim_end   = trim_end;

        /* PolyG detection: scan from 3' end, detect run of >=10 Gs */
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

/**
 * CUDA kernel for per-read statistics computation (deprecated - use unified kernel)
 * Each thread processes one read
 */
__global__ void compute_read_stats_kernel(
    const char** d_sequences,
    const char** d_qualities,
    const int* d_read_lengths,
    int num_reads,
    char qual_threshold,
    struct ReadStatistics* d_stats
) {
    int read_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (read_id >= num_reads) {
        return;
    }
    
    const char* seq = d_sequences[read_id];
    const char* qual = d_qualities[read_id];
    int read_len = d_read_lengths[read_id];
    
    // Initialize counters
    int total_bases = read_len;
    int n_bases = 0;
    int low_qual_bases = 0;
    int total_quality = 0;
    
    // Process each base in the read
    for (int i = 0; i < read_len; i++) {
        char base = seq[i];
        char quality = qual[i];
        
        // Count N bases
        if (base == 'N' || base == 'n') {
            n_bases++;
        }
        
        // Count low quality bases
        if (quality < qual_threshold) {
            low_qual_bases++;
        }
        
        // Accumulate quality score (subtract 33 for Phred33 offset)
        total_quality += (quality - 33);
    }
    
    // Store results
    d_stats[read_id].total_bases = total_bases;
    d_stats[read_id].n_bases = n_bases;
    d_stats[read_id].low_qual_bases = low_qual_bases;
    d_stats[read_id].total_quality = total_quality;
    // trim_start, trim_end, polyG_trim_pos initialized to 0 by memset
}

/**
 * Wrapper function to allocate GPU memory and launch kernel
 */
int cuda_compute_read_stats(
    const char** sequences,
    const char** qualities,
    const int* read_lengths,
    int num_reads,
    char qual_threshold,
    struct ReadStatistics* stats
) {
    if (num_reads <= 0) {
        return -1;
    }
    
    // Device pointers
    char** d_sequences = NULL;
    char** d_qualities = NULL;
    int* d_read_lengths = NULL;
    struct ReadStatistics* d_stats = NULL;
    
    // Temporary buffers for sequence and quality data
    char* d_seq_data = NULL;
    char* d_qual_data = NULL;
    
    // Error checking
    cudaError_t err = cudaSuccess;
    
    // Declare all variables at the start (for goto safety)
    size_t total_seq_size = 0;
    size_t total_qual_size = 0;
    size_t seq_offset = 0;
    size_t qual_offset = 0;
    char** h_seq_ptrs = NULL;
    char** h_qual_ptrs = NULL;
    int block_size = 0;
    int grid_size = 0;
    int i = 0;
    
    // Allocate device memory for array of pointers
    err = cudaMalloc((void**)&d_sequences, num_reads * sizeof(char*));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed for sequences: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    err = cudaMalloc((void**)&d_qualities, num_reads * sizeof(char*));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed for qualities: %s\n", cudaGetErrorString(err));
        cudaFree(d_sequences);
        return -1;
    }
    
    err = cudaMalloc((void**)&d_read_lengths, num_reads * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed for read_lengths: %s\n", cudaGetErrorString(err));
        cudaFree(d_sequences);
        cudaFree(d_qualities);
        return -1;
    }
    
    err = cudaMalloc((void**)&d_stats, num_reads * sizeof(struct ReadStatistics));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed for stats: %s\n", cudaGetErrorString(err));
        cudaFree(d_sequences);
        cudaFree(d_qualities);
        cudaFree(d_read_lengths);
        return -1;
    }
    
    // Copy read lengths to device
    err = cudaMemcpy(d_read_lengths, read_lengths, num_reads * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy failed for read_lengths: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    // Calculate total data size and allocate combined buffer
    for (i = 0; i < num_reads; i++) {
        total_seq_size += read_lengths[i];
        total_qual_size += read_lengths[i];
    }
    
    // Allocate continuous memory for sequence and quality data
    err = cudaMalloc((void**)&d_seq_data, total_seq_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed for seq_data: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    err = cudaMalloc((void**)&d_qual_data, total_qual_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed for qual_data: %s\n", cudaGetErrorString(err));
        cudaFree(d_seq_data);
        goto cleanup;
    }
    
    // Copy sequence and quality data to device and build pointer arrays
    seq_offset = 0;
    qual_offset = 0;
    h_seq_ptrs = (char**)malloc(num_reads * sizeof(char*));
    h_qual_ptrs = (char**)malloc(num_reads * sizeof(char*));
    
    for (i = 0; i < num_reads; i++) {
        int len = read_lengths[i];
        
        // Copy sequence
        err = cudaMemcpy(d_seq_data + seq_offset, (void*)sequences[i], len, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA memcpy failed for sequence %d: %s\n", i, cudaGetErrorString(err));
            free(h_seq_ptrs);
            free(h_qual_ptrs);
            goto cleanup;
        }
        h_seq_ptrs[i] = d_seq_data + seq_offset;
        seq_offset += len;
        
        // Copy quality
        err = cudaMemcpy(d_qual_data + qual_offset, (void*)qualities[i], len, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA memcpy failed for quality %d: %s\n", i, cudaGetErrorString(err));
            free(h_seq_ptrs);
            free(h_qual_ptrs);
            goto cleanup;
        }
        h_qual_ptrs[i] = d_qual_data + qual_offset;
        qual_offset += len;
    }
    
    // Copy pointer arrays to device
    err = cudaMemcpy(d_sequences, (void*)h_seq_ptrs, num_reads * sizeof(char*), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy failed for seq pointers: %s\n", cudaGetErrorString(err));
        free(h_seq_ptrs);
        free(h_qual_ptrs);
        goto cleanup;
    }
    
    err = cudaMemcpy(d_qualities, (void*)h_qual_ptrs, num_reads * sizeof(char*), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy failed for qual pointers: %s\n", cudaGetErrorString(err));
        free(h_seq_ptrs);
        free(h_qual_ptrs);
        goto cleanup;
    }
    
    // Free temporary pointer arrays
    free(h_seq_ptrs);
    free(h_qual_ptrs);
    
    // Calculate grid and block dimensions
    block_size = min(BLOCK_SIZE, num_reads);
    grid_size = (num_reads + block_size - 1) / block_size;
    
    // Launch kernel
    compute_read_stats_kernel<<<grid_size, block_size>>>(
        (const char**)d_sequences, (const char**)d_qualities, d_read_lengths, num_reads, qual_threshold, d_stats
    );
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA device synchronize failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    // Copy results back to host
    err = cudaMemcpy(stats, d_stats, num_reads * sizeof(struct ReadStatistics), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy failed for results: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
cleanup:
    // Free device memory
    if (d_sequences) cudaFree(d_sequences);
    if (d_qualities) cudaFree(d_qualities);
    if (d_read_lengths) cudaFree(d_read_lengths);
    if (d_stats) cudaFree(d_stats);
    if (d_seq_data) cudaFree(d_seq_data);
    if (d_qual_data) cudaFree(d_qual_data);
    
    return (err != cudaSuccess) ? -1 : 0;
}

/**
 * Check if CUDA is available
 */
int cuda_is_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    if (err != cudaSuccess || device_count == 0) {
        return 0;
    }
    
    return 1;
}

/**
 * Get CUDA device
 */
int cuda_get_device() {
    int device = -1;
    
    if (!cuda_is_available()) {
        return -1;
    }
    
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        return -1;
    }
    
    return device;
}

/* ===================================================================
 * BGZF-GPU FASTQ parsing kernels — Future Direction 1
 *
 * Warp-cooperative newline scanning + read-descriptor extraction.
 * Enables GPU-resident FASTQ parsing after nvCOMP BGZF decompression,
 * eliminating the Device→Host→Device round-trip for sequence pointers.
 *
 * Typical call chain:
 *   CudaGzipDecompressor::decompress() → d_decompressed_flat (device)
 *   cuda_fastq_parse_device(d_buf, buf_len, d_descs, d_count, stream)
 *   → d_descs[0..count-1] hold GPU-resident sequence/quality offsets
 *   → feed to a GpuReadDescriptor-aware stats kernel (no D→H copy needed)
 * =================================================================== */

#ifdef HAVE_CUDA

/**
 * Kernel 1: warp-cooperative newline position scanner.
 *
 * Each thread checks one byte.  Within a warp, __ballot_sync marks which
 * lanes found '\n'; the leading lane atomically claims 'count' consecutive
 * slots in d_newline_pos, then each matching lane writes its byte offset.
 *
 * @param d_buf         Device buffer of decompressed FASTQ bytes
 * @param buf_len       Valid bytes in d_buf
 * @param d_newline_pos Device array receiving newline byte-offsets
 *                      (caller pre-allocates >= buf_len/2 elements)
 * @param d_nl_count    Device uint32 accumulating total newline count
 */
__global__ void fastq_scan_newlines_kernel(
    const char* __restrict__ d_buf,
    size_t    buf_len,
    uint32_t* d_newline_pos,
    uint32_t* d_nl_count)
{
    int pos  = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int lane = threadIdx.x & 31;

    bool is_nl = ((size_t)pos < buf_len) && (d_buf[pos] == '\n');

    /* Warp ballot: bit k set iff lane k found a newline */
    unsigned ballot = __ballot_sync(0xffffffff, is_nl);
    int count = __popc(ballot);
    if (count == 0) return;

    /* Lane 0 atomically reserves 'count' consecutive output slots */
    uint32_t base = 0;
    if (lane == 0)
        base = atomicAdd(d_nl_count, (uint32_t)count);
    base = __shfl_sync(0xffffffff, base, 0);   /* broadcast to all lanes */

    if (is_nl) {
        /* Rank of this lane among newline-finding lanes in the warp */
        unsigned below = ballot & ((1u << lane) - 1u);
        d_newline_pos[base + __popc(below)] = (uint32_t)pos;
    }
}

/**
 * Kernel 2: FASTQ read-descriptor extractor.
 *
 * One thread per FASTQ record.  FASTQ records consist of exactly 4 newline-
 * terminated lines.  Given a sorted newline-position array (produced by
 * Kernel 1), groups of four newlines decode to one GpuReadDescriptor:
 *
 *   nl[4r+0]  end of @header line  → sequence starts at nl[4r+0]+1
 *   nl[4r+1]  end of sequence line → seq_len = nl[4r+1] - nl[4r+0] - 1
 *   nl[4r+2]  end of '+' line      → quality starts at nl[4r+2]+1
 *   nl[4r+3]  end of quality line  (qual_len == seq_len, not stored)
 *
 * @param d_nl          Newline offsets from Kernel 1 (device)
 * @param d_nl_cnt_ptr  Device pointer to total newline count
 * @param d_descs       Output GpuReadDescriptor array (device)
 * @param d_read_count  Output: number of complete records written
 */
__global__ void fastq_extract_descriptors_kernel(
    const uint32_t* __restrict__ d_nl,
    const uint32_t* __restrict__ d_nl_cnt_ptr,
    struct GpuReadDescriptor*    d_descs,
    uint32_t*                    d_read_count)
{
    uint32_t nl_count  = *d_nl_cnt_ptr;   /* safe: sequential stream ordering */
    uint32_t num_reads = nl_count / 4;

    /* Thread 0 stores the record count for the host */
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *d_read_count = num_reads;

    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_reads) return;

    d_descs[r].seq_offset  = d_nl[4*r + 0] + 1;
    d_descs[r].seq_len     = d_nl[4*r + 1] - d_nl[4*r + 0] - 1;
    d_descs[r].qual_offset = d_nl[4*r + 2] + 1;
    /* qual_len == seq_len always in well-formed FASTQ */
}

/**
 * Host wrapper: parse all FASTQ records from a GPU-resident byte buffer.
 *
 * All operations are stream-ordered; temporary device allocations use
 * cudaMallocAsync/cudaFreeAsync so they are freed only after the kernels
 * consuming them complete.
 *
 * @param d_buf    Device buffer of decompressed FASTQ (from nvCOMP/BGZF)
 * @param buf_len  Valid bytes in d_buf
 * @param d_descs  Pre-allocated device array (>=buf_len/8 GpuReadDescriptor)
 * @param d_count  Pre-allocated device uint32; receives record count
 * @param stream   CUDA stream for all operations
 * @return 0 on success, -1 on CUDA error
 */
int cuda_fastq_parse_device(
    const char*               d_buf,
    size_t                    buf_len,
    struct GpuReadDescriptor* d_descs,
    uint32_t*                 d_count,
    cudaStream_t              stream)
{
    if (!d_buf || buf_len == 0) return -1;

    /* Upper bound on newlines: at most 1 per 2 bytes in valid FASTQ */
    size_t max_nl = buf_len / 2 + 1;

    uint32_t* d_nl_pos = nullptr;
    uint32_t* d_nl_cnt = nullptr;

    cudaError_t err;
    err = cudaMallocAsync(&d_nl_pos, max_nl * sizeof(uint32_t), stream);
    if (err != cudaSuccess) return -1;
    err = cudaMallocAsync(&d_nl_cnt, sizeof(uint32_t), stream);
    if (err != cudaSuccess) { cudaFreeAsync(d_nl_pos, stream); return -1; }

    cudaMemsetAsync(d_nl_cnt, 0, sizeof(uint32_t), stream);

    /* Kernel 1: scan each byte for newlines (1 thread per byte) */
    const int SCAN_THREADS = 256;
    int scan_blocks = (int)((buf_len + SCAN_THREADS - 1) / SCAN_THREADS);
    fastq_scan_newlines_kernel<<<scan_blocks, SCAN_THREADS, 0, stream>>>(
        d_buf, buf_len, d_nl_pos, d_nl_cnt);

    /* Kernel 2: extract descriptors (1 thread per record).
     * Grid is sized for the maximum possible records = buf_len/8,
     * where 8 bytes is the minimum for a single-base FASTQ record.
     * Threads beyond the actual record count exit via the r >= num_reads guard. */
    size_t max_reads = buf_len / 8 + 1;
    const int DESC_THREADS = 256;
    int desc_blocks = (int)((max_reads + DESC_THREADS - 1) / DESC_THREADS);
    fastq_extract_descriptors_kernel<<<desc_blocks, DESC_THREADS, 0, stream>>>(
        d_nl_pos, d_nl_cnt, d_descs, d_count);

    err = cudaGetLastError();

    /* Stream-ordered free: released only after both kernels complete */
    cudaFreeAsync(d_nl_pos, stream);
    cudaFreeAsync(d_nl_cnt, stream);

    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA FASTQ parse kernel launch failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

#endif /* HAVE_CUDA */

/**
 * Launch warp-per-read kernel on device-side pointers using a CUDA stream.
 * Does not perform any host<->device memory transfers.
 *
 * Grid sizing: each read requires 1 warp (32 threads).
 *   BLOCK_SIZE = 256 -> READS_PER_BLOCK = 8.
 *   grid_size  = ceil(num_reads / 8)
 *   For 8192 reads -> 1024 blocks -> covers all 108 A100 SMs ~9.5 times.
 *
 * @param trim_window_size  Sliding-window width for quality trimming.
 *   1 (default): single-base threshold scan (original behaviour).
 *   >1: cooperative warp prefix-sum in Phase 2a, window-average scan in Phase 2b.
 *       Maps to Options::qualityCut.windowSizeShared.
 */
int cuda_compute_read_stats_device(
    char** d_sequences,
    char** d_qualities,
    int* d_read_lengths,
    int num_reads,
    char qual_threshold,
    struct ReadStatistics* d_stats,
    cudaStream_t stream,
    int trim_window_size
) {
    if (num_reads <= 0) return -1;

    /* 32 threads per read, BLOCK_SIZE threads per block → READS_PER_BLOCK reads per block */
    int grid_size = (num_reads + READS_PER_BLOCK - 1) / READS_PER_BLOCK;

    compute_read_stats_warp_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        (const char**)d_sequences, (const char**)d_qualities, d_read_lengths,
        num_reads, qual_threshold, d_stats, trim_window_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA warp kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

// ===================================================================
// Combined filter + post-filter statRead kernel
// ===================================================================

// Filter result constants (mirror common.h values)
#define GPU_PASS_FILTER    0
#define GPU_FAIL_N_BASE   12
#define GPU_FAIL_LENGTH   16
#define GPU_FAIL_TOO_LONG 17
#define GPU_FAIL_QUALITY  20

/**
 * Combined kernel: per-read filter decision + batch-aggregated post-filter stats.
 *
 * Warp-per-read design (same as compute_read_stats_warp_kernel).
 *
 *   Phase 1: all 32 lanes compute n_bases, low_qual, total_qual via stride-32.
 *   Phase 2: lane 0 applies filter rules → d_filter_results[read_id].
 *   Phase 3: for PASSING reads, all 32 lanes cooperatively accumulate per-cycle
 *            quality statistics into d_batch_post_stats using atomicAdd.
 *            Lane 0 additionally computes kmer frequencies (sequential dependency).
 */
__global__ void filter_and_stats_warp_kernel(
    const char** d_sequences,
    const char** d_qualities,
    const int* d_read_lengths,
    int num_reads,
    char qual_threshold,
    int* d_filter_results,
    struct GpuBatchPostStats* d_batch_post_stats,
    int trim_window_size,
    int unqual_percent_limit,
    int avg_qual_req,
    int n_base_limit,
    int length_required,
    int max_length,
    bool qual_filter_enabled,
    bool length_filter_enabled,
    bool stats_only  // When true: skip filter, compute stats for ALL reads
) {
    const int WARP_SIZE = 32;
    int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int read_id       = global_thread / WARP_SIZE;
    int lane          = global_thread % WARP_SIZE;

    if (read_id >= num_reads) return;

    const char* seq   = d_sequences[read_id];
    const char* qual  = d_qualities[read_id];
    int read_len      = d_read_lengths[read_id];

    /* ------------------------------------------------------------------ */
    /* Phase 1: per-base stats – all 32 lanes work in parallel             */
    /* ------------------------------------------------------------------ */
    int n_bases    = 0;
    int low_qual   = 0;
    int total_qual = 0;

    for (int i = lane; i < read_len; i += WARP_SIZE) {
        char c = seq[i];
        char q = qual[i];
        if (c == 'N' || c == 'n') n_bases++;
        if (q < qual_threshold)   low_qual++;
        total_qual += (int)(unsigned char)(q - 33);
    }

    /* Warp-level reduction via shuffle */
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        n_bases    += __shfl_down_sync(mask, n_bases,    offset);
        low_qual   += __shfl_down_sync(mask, low_qual,   offset);
        total_qual += __shfl_down_sync(mask, total_qual, offset);
    }

    /* ------------------------------------------------------------------ */
    /* Phase 2: filter decision – lane 0 only (skipped in stats_only mode) */
    /* ------------------------------------------------------------------ */
    int filter_result = GPU_PASS_FILTER;
    if (!stats_only) {
        if (lane == 0) {
            if (qual_filter_enabled) {
                if (low_qual > (unqual_percent_limit * read_len / 100))
                    filter_result = GPU_FAIL_QUALITY;
                else if (avg_qual_req > 0 && read_len > 0 && (total_qual / read_len) < avg_qual_req)
                    filter_result = GPU_FAIL_QUALITY;
                else if (n_bases > n_base_limit)
                    filter_result = GPU_FAIL_N_BASE;
            }
            if (filter_result == GPU_PASS_FILTER && length_filter_enabled) {
                if (read_len < length_required)
                    filter_result = GPU_FAIL_LENGTH;
                else if (max_length > 0 && read_len > max_length)
                    filter_result = GPU_FAIL_TOO_LONG;
            }
            d_filter_results[read_id] = filter_result;
        }
        /* Broadcast filter result to all lanes in the warp */
        filter_result = __shfl_sync(mask, filter_result, 0);
    }

    /* ------------------------------------------------------------------ */
    /* Phase 3: statRead – all 32 lanes (PASS reads only, or all in stats_only) */
    /* ------------------------------------------------------------------ */
    if (filter_result != GPU_PASS_FILTER) return;

    /* Accumulate per-cycle quality histograms using atomicAdd on global mem */
    const char q20_char = '5';   /* ASCII 53 = Phred 20 + 33 */
    const char q30_char = '?';   /* ASCII 63 = Phred 30 + 33 */

    for (int i = lane; i < read_len; i += WARP_SIZE) {
        char base = seq[i];
        char q    = qual[i];
        int  b    = base & 0x07;   /* base → 0..7 index */
        int  phred = (int)(unsigned char)(q - 33);

        atomicAdd(&d_batch_post_stats->cycle_base_contents[b][i], 1);
        atomicAdd(&d_batch_post_stats->cycle_base_qual[b][i], phred);
        atomicAdd(&d_batch_post_stats->cycle_total_base[i], 1);
        atomicAdd(&d_batch_post_stats->cycle_total_qual[i], phred);

        if (q >= q30_char) {
            atomicAdd(&d_batch_post_stats->cycle_q30[b][i], 1);
            atomicAdd(&d_batch_post_stats->cycle_q20[b][i], 1);
        } else if (q >= q20_char) {
            atomicAdd(&d_batch_post_stats->cycle_q20[b][i], 1);
        }

        atomicAdd(&d_batch_post_stats->base_qual_histogram[(unsigned char)q], 1);
    }

    /* Lane 0: kmer frequencies (sequential dependency on kmer state) */
    if (lane == 0) {
        atomicAdd(&d_batch_post_stats->reads_passed, 1);
        atomicAdd((unsigned long long*)&d_batch_post_stats->length_sum, (unsigned long long)read_len);

        int kmer = 0;
        bool need_full = true;
        for (int i = 0; i < read_len; i++) {
            char base = seq[i];
            if (base == 'N' || base == 'n') { need_full = true; continue; }
            if (i < 4) continue;

            int val = -1;
            switch (base) {
                case 'A': case 'a': val = 0; break;
                case 'T': case 't': val = 1; break;
                case 'C': case 'c': val = 2; break;
                case 'G': case 'g': val = 3; break;
            }
            if (val < 0) { need_full = true; continue; }

            if (!need_full) {
                kmer = ((kmer << 2) & 0x3FC) | val;
                atomicAdd(&d_batch_post_stats->kmer[kmer], 1);
            } else {
                bool valid = true;
                kmer = 0;
                for (int k = 0; k < 5; k++) {
                    char b2 = seq[i - 4 + k];
                    int v2 = -1;
                    switch (b2) {
                        case 'A': case 'a': v2 = 0; break;
                        case 'T': case 't': v2 = 1; break;
                        case 'C': case 'c': v2 = 2; break;
                        case 'G': case 'g': v2 = 3; break;
                    }
                    if (v2 < 0) { valid = false; break; }
                    kmer = ((kmer << 2) & 0x3FC) | v2;
                }
                if (!valid) { need_full = true; continue; }
                atomicAdd(&d_batch_post_stats->kmer[kmer], 1);
                need_full = false;
            }
        }
    }
}

/**
 * Launch the combined filter+stats kernel on a CUDA stream.
 */
int cuda_filter_and_stats_device(
    char** d_sequences,
    char** d_qualities,
    int* d_read_lengths,
    int num_reads,
    char qual_threshold,
    int* d_filter_results,
    struct GpuBatchPostStats* d_batch_post_stats,
    cudaStream_t stream,
    int trim_window_size,
    int unqual_percent_limit,
    int avg_qual_req,
    int n_base_limit,
    int length_required,
    int max_length,
    bool qual_filter_enabled,
    bool length_filter_enabled,
    bool stats_only
) {
    if (num_reads <= 0) return -1;

    int grid_size = (num_reads + READS_PER_BLOCK - 1) / READS_PER_BLOCK;

    filter_and_stats_warp_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        (const char**)d_sequences, (const char**)d_qualities, d_read_lengths,
        num_reads, qual_threshold, d_filter_results, d_batch_post_stats,
        trim_window_size, unqual_percent_limit, avg_qual_req,
        n_base_limit, length_required, max_length,
        qual_filter_enabled, length_filter_enabled, stats_only
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA filter+stats kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}
