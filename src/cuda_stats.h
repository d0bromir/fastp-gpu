#ifndef CUDA_STATS_H
#define CUDA_STATS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cuda_stats_wrapper.h"  // Include to get ReadStatistics + GpuBatchPostStats definitions

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

/**
 * Per-read statistics structure for GPU computation
 * Used to return computed statistics and trim information from GPU kernel
 * NOTE: Definition moved to cuda_stats_wrapper.h for both GPU and CPU compatibility
 */

/**
 * GPU-resident FASTQ read descriptor produced by cuda_fastq_parse_device().
 * Holds byte offsets into the decompressed FASTQ device buffer so that
 * sequence and quality data never need to be copied to host memory.
 * qual_len == seq_len always holds in well-formed FASTQ.
 */
struct GpuReadDescriptor {
    uint32_t seq_offset;    /* byte offset of sequence within d_buf */
    uint32_t seq_len;       /* sequence length in bytes              */
    uint32_t qual_offset;   /* byte offset of quality within d_buf  */
};

/**
 * Trimmed read output structure for GPU-side trim application
 * Returned from GPU with pre-trimmed sequence and quality data
 * Strategy 1: Keep Data on GPU Longer - trims are applied on GPU before D2H transfer
 */
struct TrimmedReadOutput {
    int trimmed_seq_len;   // Length of trimmed sequence (trim_end - trim_start)
    int trimmed_qual_len;  // Length of trimmed quality
    // Followed by:
    // - trimmed_seq_len bytes of sequence data
    // - trimmed_qual_len bytes of quality data
    // Data layout in packed buffer: [TrimmedReadOutput header][seq_data][qual_data]
};

/**
 * GPU-accelerated per-read statistics computation
 * Computes: base count, N-rate, low quality count, and average quality
 * 
 * @param sequences Array of sequence strings (host memory)
 * @param qualities Array of quality strings (host memory)
 * @param read_lengths Array of read lengths
 * @param num_reads Number of reads to process
 * @param qual_threshold Quality threshold for "low quality" classification (ASCII value)
 * @param stats Output array of statistics (preallocated, host memory)
 * @return 0 on success, non-zero on error
 */
#ifdef HAVE_CUDA
int cuda_compute_read_stats(
    const char** sequences,
    const char** qualities,
    const int* read_lengths,
    int num_reads,
    char qual_threshold,
    struct ReadStatistics* stats
);
#endif

// Variant that operates on device-side pointers and launches the kernel on the provided CUDA stream.
// d_stats is expected to be a device pointer; the caller is responsible for copying results back to host.
// trim_window_size: 1 = original single-base scan; >1 = sliding-window via warp prefix-sum.
#ifdef HAVE_CUDA
int cuda_compute_read_stats_device(
    char** d_sequences,
    char** d_qualities,
    int* d_read_lengths,
    int num_reads,
    char qual_threshold,
    struct ReadStatistics* d_stats,
    cudaStream_t stream,
    int trim_window_size = 1
);

/**
 * Combined GPU kernel: filter decision + post-filter statRead accumulation.
 *
 * For each read, computes filter stats (n_bases, low_qual, total_quality, trim)
 * and writes a pass/fail result to d_filter_results.  For passing reads,
 * atomicAdd-accumulates per-cycle quality histograms into d_batch_post_stats,
 * eliminating the need for CPU-side statRead entirely.
 *
 * @param d_filter_results  Output array (num_reads ints): PASS_FILTER (0) or fail code
 * @param d_batch_post_stats Must be zero-initialised before the call
 * @param unqual_percent_limit  qualfilter.unqualifiedPercentLimit (0-100)
 * @param avg_qual_req       qualfilter.avgQualReq (0 = disabled)
 * @param n_base_limit       qualfilter.nBaseLimit
 * @param length_required    lengthFilter.requiredLength  (0 = disabled)
 * @param max_length         lengthFilter.maxLength       (0 = disabled)
 * @param qual_filter_enabled  whether quality filtering is on
 * @param length_filter_enabled  whether length filtering is on
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
    bool stats_only = false
);

/**
 * Parse FASTQ records from a GPU-resident decompressed-BGZF byte buffer.
 * Produces one GpuReadDescriptor per read without any Device->Host transfer.
 *
 * @param d_buf    Device buffer of decompressed FASTQ bytes
 * @param buf_len  Valid bytes in d_buf
 * @param d_descs  Pre-allocated device array (>= buf_len/8 elements)
 * @param d_count  Pre-allocated device uint32; receives record count
 * @param stream   CUDA stream (all ops are stream-ordered)
 * @return 0 on success, -1 on error
 */
int cuda_fastq_parse_device(
    const char*               d_buf,
    size_t                    buf_len,
    struct GpuReadDescriptor* d_descs,
    uint32_t*                 d_count,
    cudaStream_t              stream
);
#endif

/**
 * Check if CUDA is available
 * @return 1 if CUDA is available, 0 otherwise
 */
int cuda_is_available();

/**
 * Get CUDA device properties
 * @return Device ID if available, -1 otherwise
 */
int cuda_get_device();

#endif
