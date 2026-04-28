#ifndef CUDA_STATS_WRAPPER_H
#define CUDA_STATS_WRAPPER_H

#include "read.h"
#include <vector>
#include <mutex>
#include <atomic>

using namespace std;

// Define ReadStatistics struct that doesn't depend on CUDA
// This is used both for CUDA and CPU paths
struct ReadStatistics {
    int total_bases;      // Total number of bases in read
    int n_bases;          // Number of N bases
    int low_qual_bases;   // Number of low quality bases
    int total_quality;    // Sum of all quality scores (adjusted from ASCII)
    
    // Trim information (computed on GPU to avoid CPU overhead)
    int trim_start;       // Trim position at start (after quality trimming and polyG trim)
    int trim_end;         // Trim position at end (0 = no trimming, keep all from trim_start)
    int polyG_trim_pos;   // Position where PolyG trimming would end (or -1 if no polyG)
};

// ---- GPU batch-aggregated post-filter statistics ----
// Accumulated on-GPU using atomicAdd for all reads that PASS the filter.
// Transferred D2H once per batch, eliminating CPU statRead for post-filter.
#define GPU_STATS_MAX_CYCLES 512
#define GPU_STATS_NUM_BASES  8     // base & 0x07 maps A=1,T=4,C=3,G=7,N=6
#define GPU_STATS_KMER_COUNT 1024  // 4^5 = 1024 possible 5-mers

struct GpuBatchPostStats {
    int cycle_base_contents[GPU_STATS_NUM_BASES][GPU_STATS_MAX_CYCLES];
    int cycle_base_qual[GPU_STATS_NUM_BASES][GPU_STATS_MAX_CYCLES];
    int cycle_q20[GPU_STATS_NUM_BASES][GPU_STATS_MAX_CYCLES];
    int cycle_q30[GPU_STATS_NUM_BASES][GPU_STATS_MAX_CYCLES];
    int cycle_total_base[GPU_STATS_MAX_CYCLES];
    int cycle_total_qual[GPU_STATS_MAX_CYCLES];
    int base_qual_histogram[128];
    int kmer[GPU_STATS_KMER_COUNT];
    int reads_passed;
    long long length_sum;
};

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

/**
 * C++ Wrapper class for CUDA-accelerated per-read statistics
 * Provides convenient interface for the fastp codebase
 */
class CudaStatsWrapper {
public:
    /**
     * @param device  CUDA device index to use (0-based). Defaults to 0.
     */
    explicit CudaStatsWrapper(int device = 0);
    ~CudaStatsWrapper();

    /**
     * Process a batch of reads: compute filter decisions + post-filter statistics on GPU.
     * Returns per-read filter results and batch-aggregated post-filter stats.
     * Thread-safe: uses internal slot pool with per-slot mutexes.
     *
     * @param reads           Vector of Read pointers
     * @param qual_threshold  Phred quality threshold (without +33 offset)
     * @param filter_results  Output vector of per-read filter results (PASS_FILTER or fail code)
     * @param batch_post_stats Output: GPU-aggregated post-filter statistics for passing reads
     * @param trim_window_size Sliding-window width (1 = single-base)
     * @param unqual_percent_limit  qualfilter.unqualifiedPercentLimit (0-100)
     * @param avg_qual_req       qualfilter.avgQualReq (0 = disabled)
     * @param n_base_limit       qualfilter.nBaseLimit
     * @param length_required    lengthFilter.requiredLength (0 = disabled)
     * @param max_length         lengthFilter.maxLength (0 = disabled)
     * @param qual_filter_enabled  whether quality filtering is on
     * @param length_filter_enabled  whether length filtering is on
     * @return 0 on success, non-zero on failure
     */
    int processBatchFilterAndStats(
        const vector<Read*>& reads,
        int qual_threshold,
        vector<int>& filter_results,
        struct GpuBatchPostStats& batch_post_stats,
        int trim_window_size,
        int unqual_percent_limit,
        int avg_qual_req,
        int n_base_limit,
        int length_required,
        int max_length,
        bool qual_filter_enabled,
        bool length_filter_enabled
    );

    /**
     * Process a batch of reads and compute ONLY statistics on GPU (no filtering).
     * Used for pre-filter statistics on original reads.
     */
    int processBatchStatsOnly(
        const vector<Read*>& reads,
        int qual_threshold,
        struct GpuBatchPostStats& batch_post_stats
    );

    /**
     * Process a batch of reads and compute statistics on GPU.
     * Thread-safe: protected by an internal mutex so multiple worker threads
     * sharing the same wrapper won't corrupt each other's buffers.
     * @param reads           Vector of Read pointers
     * @param qual_threshold  Phred quality threshold (without +33 offset)
     * @param stats           Output vector of ReadStatistics
     * @param trim_window_size Sliding-window width (1 = single-base, >1 = window avg)
     * @return 0 on success, non-zero on failure (caller should fall back to CPU)
     */
    int processBatch(
        const vector<Read*>& reads,
        int qual_threshold,
        vector<ReadStatistics>& stats,
        int trim_window_size = 1
    );

    bool isGPUAvailable();
    int getGPUDevice();

    /** Return the number of CUDA devices visible to this process. */
    static int getDeviceCount();

    // ---- Profiling counters (thread-safe, lock-free) ----
    struct ProfilingStats {
        std::atomic<long long> total_gpu_ns{0};   // total wall-clock ns inside processSlot
        std::atomic<long long> h2d_ns{0};         // host-to-device transfer
        std::atomic<long long> kernel_ns{0};      // kernel execution
        std::atomic<long long> d2h_ns{0};         // device-to-host transfer
        std::atomic<long long> pack_ns{0};        // CPU-side packing into pinned buffers
        std::atomic<long long> batch_count{0};    // number of processBatch calls
        std::atomic<long long> read_count{0};     // total reads processed on GPU
    };
    ProfilingStats profiling;
    void printProfilingStats() const;

private:
    bool gpu_available;
    int device_id;

    // -----------------------------------------------------------------------
    // Concurrent slot pool – replaces the old single-mutex double-buffer design.
    //
    // Problem with the old design: a single gpuMutex serialised ALL worker
    // threads for one device.  With 4 CPU threads per GPU, only ONE kernel was
    // ever running at a time, leaving the other 3 threads (and 75% of the data
    // they had ready) waiting.  Result: ~1% SM utilisation.
    //
    // New design: NUM_SLOTS independent slots, each with its own pinned host
    // buffers, GPU device buffers, CUDA stream, and per-slot mutex.
    // A CPU worker calls processBatch(), which:
    //   1. Tries each slot mutex (try_lock, round-robin).
    //   2. On success, packs data into that slot's pinned host buffers,
    //      fires cudaMemcpyAsync + kernel + cudaMemcpyAsync on the slot's stream,
    //      then cudaStreamSynchronize → only blocks peer threads from this slot.
    //   3. Falls back to blocking on slot 0 if all are busy.
    //
    // With NUM_SLOTS = 8, up to 8 kernels run concurrently per device.
    // Supports -w 32 (16 threads per GPU) while keeping all slots busy.
    static const int NUM_SLOTS = 8;

#ifdef HAVE_CUDA
    struct GpuSlot {
        cudaStream_t   stream      = 0;

        // Device-side buffers
        char*          d_seq_buf   = nullptr;   // packed sequences (contig)
        char*          d_qual_buf  = nullptr;   // packed qualities
        char**         d_seq_ptrs  = nullptr;   // per-read pointers into d_seq_buf
        char**         d_qual_ptrs = nullptr;
        int*           d_read_lens = nullptr;
        struct ReadStatistics* d_stats = nullptr;

        // Combined filter+stats buffers
        int*                      d_filter_results    = nullptr;
        struct GpuBatchPostStats* d_batch_post_stats  = nullptr;

        // Pinned host-side mirrors
        char*          h_seq_buf   = nullptr;
        char*          h_qual_buf  = nullptr;
        char**         h_seq_ptrs  = nullptr;
        char**         h_qual_ptrs = nullptr;
        int*           h_read_lens = nullptr;
        struct ReadStatistics* h_stats = nullptr;

        // Pinned host mirrors for combined filter+stats
        int*                      h_filter_results    = nullptr;
        struct GpuBatchPostStats* h_batch_post_stats  = nullptr;

        int            max_reads   = 0;
        size_t         buf_bytes   = 0;   // bytes allocated in each seq/qual buf

        std::mutex     slot_mutex;        // guards this slot only
    };

    GpuSlot slots[NUM_SLOTS];

    // Slot allocation round-robin hint (atomic for cheap try-lock loop)
    std::atomic<int> next_slot_hint{0};

    void allocateSlot(GpuSlot& slot, int max_reads, size_t buf_bytes);
    void freeSlot(GpuSlot& slot);
    int  processSlot(GpuSlot& slot, const vector<Read*>& reads,
                     int qual_threshold, vector<ReadStatistics>& stats,
                     int trim_window_size);
    int  processSlotFilterAndStats(GpuSlot& slot, const vector<Read*>& reads,
                     int qual_threshold, vector<int>& filter_results,
                     struct GpuBatchPostStats& batch_post_stats,
                     int trim_window_size,
                     int unqual_percent_limit, int avg_qual_req,
                     int n_base_limit, int length_required, int max_length,
                     bool qual_filter_enabled, bool length_filter_enabled);
#endif
};

#endif
