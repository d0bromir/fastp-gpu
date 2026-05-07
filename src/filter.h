#ifndef FILTER_H
#define FILTER_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include "options.h"
#include "read.h"

// Forward declaration for GPU support
class CudaStatsWrapper;
struct ReadStatistics;
struct GpuBatchPostStats;
class Stats;

using namespace std;

class Filter{
public:
    Filter(Options* opt);
    ~Filter();
    int passFilter(Read* r);
    bool passLowComplexityFilter(Read* r);
    Read* trimAndCut(Read* r, int front, int tail, int& frontTrimmed);
    bool filterByIndex(Read* r);
    bool filterByIndex(Read* r1, Read* r2);
    
    // GPU-accelerated batch filtering
    int filterBatchGPU(const vector<Read*>& reads, vector<int>& filter_results);
    // GPU-accelerated batch filtering for paired-end reads
    int filterBatchGPU(const vector<Read*>& reads1, const vector<Read*>& reads2, 
                       vector<int>& filter_results1, vector<int>& filter_results2);

    // Combined GPU filter + post-filter stats (eliminates CPU statRead for post-filter)
    int filterBatchGPUWithStats(const vector<Read*>& reads,
                                vector<int>& filter_results,
                                Stats* postStats);
    int filterBatchGPUWithStats(const vector<Read*>& reads1, const vector<Read*>& reads2,
                                vector<int>& filter_results1, vector<int>& filter_results2,
                                Stats* postStats1, Stats* postStats2);

    // GPU-accelerated pre-filter statistics (replaces CPU statRead for pre-filter)
    int preStatsBatchGPU(const vector<Read*>& reads, Stats* preStats);
    
    // GPU-accelerated trimming operations
    Read* trimAndCutGPU(Read* r, int front, int tail, int& frontTrimmed);
    int trimPolyGGPU(Read* r1, Read* r2);
    int trimQualityGPU(Read* r, int window_size, int quality_threshold);

    /**
     * Apply pass/fail decision using pre-computed GPU statistics.
     * Avoids re-scanning the read bytes that the GPU already processed.
     */
    int passFilterWithStats(Read* r, const ReadStatistics& stats);

    // Check if GPU acceleration is available
    bool hasGPU() const { return mNumGPUs > 0; }

    // Profiling: print GPU vs CPU filtering timing summary
    void printProfilingStats();

    static bool test();

private:
    bool match(vector<string>& list, string target, int threshold);

private:
    Options* mOptions;
    // Up to two GPU wrappers — one per CUDA device (dual-GPU A100 support).
    // mGpuStats[i] is non-null when CUDA device i was successfully initialised.
    CudaStatsWrapper* mGpuStats[2];
    int mNumGPUs;                          // 0 = CPU-only, 1 = single GPU, 2 = dual GPU
    std::atomic<int> mGpuSelector;         // Round-robin device picker
    bool mGpuInitDone;                     // Lazy CUDA initialization flag
    std::mutex mGpuInitMutex;              // Guard for thread-safe lazy init
    void ensureGPUInit();                  // Deferred GPU init (called on first GPU use)

    // Profiling counters
    std::atomic<long long> mCpuFilterNs{0};    // total ns in CPU passFilter fallback
    std::atomic<long long> mCpuFilterReads{0}; // reads filtered by CPU path
    std::atomic<long long> mGpuFilterNs{0};    // total ns in GPU filterBatchGPU (including transfer)
    std::atomic<long long> mGpuFilterReads{0}; // reads filtered by GPU path
    std::atomic<long long> mFilterBatches{0};  // total filterBatchGPU calls
};


#endif