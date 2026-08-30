#include "processor.h"
#include "peprocessor.h"
#include "seprocessor.h"
#include "overlapanalysis.h"
#include "util.h"
#include "cuda_trim.h"
#include "polyx.h"
#include "cuda_stats_wrapper.h"  // ReadStatistics + CudaStatsWrapper + GpuBatchPostStats
#include "stats.h"               // Stats::mergeBatchStats
#include <sys/stat.h>
#include <cstdlib>
#ifdef FASTP_PROFILING
#include <chrono>
#endif

// Minimum total compressed (or raw) input size in bytes for which GPU
// initialization is worthwhile. Below this, the ~3.9 s CUDA context +
// Size gate (FASTP_GPU_MIN_BYTES, default 0 = off): when combined input
// size is below the threshold, skip GPU init.  As of v1.3.3+ the
// constructor fires CUDA init asynchronously and overlaps it with reader
// I/O / worker spin-up, so the ~3.9 s init cost is hidden on any input
// large enough to take longer than 4 s on the CPU path (panels and up).
// The gate is therefore disabled by default; set FASTP_GPU_MIN_BYTES to
// a positive byte count to re-enable a hard cutoff (e.g. for degenerate
// inputs like the testdata unit tests where there is nothing to overlap
// init with).
static constexpr long long FASTP_GPU_MIN_BYTES_DEFAULT = 0LL;

static long long fastp_input_size_bytes(const std::string& path) {
    if (path.empty()) return 0;
    if (path == "/dev/stdin" || path == "-") return -1;   // unknown -> assume large
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return -1;          // unknown -> assume large
    return static_cast<long long>(st.st_size);
}

static long long fastp_gpu_min_bytes() {
    const char* env = std::getenv("FASTP_GPU_MIN_BYTES");
    if (env && *env) {
        char* end = nullptr;
        long long v = std::strtoll(env, &end, 10);
        if (end != env && v >= 0) return v;
    }
    return FASTP_GPU_MIN_BYTES_DEFAULT;
}

Filter::Filter(Options* opt){
    mOptions = opt;
    mGpuStats[0] = mGpuStats[1] = nullptr;
    mNumGPUs = 0;
    mGpuSelector.store(0);
    mGpuInitDone = false;
    // Fire CUDA init asynchronously so the ~3.9 s setup overlaps with
    // reader/worker thread spin-up.  ensureGPUInit() is mutex-protected
    // and idempotent; if a worker hits the dispatch site before init
    // finishes it will block on the same mutex.  The future is joined
    // in the destructor so we never tear down with an outstanding init.
    // A/B switch: FASTP_ASYNC_INIT=0 disables the async path so the
    // overlap effect can be measured.
    bool use_async = true;
    if (const char* env = std::getenv("FASTP_ASYNC_INIT")) {
        if (env[0] == '0') use_async = false;
    }
    if (use_async) {
        mGpuInitFuture = std::async(std::launch::async, [this]() {
            this->ensureGPUInit();
        });
    } else {
        ensureGPUInit();
    }
}

void Filter::ensureGPUInit() {
    std::lock_guard<std::mutex> lock(mGpuInitMutex);
    if (mGpuInitDone) return;
    mGpuInitDone = true;

    // Size gate: on tiny inputs the ~3.9 s CUDA init dominates wall time.
    // If both inputs are known files and their combined size is below the
    // threshold, skip GPU init entirely (mNumGPUs stays 0 -> CPU fallback).
    // Unknown sizes (stdin, missing stat) are treated as "large".
    const long long minBytes = fastp_gpu_min_bytes();
    if (minBytes > 0 && mOptions) {
        long long s1 = fastp_input_size_bytes(mOptions->in1);
        long long s2 = fastp_input_size_bytes(mOptions->in2);
        bool s1_known = (s1 >= 0);
        bool s2_known = (s2 >= 0) || mOptions->in2.empty();
        long long known_total = (s1_known ? s1 : 0) + (s2 >= 0 ? s2 : 0);
        if (s1_known && s2_known && known_total < minBytes) {
            loginfo("[GPU] Input " + to_string(known_total) + " bytes < threshold "
                    + to_string(minBytes) + " bytes - using CPU path "
                    "(skipping CUDA init; override with FASTP_GPU_MIN_BYTES=0)");
            return;
        }
    }

    // Lazy CUDA initialization: deferred from constructor to first GPU use,
    // so the ~400ms CUDA context creation overlaps with reader I/O.
    int ndev = min(CudaStatsWrapper::getDeviceCount(), 2);
    for (int i = 0; i < ndev; ++i) {
        auto* w = new CudaStatsWrapper(i);
        if (w->isGPUAvailable()) {
            mGpuStats[mNumGPUs++] = w;
        } else {
            delete w;
        }
    }
    if (mNumGPUs == 2)
        loginfo("[GPU] Dual-GPU mode: Device " + to_string(mGpuStats[0]->getGPUDevice())
                + " + Device " + to_string(mGpuStats[1]->getGPUDevice()) + " active");
    else if (mNumGPUs == 1)
        loginfo("[GPU] Single-GPU mode: Device " + to_string(mGpuStats[0]->getGPUDevice()) + " active");
}


Filter::~Filter(){
    // Make sure the background init thread is done before tearing down
    // any state it might still be touching.
    if (mGpuInitFuture.valid()) {
        mGpuInitFuture.wait();
    }
    for (int i = 0; i < 2; ++i) {
        delete mGpuStats[i];
        mGpuStats[i] = nullptr;
    }
}

void Filter::printProfilingStats() {
    // Always print per-GPU utilization summary (batch/read counts)
    for (int i = 0; i < mNumGPUs; ++i) {
        if (mGpuStats[i])
            mGpuStats[i]->printProfilingStats();
    }

#ifdef FASTP_PROFILING
    // Detailed filter-level timing (only when compiled with -DFASTP_PROFILING)
    auto ms = [](long long ns) { return ns / 1000000.0; };

    long long batches   = mFilterBatches.load();
    long long gpuReads  = mGpuFilterReads.load();
    long long cpuReads  = mCpuFilterReads.load();
    long long gpuNs     = mGpuFilterNs.load();
    long long cpuNs     = mCpuFilterNs.load();

    string msg = "[PROFILE] Filter batches=" + to_string(batches)
               + " gpu_reads=" + to_string(gpuReads)
               + " cpu_reads=" + to_string(cpuReads)
               + " gpu_filter_time=" + to_string(ms(gpuNs)) + "ms"
               + " cpu_filter_time=" + to_string(ms(cpuNs)) + "ms";
    loginfo(msg);
#endif
}

int Filter::passFilter(Read* r) {
    if(r == NULL || r->length()==0) {
        return FAIL_LENGTH;
    }

    int rlen = r->length();
    int lowQualNum = 0;
    int nBaseNum = 0;
    int totalQual = 0;

    // need to recalculate lowQualNum and nBaseNum if the corresponding filters are enabled
    if(mOptions->qualfilter.enabled || mOptions->lengthFilter.enabled) {
        const char* seqstr = r->mSeq->c_str();
        const char* qualstr = r->mQuality->c_str();

        for(int i=0; i<rlen; i++) {
            char base = seqstr[i];
            char qual = qualstr[i];

            totalQual += qual - 33;

            if(qual < mOptions->qualfilter.qualifiedQual)
                lowQualNum ++;

            if(base == 'N')
                nBaseNum++;
        }
    }

    if(mOptions->qualfilter.enabled) {
        if(lowQualNum > (mOptions->qualfilter.unqualifiedPercentLimit * rlen / 100.0) )
            return FAIL_QUALITY;
        else if(mOptions->qualfilter.avgQualReq > 0 && (totalQual / rlen)<mOptions->qualfilter.avgQualReq)
            return FAIL_QUALITY;
        else if(nBaseNum > mOptions->qualfilter.nBaseLimit )
            return FAIL_N_BASE;
    }

    if(mOptions->lengthFilter.enabled) {
        if(rlen < mOptions->lengthFilter.requiredLength)
            return FAIL_LENGTH;
        if(mOptions->lengthFilter.maxLength > 0 && rlen > mOptions->lengthFilter.maxLength)
            return FAIL_TOO_LONG;
    }

    if(mOptions->complexityFilter.enabled) {
        if(!passLowComplexityFilter(r))
            return FAIL_COMPLEXITY;
    }

    return PASS_FILTER;
}

bool Filter::passLowComplexityFilter(Read* r) {
    int diff = 0;
    int length = r->length();
    if(length <= 1)
        return false;
    const char* data = r->mSeq->c_str();
    for(int i=0; i<length-1; i++) {
        if(data[i] != data[i+1])
            diff++;
    }
    if( (double)diff/(double)(length-1) >= mOptions->complexityFilter.threshold )
        return true;
    else
        return false;
}

Read* Filter::trimAndCut(Read* r, int front, int tail, int& frontTrimmed) {
    frontTrimmed = 0;
    // return the same read for speed if no change needed
    if(front == 0 && tail == 0 && !mOptions->qualityCut.enabledFront && !mOptions->qualityCut.enabledTail && !mOptions->qualityCut.enabledRight)
        return r;


    int rlen = r->length() - front - tail ; 
    if (rlen < 0)
        return NULL;

    if(front == 0 && !mOptions->qualityCut.enabledFront && !mOptions->qualityCut.enabledTail && !mOptions->qualityCut.enabledRight){
        r->resize(rlen);
        return r;
    } else if(!mOptions->qualityCut.enabledFront && !mOptions->qualityCut.enabledTail && !mOptions->qualityCut.enabledRight){
        r->mSeq->erase(0,front);
        r->mSeq->resize(rlen);
        r->mQuality->erase(0,front);
        r->mQuality->resize(rlen);
        frontTrimmed  = front;
        return r;
    }

    // need quality cutting

    int l = r->length();
    const char* qualstr = r->mQuality->c_str();
    const char* seq = r->mSeq->c_str();
    // quality cutting forward
    if(mOptions->qualityCut.enabledFront) {
        int w = mOptions->qualityCut.windowSizeFront;
        int s = front;
        if(l - front - tail - w <= 0)
            return NULL;

        int totalQual = 0;

        // preparing rolling
        for(int i=0; i<w-1; i++)
            totalQual += qualstr[s+i];

        for(s=front; s+w<l-tail; s++) {
            totalQual += qualstr[s+w-1];
            // rolling
            if(s > front) {
                totalQual -= qualstr[s-1];
            }
            // add 33 for phred33 transforming
            if((double)totalQual / (double)w >= 33 + mOptions->qualityCut.qualityFront)
                break;
        }

        // the trimming in front is forwarded and rlen is recalculated
        if(s >0 )
            s = s+w-1;
        while(s<l && seq[s] == 'N')
            s++;
        front = s;
        rlen = l - front - tail;
    }

    // quality cutting in right mode
    if(mOptions->qualityCut.enabledRight) {
        int w = mOptions->qualityCut.windowSizeRight;
        int s = front;
        if(l - front - tail - w <= 0)
            return NULL;

        int totalQual = 0;

        // preparing rolling
        for(int i=0; i<w-1; i++)
            totalQual += qualstr[s+i];

        bool foundLowQualWindow = false;

        for(s=front; s+w<l-tail; s++) {
            totalQual += qualstr[s+w-1];
            // rolling
            if(s > front) {
                totalQual -= qualstr[s-1];
            }
            // add 33 for phred33 transforming
            if((double)totalQual / (double)w < 33 + mOptions->qualityCut.qualityRight) {
                foundLowQualWindow = true;
                break;
            }
        }

        if(foundLowQualWindow ) {
            // keep the good bases in the window
            while(s<l-1 && qualstr[s]>=33 + mOptions->qualityCut.qualityRight)
                s++;
            rlen = s - front;
        }
    }

    // quality cutting backward
    if(!mOptions->qualityCut.enabledRight && mOptions->qualityCut.enabledTail) {
        int w = mOptions->qualityCut.windowSizeTail;
        if(l - front - tail - w <= 0)
            return NULL;

        int totalQual = 0;
        int t = l - tail - 1;

        // preparing rolling
        for(int i=0; i<w-1; i++)
            totalQual += qualstr[t-i];

        for(t=l-tail-1; t-w>=front; t--) {
            totalQual += qualstr[t-w+1];
            // rolling
            if(t < l-tail-1) {
                totalQual -= qualstr[t+1];
            }
            // add 33 for phred33 transforming
            if((double)totalQual / (double)w >= 33 + mOptions->qualityCut.qualityTail)
                break;
        }

        if(t < l-1)
            t = t-w+1;
        while(t>=0 && seq[t] == 'N')
            t--;
        rlen = t - front + 1;
    }

    if(rlen <= 0 || front >= l-1)
        return NULL;

    r->mSeq->erase(0, front);
    r->mSeq->resize(rlen);
    r->mQuality->erase(0, front);
    r->mQuality->resize(rlen);

    frontTrimmed = front;

    return r;
}

bool Filter::filterByIndex(Read* r) {
    if(mOptions->indexFilter.enabled) {
        if( match(mOptions->indexFilter.blacklist1, r->firstIndex(), mOptions->indexFilter.threshold) )
            return true;
    }
    return false;
}

bool Filter::filterByIndex(Read* r1, Read* r2) {
    if(mOptions->indexFilter.enabled) {
        if( match(mOptions->indexFilter.blacklist1, r1->firstIndex(), mOptions->indexFilter.threshold) )
            return true;
        if( match(mOptions->indexFilter.blacklist2, r2->lastIndex(), mOptions->indexFilter.threshold) )
            return true;
    }
    return false;
}

bool Filter::match(vector<string>& list, string target, int threshold) {
    for(int i=0; i<list.size(); i++) {
        int diff = 0;
        int len1 = list[i].length();
        int len2 = target.length();
        for(int s=0; s<len1 && s<len2; s++) {
            if(list[i][s] != target[s]) {
                diff++;
                if(diff>threshold)
                    break;
            }
        }
        if(diff <= threshold)
            return true;
    }
    return false;
}

bool Filter::test() {
    Read r("@name",
        "TTTTAACCCCCCCCCCCCCCCCCCCCCCCCCCCCAATTTT",
        "+",
        "/////CCCCCCCCCCCC////CCCCCCCCCCCCCC////E");
    Options opt;
    opt.qualityCut.enabledFront = true;
    opt.qualityCut.enabledTail = true;
    opt.qualityCut.windowSizeFront = 4;
    opt.qualityCut.qualityFront = 20;
    opt.qualityCut.windowSizeTail = 4;
    opt.qualityCut.qualityTail = 20;
    Filter filter(&opt);
    int frontTrimmed = 0;
    Read* ret = filter.trimAndCut(&r, 0, 1, frontTrimmed);
    ret->print();
    
    return *ret->mSeq == "CCCCCCCCCCCCCCCCCCCCCCCCCCCC"
        && *ret->mQuality == "CCCCCCCCCCC////CCCCCCCCCCCCC";
}

// GPU batch filtering for single-end reads
int Filter::filterBatchGPU(const vector<Read*>& reads, vector<int>& filter_results) {
    ensureGPUInit();
#ifdef FASTP_PROFILING
    using clock = std::chrono::high_resolution_clock;
#endif

    filter_results.clear();
    filter_results.resize(reads.size(), FAIL_LENGTH);  // NULL reads default to FAIL_LENGTH

#ifdef FASTP_PROFILING
    mFilterBatches.fetch_add(1, std::memory_order_relaxed);
#endif

    // Build non-NULL subset for GPU processing (trimAndCut can return NULL)
    vector<Read*> validReads;
    vector<size_t> validIndices;
    validReads.reserve(reads.size());
    validIndices.reserve(reads.size());
    for (size_t i = 0; i < reads.size(); i++) {
        if (reads[i] != nullptr) {
            validReads.push_back(reads[i]);
            validIndices.push_back(i);
        }
    }

    if (validReads.empty()) return 0;

    // Try GPU path: pick next available device with round-robin
    if (mNumGPUs > 0) {
#ifdef FASTP_PROFILING
        auto t0 = clock::now();
#endif
        int gpu = mGpuSelector.fetch_add(1, std::memory_order_relaxed) % mNumGPUs;
        vector<ReadStatistics> stats;
        // qual_threshold: convert from ASCII (qualifiedQual) to Phred score
        int phred = (int)(unsigned char)mOptions->qualfilter.qualifiedQual - 33;
        int win   = mOptions->qualityCut.windowSizeShared;
        int r = mGpuStats[gpu]->processBatch(validReads, phred, stats, win);
        if (r == 0 && stats.size() == validReads.size()) {
            for (size_t j = 0; j < validIndices.size(); ++j)
                filter_results[validIndices[j]] = passFilterWithStats(validReads[j], stats[j]);
#ifdef FASTP_PROFILING
            auto t1 = clock::now();
            mGpuFilterNs.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
                std::memory_order_relaxed);
            mGpuFilterReads.fetch_add(validReads.size(), std::memory_order_relaxed);
#endif
            return 0;
        }
        // GPU failed - fall through to CPU
    }

    // CPU fallback
#ifdef FASTP_PROFILING
    auto t0 = clock::now();
#endif
    for (size_t i = 0; i < reads.size(); ++i)
        filter_results[i] = passFilter(reads[i]);
#ifdef FASTP_PROFILING
    auto t1 = clock::now();
    mCpuFilterNs.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
        std::memory_order_relaxed);
    mCpuFilterReads.fetch_add(reads.size(), std::memory_order_relaxed);
#endif
    return 0;
}

// GPU batch filtering for paired-end reads
int Filter::filterBatchGPU(const vector<Read*>& reads1, const vector<Read*>& reads2, 
                          vector<int>& filter_results1, vector<int>& filter_results2) {
    ensureGPUInit();
#ifdef FASTP_PROFILING
    using clock = std::chrono::high_resolution_clock;
#endif

    filter_results1.clear();
    filter_results2.clear();
    filter_results1.resize(reads1.size(), FAIL_LENGTH);  // NULL reads default to FAIL_LENGTH
    filter_results2.resize(reads2.size(), FAIL_LENGTH);

#ifdef FASTP_PROFILING
    mFilterBatches.fetch_add(1, std::memory_order_relaxed);
#endif

    // Build non-NULL subsets for GPU processing (trimAndCut can return NULL)
    vector<Read*> validReads1, validReads2;
    vector<size_t> validIndices1, validIndices2;
    validReads1.reserve(reads1.size());
    validIndices1.reserve(reads1.size());
    for (size_t i = 0; i < reads1.size(); i++) {
        if (reads1[i] != nullptr) {
            validReads1.push_back(reads1[i]);
            validIndices1.push_back(i);
        }
    }
    validReads2.reserve(reads2.size());
    validIndices2.reserve(reads2.size());
    for (size_t i = 0; i < reads2.size(); i++) {
        if (reads2[i] != nullptr) {
            validReads2.push_back(reads2[i]);
            validIndices2.push_back(i);
        }
    }

    if (mNumGPUs > 0) {
#ifdef FASTP_PROFILING
        auto t0 = clock::now();
#endif
        int phred = (int)(unsigned char)mOptions->qualfilter.qualifiedQual - 33;
        int win   = mOptions->qualityCut.windowSizeShared;
        int gpu;

        // Process R1 on GPU (round-robin device selection)
        bool r1gpu = false;
        if (!validReads1.empty()) {
            vector<ReadStatistics> stats1;
            gpu = mGpuSelector.fetch_add(1, std::memory_order_relaxed) % mNumGPUs;
            r1gpu = (mGpuStats[gpu]->processBatch(validReads1, phred, stats1, win) == 0 &&
                          stats1.size() == validReads1.size());
            if (r1gpu) {
                for (size_t j = 0; j < validIndices1.size(); ++j)
                    filter_results1[validIndices1[j]] = passFilterWithStats(validReads1[j], stats1[j]);
#ifdef FASTP_PROFILING
                mGpuFilterReads.fetch_add(validReads1.size(), std::memory_order_relaxed);
#endif
            }
        }
        if (!r1gpu) {
#ifdef FASTP_PROFILING
            auto tc0 = clock::now();
#endif
            for (size_t i = 0; i < reads1.size(); ++i)
                filter_results1[i] = passFilter(reads1[i]);
#ifdef FASTP_PROFILING
            auto tc1 = clock::now();
            mCpuFilterNs.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(tc1 - tc0).count(),
                std::memory_order_relaxed);
            mCpuFilterReads.fetch_add(reads1.size(), std::memory_order_relaxed);
#endif
        }

        // Process R2 on GPU (round-robin — uses other device when dual-GPU)
        bool r2gpu = false;
        if (!validReads2.empty()) {
            vector<ReadStatistics> stats2;
            gpu = mGpuSelector.fetch_add(1, std::memory_order_relaxed) % mNumGPUs;
            r2gpu = (mGpuStats[gpu]->processBatch(validReads2, phred, stats2, win) == 0 &&
                          stats2.size() == validReads2.size());
            if (r2gpu) {
                for (size_t j = 0; j < validIndices2.size(); ++j)
                    filter_results2[validIndices2[j]] = passFilterWithStats(validReads2[j], stats2[j]);
#ifdef FASTP_PROFILING
                mGpuFilterReads.fetch_add(validReads2.size(), std::memory_order_relaxed);
#endif
            }
        }
        if (!r2gpu) {
#ifdef FASTP_PROFILING
            auto tc0 = clock::now();
#endif
            for (size_t i = 0; i < reads2.size(); ++i)
                filter_results2[i] = passFilter(reads2[i]);
#ifdef FASTP_PROFILING
            auto tc1 = clock::now();
            mCpuFilterNs.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(tc1 - tc0).count(),
                std::memory_order_relaxed);
            mCpuFilterReads.fetch_add(reads2.size(), std::memory_order_relaxed);
#endif
        }

#ifdef FASTP_PROFILING
        auto t1 = clock::now();
        if (r1gpu || r2gpu) {
            mGpuFilterNs.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
                std::memory_order_relaxed);
        }
#endif
        return 0;
    }

    // CPU-only fallback (no GPU available)
#ifdef FASTP_PROFILING
    auto t0 = clock::now();
#endif
    for (size_t i = 0; i < reads1.size(); ++i)
        filter_results1[i] = passFilter(reads1[i]);
    for (size_t i = 0; i < reads2.size(); ++i)
        filter_results2[i] = passFilter(reads2[i]);
#ifdef FASTP_PROFILING
    auto t1 = clock::now();
    mCpuFilterNs.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
        std::memory_order_relaxed);
    mCpuFilterReads.fetch_add(reads1.size() + reads2.size(), std::memory_order_relaxed);
#endif
    return 0;
}

// GPU-accelerated head/tail trimming wrapper
Read* Filter::trimAndCutGPU(Read* r, int front, int tail, int& frontTrimmed) {
    // Direct pass-through to CPU implementation (GPU stubs return no-op)
    // Will be optimized when GPU kernels are actively used
    return trimAndCut(r, front, tail, frontTrimmed);
}

// GPU-accelerated poly-G trimming wrapper
int Filter::trimPolyGGPU(Read* r1, Read* r2) {
    // Direct pass-through to CPU PolyX implementation (GPU stubs return no-op)
    // Will be optimized when GPU kernels are actively used
    if(!r1 || !r2) return 0;
    PolyX::trimPolyG(r1, r2, NULL, mOptions->polyGTrim.minLen);
    return 0;
}

// GPU-accelerated quality trimming wrapper
int Filter::trimQualityGPU(Read* r, int window_size, int quality_threshold) {
    // GPU stubs return no-op, so just skip this
    // Quality trimming is an optional optimization
    return 0;
}

/**
 * Apply pass/fail logic using pre-computed ReadStatistics from GPU.
 * Mirrors passFilter() without re-scanning the read bytes.
 * NOTE: complexity filter still requires byte-level access and runs on CPU.
 */
int Filter::passFilterWithStats(Read* r, const ReadStatistics& stats) {
    if (r == nullptr || stats.total_bases == 0)
        return FAIL_LENGTH;

    int rlen = stats.total_bases;

    if (mOptions->qualfilter.enabled) {
        if (stats.low_qual_bases > (mOptions->qualfilter.unqualifiedPercentLimit * rlen / 100.0))
            return FAIL_QUALITY;
        if (mOptions->qualfilter.avgQualReq > 0 &&
            (stats.total_quality / rlen) < mOptions->qualfilter.avgQualReq)
            return FAIL_QUALITY;
        if (stats.n_bases > mOptions->qualfilter.nBaseLimit)
            return FAIL_N_BASE;
    }

    if (mOptions->lengthFilter.enabled) {
        if (rlen < mOptions->lengthFilter.requiredLength)
            return FAIL_LENGTH;
        if (mOptions->lengthFilter.maxLength > 0 && rlen > mOptions->lengthFilter.maxLength)
            return FAIL_TOO_LONG;
    }

    if (mOptions->complexityFilter.enabled) {
        // Complexity filter scans the sequence directly; keep on CPU
        if (!passLowComplexityFilter(r))
            return FAIL_COMPLEXITY;
    }

    return PASS_FILTER;
}

// ===================================================================
// Combined GPU filter + post-filter statRead (single-end)
// ===================================================================
int Filter::filterBatchGPUWithStats(const vector<Read*>& reads,
                                     vector<int>& filter_results,
                                     Stats* postStats) {
    ensureGPUInit();
#ifdef FASTP_PROFILING
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();
    mFilterBatches.fetch_add(1, std::memory_order_relaxed);
#endif
    filter_results.clear();
    filter_results.resize(reads.size(), FAIL_LENGTH);  // NULL reads default to FAIL_LENGTH

    // Build non-NULL subset for GPU processing (trimAndCut can return NULL)
    vector<Read*> validReads;
    vector<size_t> validIndices;
    validReads.reserve(reads.size());
    validIndices.reserve(reads.size());
    for (size_t i = 0; i < reads.size(); i++) {
        if (reads[i] != nullptr) {
            validReads.push_back(reads[i]);
            validIndices.push_back(i);
        }
    }

    if (validReads.empty()) return 0;

    if (mNumGPUs > 0) {
        int gpu = mGpuSelector.fetch_add(1, std::memory_order_relaxed) % mNumGPUs;
        int phred = (int)(unsigned char)mOptions->qualfilter.qualifiedQual - 33;
        int win   = mOptions->qualityCut.windowSizeShared;

        vector<int> gpuFilterResults;
        GpuBatchPostStats batchStats;
        int r = mGpuStats[gpu]->processBatchFilterAndStats(
            validReads, phred, gpuFilterResults, batchStats, win,
            mOptions->qualfilter.unqualifiedPercentLimit,
            mOptions->qualfilter.avgQualReq,
            mOptions->qualfilter.nBaseLimit,
            mOptions->lengthFilter.requiredLength,
            mOptions->lengthFilter.maxLength,
            mOptions->qualfilter.enabled,
            mOptions->lengthFilter.enabled);

        if (r == 0) {
            // Map GPU results back to original indices
            for (size_t j = 0; j < validIndices.size(); j++)
                filter_results[validIndices[j]] = gpuFilterResults[j];

            // Merge GPU-computed post-filter stats into thread-local Stats
            // (must happen before complexity filter so we can unstatRead rejected reads)
            if (postStats)
                postStats->mergeBatchStats(batchStats);

            // Apply complexity filter on CPU for reads that passed GPU filter
            if (mOptions->complexityFilter.enabled) {
                for (size_t j = 0; j < validIndices.size(); j++) {
                    size_t i = validIndices[j];
                    if (filter_results[i] == PASS_FILTER) {
                        if (!passLowComplexityFilter(reads[i])) {
                            filter_results[i] = FAIL_COMPLEXITY;
                            // GPU already counted this read's stats — subtract them
                            if (postStats)
                                postStats->unstatRead(reads[i]);
                        }
                    }
                }
            }
#ifdef FASTP_PROFILING
            auto t1 = clock::now();
            mGpuFilterNs.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
                std::memory_order_relaxed);
            mGpuFilterReads.fetch_add(validReads.size(), std::memory_order_relaxed);
#endif
            return 0;
        }
        // GPU failed - fall through to CPU
    }

    // CPU fallback
    for (size_t i = 0; i < reads.size(); ++i) {
        filter_results[i] = passFilter(reads[i]);
        if (filter_results[i] == PASS_FILTER && reads[i] != nullptr && postStats)
            postStats->statRead(reads[i]);
    }
#ifdef FASTP_PROFILING
    {
        auto t1 = clock::now();
        mCpuFilterNs.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
            std::memory_order_relaxed);
        mCpuFilterReads.fetch_add(reads.size(), std::memory_order_relaxed);
    }
#endif
    return 0;
}

// ===================================================================
// Combined GPU filter + post-filter statRead (paired-end)
// ===================================================================
int Filter::filterBatchGPUWithStats(const vector<Read*>& reads1, const vector<Read*>& reads2,
                                     vector<int>& filter_results1, vector<int>& filter_results2,
                                     Stats* postStats1, Stats* postStats2) {
    ensureGPUInit();
#ifdef FASTP_PROFILING
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();
    mFilterBatches.fetch_add(1, std::memory_order_relaxed);
#endif
    filter_results1.clear();
    filter_results2.clear();
    filter_results1.resize(reads1.size(), FAIL_LENGTH);  // NULL reads default to FAIL_LENGTH
    filter_results2.resize(reads2.size(), FAIL_LENGTH);

    // Build non-NULL subsets for GPU processing (trimAndCut can return NULL)
    vector<Read*> validReads1, validReads2;
    vector<size_t> validIndices1, validIndices2;
    validReads1.reserve(reads1.size());
    validIndices1.reserve(reads1.size());
    for (size_t i = 0; i < reads1.size(); i++) {
        if (reads1[i] != nullptr) {
            validReads1.push_back(reads1[i]);
            validIndices1.push_back(i);
        }
    }
    validReads2.reserve(reads2.size());
    validIndices2.reserve(reads2.size());
    for (size_t i = 0; i < reads2.size(); i++) {
        if (reads2[i] != nullptr) {
            validReads2.push_back(reads2[i]);
            validIndices2.push_back(i);
        }
    }

    if (mNumGPUs > 0) {
        int phred = (int)(unsigned char)mOptions->qualfilter.qualifiedQual - 33;
        int win   = mOptions->qualityCut.windowSizeShared;
        int gpu;

        // R1 on GPU
        bool r1gpu = false;
        if (!validReads1.empty()) {
            gpu = mGpuSelector.fetch_add(1, std::memory_order_relaxed) % mNumGPUs;
            vector<int> gpuResults1;
            GpuBatchPostStats batchStats1;
            int r1res = mGpuStats[gpu]->processBatchFilterAndStats(
                validReads1, phred, gpuResults1, batchStats1, win,
                mOptions->qualfilter.unqualifiedPercentLimit,
                mOptions->qualfilter.avgQualReq,
                mOptions->qualfilter.nBaseLimit,
                mOptions->lengthFilter.requiredLength,
                mOptions->lengthFilter.maxLength,
                mOptions->qualfilter.enabled,
                mOptions->lengthFilter.enabled);

            r1gpu = (r1res == 0);
            if (r1gpu) {
                for (size_t j = 0; j < validIndices1.size(); j++)
                    filter_results1[validIndices1[j]] = gpuResults1[j];
                if (postStats1) postStats1->mergeBatchStats(batchStats1);
                if (mOptions->complexityFilter.enabled) {
                    for (size_t j = 0; j < validIndices1.size(); j++) {
                        size_t i = validIndices1[j];
                        if (filter_results1[i] == PASS_FILTER) {
                            if (!passLowComplexityFilter(reads1[i])) {
                                filter_results1[i] = FAIL_COMPLEXITY;
                                if (postStats1) postStats1->unstatRead(reads1[i]);
                            }
                        }
                    }
                }
            }
        }
        if (!r1gpu) {
            for (size_t i = 0; i < reads1.size(); ++i) {
                filter_results1[i] = passFilter(reads1[i]);
                if (filter_results1[i] == PASS_FILTER && reads1[i] != nullptr && postStats1)
                    postStats1->statRead(reads1[i]);
            }
        }

        // R2 on GPU
        bool r2gpu = false;
        if (!validReads2.empty()) {
            gpu = mGpuSelector.fetch_add(1, std::memory_order_relaxed) % mNumGPUs;
            vector<int> gpuResults2;
            GpuBatchPostStats batchStats2;
            int r2res = mGpuStats[gpu]->processBatchFilterAndStats(
                validReads2, phred, gpuResults2, batchStats2, win,
                mOptions->qualfilter.unqualifiedPercentLimit,
                mOptions->qualfilter.avgQualReq,
                mOptions->qualfilter.nBaseLimit,
                mOptions->lengthFilter.requiredLength,
                mOptions->lengthFilter.maxLength,
                mOptions->qualfilter.enabled,
                mOptions->lengthFilter.enabled);

            r2gpu = (r2res == 0);
            if (r2gpu) {
                for (size_t j = 0; j < validIndices2.size(); j++)
                    filter_results2[validIndices2[j]] = gpuResults2[j];
                if (postStats2) postStats2->mergeBatchStats(batchStats2);
                if (mOptions->complexityFilter.enabled) {
                    for (size_t j = 0; j < validIndices2.size(); j++) {
                        size_t i = validIndices2[j];
                        if (filter_results2[i] == PASS_FILTER) {
                            if (!passLowComplexityFilter(reads2[i])) {
                                filter_results2[i] = FAIL_COMPLEXITY;
                                if (postStats2) postStats2->unstatRead(reads2[i]);
                            }
                        }
                    }
                }
            }
        }
        if (!r2gpu) {
            for (size_t i = 0; i < reads2.size(); ++i) {
                filter_results2[i] = passFilter(reads2[i]);
                if (filter_results2[i] == PASS_FILTER && reads2[i] != nullptr && postStats2)
                    postStats2->statRead(reads2[i]);
            }
        }
#ifdef FASTP_PROFILING
        {
            auto t1 = clock::now();
            long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            if (r1gpu || r2gpu) {
                mGpuFilterNs.fetch_add(ns, std::memory_order_relaxed);
                mGpuFilterReads.fetch_add(
                    (r1gpu ? validReads1.size() : 0) + (r2gpu ? validReads2.size() : 0),
                    std::memory_order_relaxed);
            } else {
                mCpuFilterNs.fetch_add(ns, std::memory_order_relaxed);
                mCpuFilterReads.fetch_add(reads1.size() + reads2.size(), std::memory_order_relaxed);
            }
        }
#endif
        return 0;
    }

    // CPU-only fallback
    for (size_t i = 0; i < reads1.size(); ++i) {
        filter_results1[i] = passFilter(reads1[i]);
        if (filter_results1[i] == PASS_FILTER && reads1[i] != nullptr && postStats1)
            postStats1->statRead(reads1[i]);
    }
    for (size_t i = 0; i < reads2.size(); ++i) {
        filter_results2[i] = passFilter(reads2[i]);
        if (filter_results2[i] == PASS_FILTER && reads2[i] != nullptr && postStats2)
            postStats2->statRead(reads2[i]);
    }
#ifdef FASTP_PROFILING
    {
        auto t1 = clock::now();
        mCpuFilterNs.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
            std::memory_order_relaxed);
        mCpuFilterReads.fetch_add(reads1.size() + reads2.size(), std::memory_order_relaxed);
    }
#endif
    return 0;
}

// ===================================================================
// GPU-accelerated pre-filter statistics
// ===================================================================
int Filter::preStatsBatchGPU(const vector<Read*>& reads, Stats* preStats) {
    ensureGPUInit();
    if (reads.empty() || !preStats) return -1;

    if (mNumGPUs > 0) {
        int gpu = mGpuSelector.fetch_add(1, std::memory_order_relaxed) % mNumGPUs;
        int phred = (int)(unsigned char)mOptions->qualfilter.qualifiedQual - 33;

        GpuBatchPostStats batchStats;
        int r = mGpuStats[gpu]->processBatchStatsOnly(reads, phred, batchStats);
        if (r == 0) {
            preStats->mergeBatchStats(batchStats);
            return 0;
        }
    }

    // CPU fallback: use statReadBasic (matches GPU behavior: no kmer/overrep)
    for (size_t i = 0; i < reads.size(); ++i) {
        if (reads[i]) preStats->statReadBasic(reads[i]);
    }
    return 0;
}
