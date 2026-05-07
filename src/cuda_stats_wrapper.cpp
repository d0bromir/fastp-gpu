#include "cuda_stats_wrapper.h"
#include "common.h"
#include <cstring>
#include <iostream>
#include <chrono>
#include "util.h"

#ifdef HAVE_CUDA
#include "cuda_stats.h"
#include <cuda_runtime.h>
#endif

// -----------------------------------------------------------------------
// Slot-pool sizing.  Each slot holds buffers for up to SLOT_MAX_READS reads.
// With PACK_SIZE=8192 we need at least 8192; use 2× for headroom.
// Each slot's seq/qual packed buffer = SLOT_BUF_BYTES bytes.
// Total GPU memory per device = NUM_SLOTS × 2 × (SLOT_BUF_BYTES + pointer/int
// arrays) ≈ 4 × 2 × 10 MB = 80 MB – well within A100-80 GB.
// -----------------------------------------------------------------------
static const int    SLOT_MAX_READS = 16384;
static const size_t SLOT_BUF_BYTES = static_cast<size_t>(SLOT_MAX_READS) * 1024; // 16 MB/slot – handles up to 16384 reads × ~1KB avg

CudaStatsWrapper::CudaStatsWrapper(int device)
    : gpu_available(false), device_id(-1), next_slot_hint(0)
{
#ifdef HAVE_CUDA
    cudaSetDevice(device);

    gpu_available = cuda_is_available();
    device_id     = gpu_available ? device : -1;

    if (gpu_available) {
        string msg = string("[GPU] CUDA GPU available (Device ") +
                     to_string(device_id) + ") - statistics GPU-accelerated";
        loginfo(msg);

        for (int i = 0; i < NUM_SLOTS; i++) {
            allocateSlot(slots[i], SLOT_MAX_READS, SLOT_BUF_BYTES);
        }
    } else {
        loginfo("[GPU] CUDA GPU not available - statistics will use CPU");
    }
#else
    gpu_available = false;
    device_id     = -1;
    loginfo("[GPU] CUDA support not compiled - statistics will use CPU");
#endif
}

CudaStatsWrapper::~CudaStatsWrapper()
{
#ifdef HAVE_CUDA
    if (gpu_available) {
        cudaSetDevice(device_id);
        for (int i = 0; i < NUM_SLOTS; i++) freeSlot(slots[i]);
    }
#endif
}

// ------------------------------------------------------------
// Static helper

int CudaStatsWrapper::getDeviceCount()
{
#ifdef HAVE_CUDA
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    return count;
#else
    return 0;
#endif
}

bool CudaStatsWrapper::isGPUAvailable() { return gpu_available; }
int  CudaStatsWrapper::getGPUDevice()   { return device_id; }

void CudaStatsWrapper::printProfilingStats() const {
    long long batches = profiling.batch_count.load();
    long long reads   = profiling.read_count.load();
    if (batches == 0) return;

    // Always print lightweight utilization summary (batch/read counts)
    string msg = "[GPU] Device " + to_string(device_id)
               + ": batches=" + to_string(batches)
               + " reads=" + to_string(reads);

#ifdef FASTP_PROFILING
    // Detailed timing breakdown (only when compiled with -DFASTP_PROFILING)
    auto ms = [](long long ns) { return ns / 1000000.0; };
    msg += " total_gpu=" + to_string(ms(profiling.total_gpu_ns.load())) + "ms"
         + " pack=" + to_string(ms(profiling.pack_ns.load())) + "ms"
         + " h2d=" + to_string(ms(profiling.h2d_ns.load())) + "ms"
         + " kernel=" + to_string(ms(profiling.kernel_ns.load())) + "ms"
         + " d2h=" + to_string(ms(profiling.d2h_ns.load())) + "ms";
#endif

    loginfo(msg);
}

// ===================================================================
#ifdef HAVE_CUDA
// ===================================================================

void CudaStatsWrapper::allocateSlot(GpuSlot& slot, int max_reads, size_t buf_bytes)
{
    slot.max_reads = max_reads;
    slot.buf_bytes = buf_bytes;

    // Device buffers
    cudaMalloc((void**)&slot.d_seq_buf,   buf_bytes);
    cudaMalloc((void**)&slot.d_qual_buf,  buf_bytes);
    cudaMalloc((void**)&slot.d_seq_ptrs,  max_reads * sizeof(char*));
    cudaMalloc((void**)&slot.d_qual_ptrs, max_reads * sizeof(char*));
    cudaMalloc((void**)&slot.d_read_lens, max_reads * sizeof(int));
    cudaMalloc((void**)&slot.d_stats,     max_reads * sizeof(struct ReadStatistics));

    // Combined filter+stats device buffers
    cudaMalloc((void**)&slot.d_filter_results,   max_reads * sizeof(int));
    cudaMalloc((void**)&slot.d_batch_post_stats,  sizeof(struct GpuBatchPostStats));

    // Pinned host mirrors
    cudaHostAlloc((void**)&slot.h_seq_buf,   buf_bytes,              cudaHostAllocDefault);
    cudaHostAlloc((void**)&slot.h_qual_buf,  buf_bytes,              cudaHostAllocDefault);
    cudaHostAlloc((void**)&slot.h_seq_ptrs,  max_reads*sizeof(char*),cudaHostAllocDefault);
    cudaHostAlloc((void**)&slot.h_qual_ptrs, max_reads*sizeof(char*),cudaHostAllocDefault);
    cudaHostAlloc((void**)&slot.h_read_lens, max_reads*sizeof(int),  cudaHostAllocDefault);
    cudaHostAlloc((void**)&slot.h_stats,     max_reads*sizeof(struct ReadStatistics),
                  cudaHostAllocDefault);

    // Combined filter+stats pinned host mirrors
    cudaHostAlloc((void**)&slot.h_filter_results, max_reads*sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&slot.h_batch_post_stats, sizeof(struct GpuBatchPostStats), cudaHostAllocDefault);

    cudaStreamCreate(&slot.stream);
}

void CudaStatsWrapper::freeSlot(GpuSlot& slot)
{
    if (slot.d_seq_buf)   { cudaFree(slot.d_seq_buf);   slot.d_seq_buf   = nullptr; }
    if (slot.d_qual_buf)  { cudaFree(slot.d_qual_buf);  slot.d_qual_buf  = nullptr; }
    if (slot.d_seq_ptrs)  { cudaFree(slot.d_seq_ptrs);  slot.d_seq_ptrs  = nullptr; }
    if (slot.d_qual_ptrs) { cudaFree(slot.d_qual_ptrs); slot.d_qual_ptrs = nullptr; }
    if (slot.d_read_lens) { cudaFree(slot.d_read_lens); slot.d_read_lens = nullptr; }
    if (slot.d_stats)     { cudaFree(slot.d_stats);     slot.d_stats     = nullptr; }
    if (slot.d_filter_results)   { cudaFree(slot.d_filter_results);   slot.d_filter_results   = nullptr; }
    if (slot.d_batch_post_stats) { cudaFree(slot.d_batch_post_stats); slot.d_batch_post_stats = nullptr; }

    if (slot.h_seq_buf)   { cudaFreeHost(slot.h_seq_buf);   slot.h_seq_buf   = nullptr; }
    if (slot.h_qual_buf)  { cudaFreeHost(slot.h_qual_buf);  slot.h_qual_buf  = nullptr; }
    if (slot.h_seq_ptrs)  { cudaFreeHost(slot.h_seq_ptrs);  slot.h_seq_ptrs  = nullptr; }
    if (slot.h_qual_ptrs) { cudaFreeHost(slot.h_qual_ptrs); slot.h_qual_ptrs = nullptr; }
    if (slot.h_read_lens) { cudaFreeHost(slot.h_read_lens); slot.h_read_lens = nullptr; }
    if (slot.h_stats)     { cudaFreeHost(slot.h_stats);     slot.h_stats     = nullptr; }
    if (slot.h_filter_results)   { cudaFreeHost(slot.h_filter_results);   slot.h_filter_results   = nullptr; }
    if (slot.h_batch_post_stats) { cudaFreeHost(slot.h_batch_post_stats); slot.h_batch_post_stats = nullptr; }

    if (slot.stream) { cudaStreamDestroy(slot.stream); slot.stream = 0; }
}

// -------------------------------------------------------------------
// Core: process one batch on a pre-acquired slot.
// The slot mutex MUST be held by the caller when this is invoked.
// -------------------------------------------------------------------
int CudaStatsWrapper::processSlot(GpuSlot& slot,
                                   const vector<Read*>& reads,
                                   int qual_threshold,
                                   vector<ReadStatistics>& stats,
                                   int trim_window_size)
{
#ifdef FASTP_PROFILING
    using clock = std::chrono::high_resolution_clock;
    auto t_slot_start = clock::now();
    long long pack_acc = 0, h2d_acc = 0, kern_acc = 0, d2h_acc = 0;
#endif

    cudaSetDevice(device_id);   // re-bind device for this calling thread

    int total_reads = (int)reads.size();
    stats.clear();
    stats.reserve(total_reads);

    // Process in chunks no larger than the slot's capacity.
    int processed = 0;
    while (processed < total_reads) {
        int chunk = std::min(total_reads - processed, slot.max_reads);

        // ---- pack seq/qual into contiguous pinned host buffers ----
#ifdef FASTP_PROFILING
        auto t0 = clock::now();
#endif
        size_t seq_off = 0, qual_off = 0;
        for (int i = 0; i < chunk; i++) {
            const string& s = *reads[processed + i]->mSeq;
            const string& q = *reads[processed + i]->mQuality;
            int len = (int)s.size();
            slot.h_read_lens[i] = len;

            // Fallback: if this single read exceeds the buffer, truncate silently.
            if (seq_off + len > slot.buf_bytes) {
                chunk = i;   // process what we have so far, then loop again
                break;
            }
            memcpy(slot.h_seq_buf  + seq_off,  s.c_str(), len);
            memcpy(slot.h_qual_buf + qual_off, q.c_str(), len);
            slot.h_seq_ptrs[i]  = slot.d_seq_buf  + seq_off;
            slot.h_qual_ptrs[i] = slot.d_qual_buf + qual_off;
            seq_off  += len;
            qual_off += len;
        }
        if (chunk == 0) { processed++; continue; }  // single oversized read
#ifdef FASTP_PROFILING
        auto t1 = clock::now();
        pack_acc += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
#endif

        // ---- async H→D transfers ----
#ifdef FASTP_PROFILING
        auto t_h2d_start = clock::now();
#endif
        cudaMemcpyAsync(slot.d_seq_buf,   slot.h_seq_buf,   seq_off,
                        cudaMemcpyHostToDevice, slot.stream);
        cudaMemcpyAsync(slot.d_qual_buf,  slot.h_qual_buf,  qual_off,
                        cudaMemcpyHostToDevice, slot.stream);
        cudaMemcpyAsync(slot.d_seq_ptrs,  slot.h_seq_ptrs,  chunk*sizeof(char*),
                        cudaMemcpyHostToDevice, slot.stream);
        cudaMemcpyAsync(slot.d_qual_ptrs, slot.h_qual_ptrs, chunk*sizeof(char*),
                        cudaMemcpyHostToDevice, slot.stream);
        cudaMemcpyAsync(slot.d_read_lens, slot.h_read_lens, chunk*sizeof(int),
                        cudaMemcpyHostToDevice, slot.stream);
#ifdef FASTP_PROFILING
        cudaStreamSynchronize(slot.stream);
        auto t_h2d_end = clock::now();
        h2d_acc += std::chrono::duration_cast<std::chrono::nanoseconds>(t_h2d_end - t_h2d_start).count();
#endif

        // ---- launch warp-per-read kernel ----
#ifdef FASTP_PROFILING
        auto t_kern_start = clock::now();
#endif
        int r = cuda_compute_read_stats_device(
                    slot.d_seq_ptrs, slot.d_qual_ptrs, slot.d_read_lens,
                    chunk, (char)(qual_threshold + 33),
                    slot.d_stats, slot.stream, trim_window_size);
        if (r != 0) return r;
#ifdef FASTP_PROFILING
        cudaStreamSynchronize(slot.stream);
        auto t_kern_end = clock::now();
        kern_acc += std::chrono::duration_cast<std::chrono::nanoseconds>(t_kern_end - t_kern_start).count();
#endif

        // ---- async D→H results ----
#ifdef FASTP_PROFILING
        auto t_d2h_start = clock::now();
#endif
        cudaMemcpyAsync(slot.h_stats, slot.d_stats,
                        chunk * sizeof(struct ReadStatistics),
                        cudaMemcpyDeviceToHost, slot.stream);

        // ---- wait for this slot's stream only ----
        cudaStreamSynchronize(slot.stream);
#ifdef FASTP_PROFILING
        auto t_d2h_end = clock::now();
        d2h_acc += std::chrono::duration_cast<std::chrono::nanoseconds>(t_d2h_end - t_d2h_start).count();
#endif

        for (int i = 0; i < chunk; i++) stats.push_back(slot.h_stats[i]);
        processed += chunk;
    }

    // Always track batch/read counts (negligible overhead — two atomic adds)
    profiling.batch_count.fetch_add(1, std::memory_order_relaxed);
    profiling.read_count.fetch_add(total_reads, std::memory_order_relaxed);

#ifdef FASTP_PROFILING
    auto t_slot_end = clock::now();
    profiling.total_gpu_ns.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_slot_end - t_slot_start).count(),
        std::memory_order_relaxed);
    profiling.pack_ns.fetch_add(pack_acc, std::memory_order_relaxed);
    profiling.h2d_ns.fetch_add(h2d_acc, std::memory_order_relaxed);
    profiling.kernel_ns.fetch_add(kern_acc, std::memory_order_relaxed);
    profiling.d2h_ns.fetch_add(d2h_acc, std::memory_order_relaxed);
#endif

    return 0;
}

// -------------------------------------------------------------------
// Public entry point.
//
// Acquires any free slot via try_lock (round-robin hint); if all NUM_SLOTS
// slots are busy it blocks on slot 0.  This allows up to NUM_SLOTS concurrent
// GPU kernels from different worker threads on the SAME device, eliminating
// the serialisation bottleneck of the old single-mutex design.
// -------------------------------------------------------------------
int CudaStatsWrapper::processBatch(
    const vector<Read*>& reads,
    int qual_threshold,
    vector<ReadStatistics>& stats,
    int trim_window_size)
{
    if (reads.empty()) { stats.clear(); return 0; }

    // Round-robin starting slot for this invocation
    int hint = next_slot_hint.fetch_add(1, std::memory_order_relaxed) % NUM_SLOTS;

    // Try non-blocking acquisition on all slots starting from hint
    for (int i = 0; i < NUM_SLOTS; i++) {
        int idx = (hint + i) % NUM_SLOTS;
        if (slots[idx].slot_mutex.try_lock()) {
            int r = processSlot(slots[idx], reads, qual_threshold, stats, trim_window_size);
            slots[idx].slot_mutex.unlock();
            return r;
        }
    }
    // All slots busy – block on the hinted slot
    slots[hint].slot_mutex.lock();
    int r = processSlot(slots[hint], reads, qual_threshold, stats, trim_window_size);
    slots[hint].slot_mutex.unlock();
    return r;
}

// -------------------------------------------------------------------
// Combined filter + post-filter stats: slot-level implementation.
// -------------------------------------------------------------------
int CudaStatsWrapper::processSlotFilterAndStats(
    GpuSlot& slot,
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
    bool length_filter_enabled)
{
    cudaSetDevice(device_id);

    int total_reads = (int)reads.size();
    filter_results.clear();
    filter_results.resize(total_reads, 0);
    memset(&batch_post_stats, 0, sizeof(batch_post_stats));

    int processed = 0;
    // Zero-init device batch post stats once (will accumulate across chunks)
    cudaMemsetAsync(slot.d_batch_post_stats, 0, sizeof(struct GpuBatchPostStats), slot.stream);

    while (processed < total_reads) {
        int chunk = std::min(total_reads - processed, slot.max_reads);

        // ---- pack seq/qual into contiguous pinned host buffers ----
        size_t seq_off = 0, qual_off = 0;
        for (int i = 0; i < chunk; i++) {
            const string& s = *reads[processed + i]->mSeq;
            const string& q = *reads[processed + i]->mQuality;
            int len = (int)s.size();
            slot.h_read_lens[i] = len;

            if (seq_off + len > slot.buf_bytes) {
                chunk = i;
                break;
            }
            memcpy(slot.h_seq_buf  + seq_off,  s.c_str(), len);
            memcpy(slot.h_qual_buf + qual_off, q.c_str(), len);
            slot.h_seq_ptrs[i]  = slot.d_seq_buf  + seq_off;
            slot.h_qual_ptrs[i] = slot.d_qual_buf + qual_off;
            seq_off  += len;
            qual_off += len;
        }
        if (chunk == 0) { processed++; continue; }

        // ---- async H→D transfers ----
        cudaMemcpyAsync(slot.d_seq_buf,   slot.h_seq_buf,   seq_off,
                        cudaMemcpyHostToDevice, slot.stream);
        cudaMemcpyAsync(slot.d_qual_buf,  slot.h_qual_buf,  qual_off,
                        cudaMemcpyHostToDevice, slot.stream);
        cudaMemcpyAsync(slot.d_seq_ptrs,  slot.h_seq_ptrs,  chunk*sizeof(char*),
                        cudaMemcpyHostToDevice, slot.stream);
        cudaMemcpyAsync(slot.d_qual_ptrs, slot.h_qual_ptrs, chunk*sizeof(char*),
                        cudaMemcpyHostToDevice, slot.stream);
        cudaMemcpyAsync(slot.d_read_lens, slot.h_read_lens, chunk*sizeof(int),
                        cudaMemcpyHostToDevice, slot.stream);

        // ---- launch combined filter+stats kernel ----
        int r = cuda_filter_and_stats_device(
                    slot.d_seq_ptrs, slot.d_qual_ptrs, slot.d_read_lens,
                    chunk, (char)(qual_threshold + 33),
                    slot.d_filter_results, slot.d_batch_post_stats,
                    slot.stream, trim_window_size,
                    unqual_percent_limit, avg_qual_req, n_base_limit,
                    length_required, max_length,
                    qual_filter_enabled, length_filter_enabled);
        if (r != 0) return r;

        // ---- async D→H: per-read filter results ----
        cudaMemcpyAsync(slot.h_filter_results, slot.d_filter_results,
                        chunk * sizeof(int),
                        cudaMemcpyDeviceToHost, slot.stream);

        cudaStreamSynchronize(slot.stream);

        // Copy filter results into output vector
        for (int i = 0; i < chunk; i++)
            filter_results[processed + i] = slot.h_filter_results[i];

        processed += chunk;
    }

    // Transfer accumulated batch post stats D→H
    cudaMemcpyAsync(slot.h_batch_post_stats, slot.d_batch_post_stats,
                    sizeof(struct GpuBatchPostStats),
                    cudaMemcpyDeviceToHost, slot.stream);
    cudaStreamSynchronize(slot.stream);
    memcpy(&batch_post_stats, slot.h_batch_post_stats, sizeof(struct GpuBatchPostStats));

    profiling.batch_count.fetch_add(1, std::memory_order_relaxed);
    profiling.read_count.fetch_add(total_reads, std::memory_order_relaxed);

    return 0;
}

// -------------------------------------------------------------------
// Public entry point for stats-only (pre-filter stats).
// -------------------------------------------------------------------
int CudaStatsWrapper::processBatchStatsOnly(
    const vector<Read*>& reads,
    int qual_threshold,
    struct GpuBatchPostStats& batch_post_stats)
{
    if (reads.empty()) return 0;

    int hint = next_slot_hint.fetch_add(1, std::memory_order_relaxed) % NUM_SLOTS;

    for (int i = 0; i < NUM_SLOTS; i++) {
        int idx = (hint + i) % NUM_SLOTS;
        if (slots[idx].slot_mutex.try_lock()) {
            cudaSetDevice(device_id);
            int total_reads = (int)reads.size();
            memset(&batch_post_stats, 0, sizeof(batch_post_stats));
            int processed = 0;
            GpuSlot& slot = slots[idx];
            cudaMemsetAsync(slot.d_batch_post_stats, 0, sizeof(struct GpuBatchPostStats), slot.stream);

            while (processed < total_reads) {
                int chunk = std::min(total_reads - processed, slot.max_reads);
                size_t seq_off = 0, qual_off = 0;
                for (int j = 0; j < chunk; j++) {
                    const string& s = *reads[processed + j]->mSeq;
                    const string& q = *reads[processed + j]->mQuality;
                    int len = (int)s.size();
                    slot.h_read_lens[j] = len;
                    if (seq_off + len > slot.buf_bytes) { chunk = j; break; }
                    memcpy(slot.h_seq_buf  + seq_off,  s.c_str(), len);
                    memcpy(slot.h_qual_buf + qual_off, q.c_str(), len);
                    slot.h_seq_ptrs[j]  = slot.d_seq_buf  + seq_off;
                    slot.h_qual_ptrs[j] = slot.d_qual_buf + qual_off;
                    seq_off  += len;
                    qual_off += len;
                }
                if (chunk == 0) { processed++; continue; }

                cudaMemcpyAsync(slot.d_seq_buf,   slot.h_seq_buf,   seq_off,  cudaMemcpyHostToDevice, slot.stream);
                cudaMemcpyAsync(slot.d_qual_buf,  slot.h_qual_buf,  qual_off, cudaMemcpyHostToDevice, slot.stream);
                cudaMemcpyAsync(slot.d_seq_ptrs,  slot.h_seq_ptrs,  chunk*sizeof(char*), cudaMemcpyHostToDevice, slot.stream);
                cudaMemcpyAsync(slot.d_qual_ptrs, slot.h_qual_ptrs, chunk*sizeof(char*), cudaMemcpyHostToDevice, slot.stream);
                cudaMemcpyAsync(slot.d_read_lens, slot.h_read_lens, chunk*sizeof(int),   cudaMemcpyHostToDevice, slot.stream);

                int r = cuda_filter_and_stats_device(
                            slot.d_seq_ptrs, slot.d_qual_ptrs, slot.d_read_lens,
                            chunk, (char)(qual_threshold + 33),
                            slot.d_filter_results, slot.d_batch_post_stats,
                            slot.stream, 0, 0, 0, 0, 0, 0, false, false,
                            /*stats_only=*/true);
                if (r != 0) { slots[idx].slot_mutex.unlock(); return r; }

                cudaStreamSynchronize(slot.stream);
                processed += chunk;
            }

            cudaMemcpyAsync(slot.h_batch_post_stats, slot.d_batch_post_stats,
                            sizeof(struct GpuBatchPostStats), cudaMemcpyDeviceToHost, slot.stream);
            cudaStreamSynchronize(slot.stream);
            memcpy(&batch_post_stats, slot.h_batch_post_stats, sizeof(struct GpuBatchPostStats));
            slots[idx].slot_mutex.unlock();
            return 0;
        }
    }

    // All slots busy — wait on hint slot
    slots[hint].slot_mutex.lock();
    cudaSetDevice(device_id);
    int total_reads = (int)reads.size();
    memset(&batch_post_stats, 0, sizeof(batch_post_stats));
    int processed = 0;
    GpuSlot& slot = slots[hint];
    cudaMemsetAsync(slot.d_batch_post_stats, 0, sizeof(struct GpuBatchPostStats), slot.stream);

    while (processed < total_reads) {
        int chunk = std::min(total_reads - processed, slot.max_reads);
        size_t seq_off = 0, qual_off = 0;
        for (int j = 0; j < chunk; j++) {
            const string& s = *reads[processed + j]->mSeq;
            const string& q = *reads[processed + j]->mQuality;
            int len = (int)s.size();
            slot.h_read_lens[j] = len;
            if (seq_off + len > slot.buf_bytes) { chunk = j; break; }
            memcpy(slot.h_seq_buf  + seq_off,  s.c_str(), len);
            memcpy(slot.h_qual_buf + qual_off, q.c_str(), len);
            slot.h_seq_ptrs[j]  = slot.d_seq_buf  + seq_off;
            slot.h_qual_ptrs[j] = slot.d_qual_buf + qual_off;
            seq_off  += len;
            qual_off += len;
        }
        if (chunk == 0) { processed++; continue; }

        cudaMemcpyAsync(slot.d_seq_buf,   slot.h_seq_buf,   seq_off,  cudaMemcpyHostToDevice, slot.stream);
        cudaMemcpyAsync(slot.d_qual_buf,  slot.h_qual_buf,  qual_off, cudaMemcpyHostToDevice, slot.stream);
        cudaMemcpyAsync(slot.d_seq_ptrs,  slot.h_seq_ptrs,  chunk*sizeof(char*), cudaMemcpyHostToDevice, slot.stream);
        cudaMemcpyAsync(slot.d_qual_ptrs, slot.h_qual_ptrs, chunk*sizeof(char*), cudaMemcpyHostToDevice, slot.stream);
        cudaMemcpyAsync(slot.d_read_lens, slot.h_read_lens, chunk*sizeof(int),   cudaMemcpyHostToDevice, slot.stream);

        int r = cuda_filter_and_stats_device(
                    slot.d_seq_ptrs, slot.d_qual_ptrs, slot.d_read_lens,
                    chunk, (char)(qual_threshold + 33),
                    slot.d_filter_results, slot.d_batch_post_stats,
                    slot.stream, 0, 0, 0, 0, 0, 0, false, false,
                    /*stats_only=*/true);
        if (r != 0) { slots[hint].slot_mutex.unlock(); return r; }

        cudaStreamSynchronize(slot.stream);
        processed += chunk;
    }

    cudaMemcpyAsync(slot.h_batch_post_stats, slot.d_batch_post_stats,
                    sizeof(struct GpuBatchPostStats), cudaMemcpyDeviceToHost, slot.stream);
    cudaStreamSynchronize(slot.stream);
    memcpy(&batch_post_stats, slot.h_batch_post_stats, sizeof(struct GpuBatchPostStats));
    slots[hint].slot_mutex.unlock();
    return 0;
}

// -------------------------------------------------------------------
// Public entry point for combined filter + stats.
// -------------------------------------------------------------------
int CudaStatsWrapper::processBatchFilterAndStats(
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
    bool length_filter_enabled)
{
    if (reads.empty()) { filter_results.clear(); return 0; }

    int hint = next_slot_hint.fetch_add(1, std::memory_order_relaxed) % NUM_SLOTS;

    for (int i = 0; i < NUM_SLOTS; i++) {
        int idx = (hint + i) % NUM_SLOTS;
        if (slots[idx].slot_mutex.try_lock()) {
            int r = processSlotFilterAndStats(slots[idx], reads, qual_threshold,
                        filter_results, batch_post_stats, trim_window_size,
                        unqual_percent_limit, avg_qual_req, n_base_limit,
                        length_required, max_length,
                        qual_filter_enabled, length_filter_enabled);
            slots[idx].slot_mutex.unlock();
            return r;
        }
    }
    slots[hint].slot_mutex.lock();
    int r = processSlotFilterAndStats(slots[hint], reads, qual_threshold,
                filter_results, batch_post_stats, trim_window_size,
                unqual_percent_limit, avg_qual_req, n_base_limit,
                length_required, max_length,
                qual_filter_enabled, length_filter_enabled);
    slots[hint].slot_mutex.unlock();
    return r;
}

// ===================================================================
#else  // !HAVE_CUDA – CPU stub
// ===================================================================

int CudaStatsWrapper::processBatch(
    const vector<Read*>& reads,
    int qual_threshold,
    vector<ReadStatistics>& stats)
{
    return -1;   // GPU not compiled in; caller falls back to CPU path
}

#endif // HAVE_CUDA

