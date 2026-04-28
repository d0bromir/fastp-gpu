#include "cuda_stats.h"
#include "cuda_stats_wrapper.h"
#include "util.h"
#include <cstring>

int cuda_is_available() {
    return 0;
}

int cuda_get_device() {
    return -1;
}

int cuda_compute_read_stats(
    const char** sequences,
    const char** qualities,
    const int* read_lengths,
    int num_reads,
    char qual_threshold,
    struct ReadStatistics* stats
){
    if (num_reads <= 0 || !stats)
        return -1;

    for (int i = 0; i < num_reads; ++i) {
        const char* seq = sequences ? sequences[i] : NULL;
        const char* qual = qualities ? qualities[i] : NULL;
        int read_len = read_lengths ? read_lengths[i] : 0;

        int total_bases = read_len;
        int n_bases = 0;
        int low_qual_bases = 0;
        int total_quality = 0;

        if (seq && qual) {
            for (int j = 0; j < read_len; ++j) {
                char base = seq[j];
                char q = qual[j];
                if (base == 'N' || base == 'n')
                    ++n_bases;
                if (q < qual_threshold)
                    ++low_qual_bases;
                total_quality += (q - 33);
            }
        }

        stats[i].total_bases = total_bases;
        stats[i].n_bases = n_bases;
        stats[i].low_qual_bases = low_qual_bases;
        stats[i].total_quality = total_quality;
    }

    return 0;
}

// CPU-only stubs for CudaStatsWrapper
CudaStatsWrapper::CudaStatsWrapper(int device)
    : gpu_available(false), device_id(-1) {
    (void)device;
    loginfo("[GPU] CUDA support not compiled - statistics will use CPU");
}

CudaStatsWrapper::~CudaStatsWrapper() {
}

int CudaStatsWrapper::getDeviceCount() { return 0; }
bool CudaStatsWrapper::isGPUAvailable() { return false; }
int CudaStatsWrapper::getGPUDevice() { return -1; }
void CudaStatsWrapper::printProfilingStats() const { /* no-op in CPU build */ }

int CudaStatsWrapper::processBatch(
    const vector<Read*>& reads,
    int qual_threshold,
    vector<ReadStatistics>& stats,
    int trim_window_size
) {
    // CPU fallback: compute stats without CUDA
    stats.clear();
    stats.reserve(reads.size());
    char q_thresh = (char)(qual_threshold + 33);
    for (auto* r : reads) {
        ReadStatistics s{};
        if (r) {
            const char* seq  = r->mSeq->c_str();
            const char* qual = r->mQuality->c_str();
            int len = r->length();
            s.total_bases = len;
            for (int i = 0; i < len; ++i) {
                if (seq[i] == 'N' || seq[i] == 'n') ++s.n_bases;
                if (qual[i] < q_thresh)             ++s.low_qual_bases;
                s.total_quality += (qual[i] - 33);
            }
        }
        stats.push_back(s);
    }
    return 0;
}

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
    return -1;   // GPU not compiled in; caller falls back to CPU path
}

int CudaStatsWrapper::processBatchStatsOnly(
    const vector<Read*>& reads,
    int qual_threshold,
    struct GpuBatchPostStats& batch_post_stats)
{
    return -1;   // GPU not compiled in; caller falls back to CPU path
}
