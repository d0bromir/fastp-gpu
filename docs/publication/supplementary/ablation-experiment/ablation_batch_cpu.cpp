// ablation_batch_cpu.cpp -- standalone 3-way ablation tool for the
// Application Notes review (BIOADV, Reviewer 3, Comment 4):
//
//   "an ablation comparing stock fastp, the batched pipeline with a CPU
//    kernel, and the batched pipeline with the CUDA kernel would help
//    validate the reported gain"
//
// This isolates the batching/packing overhead from the actual GPU kernel
// by running the *identical* pack step used by the real GPU path
// (src/cuda_stats_wrapper.cpp:processSlotFilterAndStats, the
// "pack seq/qual into contiguous host buffers" loop, copied verbatim
// below for fidelity) followed by a CPU computation of the same
// filter+stats math the GPU kernel performs (src/cuda_stats.cu,
// filter_and_stats_warp_kernel: N-base/low-qual/quality-sum scan,
// threshold filter, per-cycle histogram + 4-mer for passing reads),
// instead of a CUDA dispatch.
//
// Deliberately standalone rather than wired into the shipped pipeline:
// this is an experimental measurement tool, not a new build mode of
// fastp-gpu itself, so it carries zero risk to the production binary.
//
// Build: see experiments/build_ablation.sh
// Usage: ./ablation_batch_cpu <in1.fastq.gz> [in2.fastq.gz] <threads> <batch_size>

#include "../src/fastqreader.h"
#include "../src/read.h"
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#include <atomic>

using namespace std;
using Clock = chrono::steady_clock;

// Mirrors GpuBatchPostStats layout closely enough for a fair CPU compute
// cost comparison (same fields computed, same access pattern granularity).
struct BatchPostStats {
    long long cycle_total_base[512] = {0};
    long long cycle_total_qual[512] = {0};
    long long cycle_q20[512] = {0};
    long long cycle_q30[512] = {0};
    long long kmer[1024] = {0};
    long long reads_passed = 0;
};

struct FilterParams {
    int qual_threshold = 15;      // Q15, matches fastp default low-qual cutoff (33+15)
    int unqual_percent_limit = 40;
    int n_base_limit = 5;
    int length_required = 15;
};

// Pack step: byte-for-byte the same operation as
// cuda_stats_wrapper.cpp's processSlotFilterAndStats pack loop --
// contiguous memcpy of seq/qual into a flat host buffer, building a
// per-read pointer + length table. No GPU-specific pinned-memory flag
// is used (regular malloc) since there is no real device involved here;
// the memcpy cost this measures is identical either way.
struct PackedBatch {
    vector<char> seq_buf, qual_buf;
    vector<const char*> seq_ptrs, qual_ptrs;
    vector<int> read_lens;
    int n = 0;
};

static void pack_batch(const vector<Read*>& reads, int begin, int end, PackedBatch& pb) {
    size_t total = 0;
    for (int i = begin; i < end; i++) total += reads[i]->mSeq->size();
    pb.seq_buf.resize(total);
    pb.qual_buf.resize(total);
    pb.seq_ptrs.resize(end - begin);
    pb.qual_ptrs.resize(end - begin);
    pb.read_lens.resize(end - begin);
    pb.n = end - begin;

    size_t off = 0;
    for (int i = begin; i < end; i++) {
        const string& s = *reads[i]->mSeq;
        const string& q = *reads[i]->mQuality;
        int len = (int)s.size();
        int j = i - begin;
        pb.read_lens[j] = len;
        memcpy(pb.seq_buf.data() + off, s.c_str(), len);
        memcpy(pb.qual_buf.data() + off, q.c_str(), len);
        pb.seq_ptrs[j] = pb.seq_buf.data() + off;
        pb.qual_ptrs[j] = pb.qual_buf.data() + off;
        off += len;
    }
}

// CPU compute step: same math as filter_and_stats_warp_kernel's Phase
// 1 (N/low-qual/quality scan), Phase 2 (threshold filter), Phase 3
// (per-cycle histogram + 4-mer for passing reads) -- sequential per
// read instead of one warp per read, since there is no GPU here.
static void compute_batch_cpu(const PackedBatch& pb, const FilterParams& fp,
                               BatchPostStats& stats, vector<int>& filter_results) {
    filter_results.resize(pb.n);
    for (int r = 0; r < pb.n; r++) {
        const char* s = pb.seq_ptrs[r];
        const char* q = pb.qual_ptrs[r];
        int L = pb.read_lens[r];

        int n_bases = 0, low_qual = 0, total_qual = 0;
        for (int i = 0; i < L; i++) {
            if (s[i] == 'N' || s[i] == 'n') n_bases++;
            char phred = q[i] - 33;
            if (phred < fp.qual_threshold) low_qual++;
            total_qual += phred;
        }

        bool pass = true;
        if (low_qual > (fp.unqual_percent_limit * L / 100)) pass = false;
        if (n_bases > fp.n_base_limit) pass = false;
        if (L < fp.length_required) pass = false;
        filter_results[r] = pass ? 1 : 0;
        if (!pass) continue;

        stats.reads_passed++;
        int kmer = 0;
        for (int i = 0; i < L && i < 512; i++) {
            char phred = q[i] - 33;
            stats.cycle_total_base[i]++;
            stats.cycle_total_qual[i] += phred;
            if (phred >= 30) { stats.cycle_q30[i]++; stats.cycle_q20[i]++; }
            else if (phred >= 20) { stats.cycle_q20[i]++; }

            int val = -1;
            switch (s[i]) {
                case 'A': case 'a': val = 0; break;
                case 'T': case 't': val = 1; break;
                case 'C': case 'c': val = 2; break;
                case 'G': case 'g': val = 3; break;
            }
            if (val < 0) { kmer = 0; continue; }
            kmer = ((kmer << 2) & 0x3FC) | val;
            if (i >= 4) stats.kmer[kmer]++;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <in1.fastq.gz> <threads> <batch_size> [in2.fastq.gz]\n", argv[0]);
        return 1;
    }
    string in1 = argv[1];
    int threads = atoi(argv[2]);
    int batch_size = atoi(argv[3]);
    string in2 = (argc > 4) ? argv[4] : "";

    FastqReader r1(in1);
    FastqReader* r2 = in2.empty() ? nullptr : new FastqReader(in2);

    vector<Read*> reads;
    reads.reserve(batch_size * 4);
    Read* r;
    long long total_reads_seen = 0;
    long long total_pack_ns = 0, total_compute_ns = 0;
    BatchPostStats global_stats;
    long long global_passed = 0;

    auto t_wall_start = Clock::now();

    // Single-threaded batch loop (thread count is accepted for CLI
    // symmetry with fastp -w but this tool measures per-batch
    // pack+compute cost; multi-threading the batch loop would need a
    // work queue identical to fastp's own worker pool, out of scope for
    // an isolated micro-benchmark of the pack+compute step itself).
    (void)threads;

    while ((r = r1.read()) != nullptr) {
        reads.push_back(r);
        if (r2) {
            Read* r2read = r2->read();
            if (r2read) { reads.push_back(r2read); }
        }
        total_reads_seen++;

        if ((int)reads.size() >= batch_size) {
            PackedBatch pb;
            vector<int> filter_results;
            auto t0 = Clock::now();
            pack_batch(reads, 0, (int)reads.size(), pb);
            auto t1 = Clock::now();
            compute_batch_cpu(pb, FilterParams{}, global_stats, filter_results);
            auto t2 = Clock::now();
            total_pack_ns += chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count();
            total_compute_ns += chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();
            for (int fr : filter_results) if (fr) global_passed++;

            for (auto* rd : reads) delete rd;
            reads.clear();
        }
    }
    if (!reads.empty()) {
        PackedBatch pb;
        vector<int> filter_results;
        auto t0 = Clock::now();
        pack_batch(reads, 0, (int)reads.size(), pb);
        auto t1 = Clock::now();
        compute_batch_cpu(pb, FilterParams{}, global_stats, filter_results);
        auto t2 = Clock::now();
        total_pack_ns += chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count();
        total_compute_ns += chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();
        for (int fr : filter_results) if (fr) global_passed++;
        for (auto* rd : reads) delete rd;
    }

    auto t_wall_end = Clock::now();
    double wall_s = chrono::duration_cast<chrono::duration<double>>(t_wall_end - t_wall_start).count();

    printf("{\n");
    printf("  \"reads_in\": %lld,\n", total_reads_seen);
    printf("  \"reads_passed\": %lld,\n", global_passed);
    printf("  \"wall_s\": %.4f,\n", wall_s);
    printf("  \"pack_s\": %.4f,\n", total_pack_ns / 1e9);
    printf("  \"compute_s\": %.4f,\n", total_compute_ns / 1e9);
    printf("  \"batch_size\": %d\n", batch_size);
    printf("}\n");

    if (r2) delete r2;
    return 0;
}
