/*
 * profiling.h — Centralized profiling counters for paper benchmarking
 *
 * Compile with -DFASTP_PROFILING to enable.
 * All counters are atomic (thread-safe, lock-free).
 * At program exit, call printProfilingSummary() to emit a structured report.
 */
#pragma once

#include <atomic>
#include <cstdio>
#include <ctime>

#ifdef FASTP_PROFILING

struct GlobalProfiling {
    /* ── I/O: disk read + decompression ─────────────────────────────── */
    std::atomic<long long> io_read_ns{0};       // wall time in FastqReader reading from disk
    std::atomic<long long> decompress_ns{0};    // nvCOMP / igzip decompression time
    std::atomic<long long> io_bytes{0};         // total bytes read from disk (compressed)

    /* ── CPU processing (in worker threads) ─────────────────────────── */
    std::atomic<long long> cpu_trim_adapter_ns{0}; // trim + adapter removal (1st pass)
    std::atomic<long long> cpu_statread_ns{0};     // statRead (pre + speculative post)
    std::atomic<long long> cpu_unstatread_ns{0};   // unstatRead for failed reads
    std::atomic<long long> cpu_filter_ns{0};       // CPU-side passFilter calls
    std::atomic<long long> cpu_output_ns{0};       // appendToString / output formatting

    /* ── Output compression + write ─────────────────────────────────── */
    std::atomic<long long> compress_ns{0};  // bgzf_compress time (compress pool workers)
    std::atomic<long long> write_ns{0};     // fwrite time (writer worker)

    /* ── Total wall time ────────────────────────────────────────────── */
    std::atomic<long long> total_wall_ns{0};
    std::atomic<long long> total_reads{0};
};

/* Singleton instance */
extern GlobalProfiling g_profiling;

/* Helper: get current time in ns */
inline long long profiling_now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* Print structured summary to stderr */
void printProfilingSummary();

#define PROF_START(var)  long long var = profiling_now_ns()
#define PROF_END(var, counter) do { \
    long long _end = profiling_now_ns(); \
    g_profiling.counter.fetch_add(_end - (var), std::memory_order_relaxed); \
} while(0)

#define PROF_ADD(counter, value) \
    g_profiling.counter.fetch_add((long long)(value), std::memory_order_relaxed)

#else /* !FASTP_PROFILING */

#define PROF_START(var)        ((void)0)
#define PROF_END(var, counter) ((void)0)
#define PROF_ADD(counter, value) ((void)0)

#endif /* FASTP_PROFILING */
