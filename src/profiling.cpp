/*
 * profiling.cpp — Global profiling singleton and summary printer
 */
#include "profiling.h"
#include <ctime>

#ifdef FASTP_PROFILING

GlobalProfiling g_profiling;

void printProfilingSummary() {
    auto ms = [](long long ns) -> double { return ns / 1000000.0; };

    long long total = g_profiling.total_wall_ns.load();
    long long io    = g_profiling.io_read_ns.load();
    long long dec   = g_profiling.decompress_ns.load();
    long long trim  = g_profiling.cpu_trim_adapter_ns.load();
    long long stat  = g_profiling.cpu_statread_ns.load();
    long long unst  = g_profiling.cpu_unstatread_ns.load();
    long long filt  = g_profiling.cpu_filter_ns.load();
    long long outp  = g_profiling.cpu_output_ns.load();
    long long comp  = g_profiling.compress_ns.load();
    long long writ  = g_profiling.write_ns.load();
    long long reads = g_profiling.total_reads.load();
    long long bytes = g_profiling.io_bytes.load();

    fprintf(stderr, "\n");
    fprintf(stderr, "╔══════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║              PROFILING SUMMARY (FASTP_PROFILING)            ║\n");
    fprintf(stderr, "╠══════════════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║ Total wall time:          %10.1f ms                     ║\n", ms(total));
    fprintf(stderr, "║ Total reads:              %10lld                        ║\n", reads);
    fprintf(stderr, "║ I/O bytes (compressed):   %10lld                        ║\n", bytes);
    fprintf(stderr, "╠══════════════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║ I/O (disk read):          %10.1f ms  (%5.1f%%)           ║\n",
            ms(io), total > 0 ? 100.0 * io / total : 0.0);
    fprintf(stderr, "║ Decompression:            %10.1f ms  (%5.1f%%)           ║\n",
            ms(dec), total > 0 ? 100.0 * dec / total : 0.0);
    fprintf(stderr, "║ CPU trim+adapter:         %10.1f ms  (%5.1f%%)           ║\n",
            ms(trim), total > 0 ? 100.0 * trim / total : 0.0);
    fprintf(stderr, "║ CPU statRead:             %10.1f ms  (%5.1f%%)           ║\n",
            ms(stat), total > 0 ? 100.0 * stat / total : 0.0);
    fprintf(stderr, "║ CPU unstatRead:           %10.1f ms  (%5.1f%%)           ║\n",
            ms(unst), total > 0 ? 100.0 * unst / total : 0.0);
    fprintf(stderr, "║ filter calls (incl. GPU): %10.1f ms  (%5.1f%%)           ║\n",
            ms(filt), total > 0 ? 100.0 * filt / total : 0.0);    // sum of filterBatchGPUWithStats call durations across all threads
    fprintf(stderr, "║ CPU output format:        %10.1f ms  (%5.1f%%)           ║\n",
            ms(outp), total > 0 ? 100.0 * outp / total : 0.0);
    fprintf(stderr, "║ Output compression:       %10.1f ms  (%5.1f%%)           ║\n",
            ms(comp), total > 0 ? 100.0 * comp / total : 0.0);
    fprintf(stderr, "║ Output write (fwrite):    %10.1f ms  (%5.1f%%)           ║\n",
            ms(writ), total > 0 ? 100.0 * writ / total : 0.0);
    fprintf(stderr, "╠══════════════════════════════════════════════════════════════╣\n");

    /* GPU breakdown is printed separately by CudaStatsWrapper */

    fprintf(stderr, "║ (GPU breakdown printed by CudaStatsWrapper above)          ║\n");
    fprintf(stderr, "╚══════════════════════════════════════════════════════════════╝\n");

    /* CSV-friendly line for automated parsing */
    fprintf(stderr, "[PROF_CSV] wall_ms,io_ms,decomp_ms,cpu_trim_ms,cpu_stat_ms,"
            "cpu_unstat_ms,filter_calls_ms,cpu_output_ms,compress_ms,write_ms,reads,io_bytes\n");
    fprintf(stderr, "[PROF_CSV] %.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%lld,%lld\n",
            ms(total), ms(io), ms(dec), ms(trim), ms(stat),
            ms(unst), ms(filt), ms(outp), ms(comp), ms(writ), reads, bytes);
}

#endif /* FASTP_PROFILING */
