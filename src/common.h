#ifndef COMMON_H
#define COMMON_H

#define FASTP_VER "1.3.3-d0bromir"

#define _DEBUG false

// GPU debug output control (set to 1 to enable debug messages)
#define GPU_DEBUG 0

#if GPU_DEBUG
#define GPU_FPRINTF(...) fprintf(stderr, __VA_ARGS__)
#else
#define GPU_FPRINTF(...) do {} while(0)
#endif

#ifndef _WIN32
	typedef long int64;
	typedef unsigned long uint64;
#else
	typedef long long int64;
	typedef unsigned long long uint64;
#endif

typedef int int32;
typedef unsigned int uint32;

typedef short int16;
typedef unsigned short uint16;

typedef char int8;
typedef unsigned char uint8;

const char ATCG_BASES[] = {'A', 'T', 'C', 'G'};

// Maximum reads per pack (upper bound).  The GPU kernel launch is sized for
// this many reads per batch: 8192 × ~150bp avg = ~1.2 MB seq data per pack;
// at BLOCK_SIZE=256 / 1 warp per read this produces 1024 blocks per launch,
// covering all 108 A100 SMs at 50–80% utilisation.
// The actual pack size used at runtime is adaptive (see effectivePackSize in
// seprocessor.cpp / peprocessor.cpp) and will be ≤ MAX_PACK_SIZE.
static const int MAX_PACK_SIZE = 8192;

// if one pack is produced, but not consumed, it will be kept in the memory
// this number limit the number of in memory packs
// if the number of in memory packs is full, the producer thread should sleep
// Limit is kept at MAX_PACK_SIZE × 64 = 524 288 reads max in-flight regardless
// of the adaptive pack size chosen at runtime.
static const int PACK_IN_MEM_LIMIT = 64;

// Packs are distributed round-robin across effectiveThreads per-worker
// queues, so a *global* in-flight budget of PACK_IN_MEM_LIMIT gives each
// worker only PACK_IN_MEM_LIMIT/effectiveThreads packs of headroom on
// average. At effectiveThreads >= PACK_IN_MEM_LIMIT (e.g. -w 64, the
// tool's own thread ceiling) that headroom collapses to ~1 pack per
// worker, forcing the reader and every worker into near-lockstep
// handoff and eliminating the overlap this pipeline depends on -- this
// was the root cause of a ~16x wall-time regression measured at -w 64
// on a 40GB paired-end dataset (see
// docs/publication/supplementary/thread-ceiling-collapse-and-fix.md).
// PACK_IN_MEM_HEADROOM
// guarantees a minimum per-worker headroom regardless of thread count,
// while max() leaves the original constant (and thus memory footprint)
// unchanged at the low-to-moderate thread counts it was tuned for.
static const int PACK_IN_MEM_HEADROOM = 4;
static inline long packInMemLimit(int effectiveThreads) {
    long perWorker = (long)effectiveThreads * PACK_IN_MEM_HEADROOM;
    return perWorker > PACK_IN_MEM_LIMIT ? perWorker : PACK_IN_MEM_LIMIT;
}


// different filtering results, bigger number means worse
// if r1 and r2 are both failed, then the bigger one of the two results will be recorded
// we reserve some gaps for future types to be added
static const int PASS_FILTER = 0;
static const int FAIL_POLY_X = 4;
static const int FAIL_OVERLAP = 8;
static const int FAIL_N_BASE = 12;
static const int FAIL_LENGTH = 16;
static const int FAIL_TOO_LONG = 17;
static const int FAIL_QUALITY = 20;
static const int FAIL_COMPLEXITY = 24;
static const int FAIL_ADAPTER_DIMER = 28;

// how many types in total we support
static const int FILTER_RESULT_TYPES = 32;

const static char* FAILED_TYPES[FILTER_RESULT_TYPES] = {
	"passed", "", "", "",
	"failed_polyx_filter", "", "", "",
	"failed_bad_overlap", "", "", "",
	"failed_too_many_n_bases", "", "", "",
	"failed_too_short", "failed_too_long", "", "",
	"failed_quality_filter", "", "", "",
	"failed_low_complexity", "", "", "",
	"failed_adapter_dimer", "", "", ""
};

#endif /* COMMON_H */
