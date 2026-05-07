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

// how many reads one pack has
// Optimized for I/O efficiency: larger packs reduce synchronization overhead
// and allow better buffering in decompression pipeline
// Larger pack size gives the GPU a much larger batch per kernel launch.
// 8192 reads × ~150bp avg = ~1.2 MB seq data per pack; at BLOCK_SIZE=256 and
// 1 warp (32 threads) per read this produces 1024 blocks per kernel launch,
// covering all 108 A100 SMs many times and driving SM utilisation to ~50–80%.
static const int PACK_SIZE = 8192;

// if one pack is produced, but not consumed, it will be kept in the memory
// this number limit the number of in memory packs
// if the number of in memory packs is full, the producer thread should sleep
// Reduced from 256 because each pack is now 16× larger (8192 vs 512 reads).
// 64 × 8192 = 524 288 reads max in-flight, comparable to old 256 × 512.
static const int PACK_IN_MEM_LIMIT = 64;


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
