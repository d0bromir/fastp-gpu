/*
 * contaminant_db.cpp — GPU-fastp k-mer contamination detection
 *
 * NTHash formulation:
 *   For each k-mer we compute the canonical (min of forward, reverse-complement)
 *   NTHash and use the upper 32 bits as the hash table key.  The lower 32 bits
 *   are discarded; with M ≈ 600 k-mers in the table the collision probability is
 *   negligible (< 2^{-10} per insertion).
 *
 * NTHash seeds (Mohamadi et al. 2016, bioinformatics.oxfordjournals.org):
 *   SEED_A = 0x3c8bfbb395c60474ULL
 *   SEED_C = 0x3193c18562a02b4cULL
 *   SEED_G = 0x20323ed082572324ULL
 *   SEED_T = 0x295549f54be24456ULL
 * Reverse-complement seeds: swap A↔T, C↔G.
 */

#include "contaminant_db.h"
#include "options.h"
#include "read.h"
#include "sequence.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

// -------------------------------------------------------------------------
// NTHash seeds
// -------------------------------------------------------------------------

static const uint64_t SEED_A = 0x3c8bfbb395c60474ULL;
static const uint64_t SEED_C = 0x3193c18562a02b4cULL;
static const uint64_t SEED_G = 0x20323ed082572324ULL;
static const uint64_t SEED_T = 0x295549f54be24456ULL;

static const uint64_t SEED_RC_A = SEED_T;  // complement of A is T
static const uint64_t SEED_RC_C = SEED_G;  // complement of C is G
static const uint64_t SEED_RC_G = SEED_C;
static const uint64_t SEED_RC_T = SEED_A;

// -------------------------------------------------------------------------
// Built-in adapter sequences
// Each source group is a comma-separated collection of sequence strings.
// All sequences in a group share one source ID.
// -------------------------------------------------------------------------

// Illumina TruSeq: R1 + R2 adapters.
static const char* BUILTIN_TRUSEQ[] = {
    "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA",   // TruSeq R1
    "AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT",   // TruSeq R2
    NULL
};

// Nextera transposase adapters.
static const char* BUILTIN_NEXTERA[] = {
    "CTGTCTCTTATACACATCT",   // Nextera R1 (short core)
    "CTGTCTCTTATACACATCTCCGAGCCCACGAGAC",  // Nextera R1 full
    "CTGTCTCTTATACACATCTGACGCTGCCGACGA",   // Nextera R2 full
    NULL
};

// -------------------------------------------------------------------------
// Bit-rotate helpers
// -------------------------------------------------------------------------

static inline uint64_t rol64(uint64_t v, int n) {
    return (v << n) | (v >> (64 - n));
}

static inline uint64_t ror64(uint64_t v, int n) {
    return (v >> n) | (v << (64 - n));
}

// -------------------------------------------------------------------------
// NTHash for a single base
// -------------------------------------------------------------------------

static inline uint64_t seed_fwd(char c) {
    switch (c | 0x20) {  // tolower
        case 'a': return SEED_A;
        case 'c': return SEED_C;
        case 'g': return SEED_G;
        case 't': return SEED_T;
        default:  return 0ULL;
    }
}

static inline uint64_t seed_rc(char c) {
    switch (c | 0x20) {
        case 'a': return SEED_RC_A;
        case 'c': return SEED_RC_C;
        case 'g': return SEED_RC_G;
        case 't': return SEED_RC_T;
        default:  return 0ULL;
    }
}

// Compute canonical NTHash for sequence[0..k-1].
// Returns the minimum of forward and reverse-complement hashes.
// Returns 0 if the k-mer contains an N (non-ACGT base).
static uint64_t canonical_nthash(const char* seq, int k) {
    uint64_t fwd = 0ULL;
    uint64_t rev = 0ULL;
    for (int i = 0; i < k; i++) {
        uint64_t sf = seed_fwd(seq[i]);
        if (sf == 0ULL) return 0ULL;  // N or unknown
        fwd ^= rol64(sf, k - 1 - i);
        rev ^= rol64(seed_rc(seq[i]), i);
    }
    return (fwd < rev) ? fwd : rev;
}

// -------------------------------------------------------------------------
// ContaminantDB implementation
// -------------------------------------------------------------------------

ContaminantDB::ContaminantDB()
    : mTable(nullptr), mTableSize(0), mTableMask(0),
      mMinKmerHits(3), mNumKmers(0) {
    mSourceNames.push_back("none");  // index 0
}

ContaminantDB::~ContaminantDB() {
    free(mTable);
    mTable = nullptr;
}

// -------------------------------------------------------------------------
// Helper: resolve the directory containing the running binary.
// -------------------------------------------------------------------------
static std::string getBinaryDir() {
    char buf[4096];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0) return "";
    buf[len] = '\0';
    char* slash = strrchr(buf, '/');
    if (slash) *slash = '\0';
    return std::string(buf);
}

// Count sequence bases in a FASTA/multi-source file (for table sizing).
// Ignores '>' header lines and '#' comment lines.
static long countFastaBases(const std::string& path) {
    std::ifstream f(path);
    if (!f.good()) return 0;
    std::string line;
    long total = 0;
    while (std::getline(f, line)) {
        if (!line.empty() && line[0] != '>' && line[0] != '#')
            total += (long)line.size();
    }
    return total;
}

void ContaminantDB::build(Options* opt) {
    if (!opt->contaminant.enabled) return;

    mMinKmerHits = opt->contaminant.minKmerHits;

    // ------------------------------------------------------------------
    // Locate the shipped contaminants.fa (next to the binary, or ./data/).
    // ------------------------------------------------------------------
    std::string shippedDb;
    {
        std::string binDir = getBinaryDir();
        if (!binDir.empty()) {
            std::string candidate = binDir + "/data/contaminants.fa";
            if (access(candidate.c_str(), R_OK) == 0) shippedDb = candidate;
        }
        if (shippedDb.empty() && access("data/contaminants.fa", R_OK) == 0)
            shippedDb = "data/contaminants.fa";
    }

    // ------------------------------------------------------------------
    // Size estimation (two-pass: count bases, then insert).
    // ------------------------------------------------------------------
    int estimatedKmers = 0;

    if (!shippedDb.empty()) {
        long bases = countFastaBases(shippedDb);
        estimatedKmers += (int)(bases > CONTAM_K ? bases - CONTAM_K + 1 : 0);
    } else {
        // Fall back to hardcoded built-ins for sizing.
        for (int i = 0; BUILTIN_TRUSEQ[i]; i++)
            estimatedKmers += (int)strlen(BUILTIN_TRUSEQ[i]) - CONTAM_K + 1;
        for (int i = 0; BUILTIN_NEXTERA[i]; i++)
            estimatedKmers += (int)strlen(BUILTIN_NEXTERA[i]) - CONTAM_K + 1;
    }

    if (!opt->contaminant.dbFile.empty()) {
        long bases = countFastaBases(opt->contaminant.dbFile);
        if (bases == 0) {
            fprintf(stderr, "[fastp] WARNING: contaminant DB file not found or "
                    "empty: %s\n", opt->contaminant.dbFile.c_str());
        }
        estimatedKmers += (int)(bases > CONTAM_K ? bases - CONTAM_K + 1 : 0);
    }

    // Allocate: next power-of-two > 2 × estimatedKmers (load factor ≤ 0.5).
    uint32_t cap = 1024;
    while (cap < (uint32_t)(estimatedKmers * 2 + 64)) cap <<= 1;
    allocTable(cap);

    // ------------------------------------------------------------------
    // Insert sequences.
    // ------------------------------------------------------------------
    if (!shippedDb.empty()) {
        int n = addMultiSourceFasta(shippedDb);
        if (n == 0) {
            // File was found but parsed no sources — warn and fall back.
            fprintf(stderr, "[fastp] WARNING: contaminants.fa loaded no sources; "
                    "falling back to built-in sequences.\n");
            shippedDb.clear();
        }
    }

    if (shippedDb.empty()) {
        // Hardcoded fall-back: register TruSeq (1) and Nextera (2) manually.
        int srcTruSeq  = registerSource("TruSeq");
        int srcNextera = registerSource("Nextera");
        for (int i = 0; BUILTIN_TRUSEQ[i]; i++)
            addSequence(BUILTIN_TRUSEQ[i], srcTruSeq);
        for (int i = 0; BUILTIN_NEXTERA[i]; i++)
            addSequence(BUILTIN_NEXTERA[i], srcNextera);
    }

    // User-supplied extra FASTA (--contaminant_db).
    if (!opt->contaminant.dbFile.empty()) {
        // Assign each FASTA entry in the user file its own source slot.
        int firstUserSrc = (int)mSourceNames.size();
        addFastaFile(opt->contaminant.dbFile, "user", firstUserSrc);
    }

    // Publish source name list back to Options so reporters can emit names
    // in the JSON/HTML output without holding a ContaminantDB pointer.
    opt->contaminant.sourceNames = mSourceNames;
}

int ContaminantDB::registerSource(const std::string& name) {
    if ((int)mSourceNames.size() >= MAX_CONTAM_SOURCES) {
        fprintf(stderr, "[fastp] WARNING: exceeded MAX_CONTAM_SOURCES (%d); "
                "ignoring source '%s'\n", MAX_CONTAM_SOURCES, name.c_str());
        return -1;
    }
    mSourceNames.push_back(name);
    return (int)mSourceNames.size() - 1;
}

void ContaminantDB::allocTable(uint32_t capacity) {
    free(mTable);
    mTableSize = capacity;
    mTableMask = capacity - 1;
    mTable = (KhtEntry*)calloc(capacity, sizeof(KhtEntry));
    if (!mTable) {
        fprintf(stderr, "[fastp] FATAL: could not allocate contamination hash table "
                "(%u entries)\n", capacity);
        exit(1);
    }
}

bool ContaminantDB::tableInsert(uint64_t hashVal, uint8_t srcId) {
    if (hashVal == 0ULL) return false;  // skip N-containing k-mers
    uint32_t key = (uint32_t)(hashVal >> 32);
    if (key == 0) key = 1;  // 0 is the empty sentinel; shift by 1

    uint32_t slot = key & mTableMask;
    for (uint32_t i = 0; i < mTableSize; i++) {
        uint32_t s = (slot + i) & mTableMask;
        if (mTable[s].key == 0) {
            mTable[s].key = key;
            mTable[s].src = srcId;
            mNumKmers++;
            return true;
        }
        if (mTable[s].key == key) {
            // Already present (possibly from a different source); keep first.
            return true;
        }
    }
    // Table full — should not happen if sizing logic is correct.
    fprintf(stderr, "[fastp] WARNING: contamination hash table is full; "
            "some k-mers were not inserted.\n");
    return false;
}

int ContaminantDB::tableLookup(uint64_t hashVal) const {
    if (hashVal == 0ULL || mTableSize == 0) return 0;
    uint32_t key = (uint32_t)(hashVal >> 32);
    if (key == 0) key = 1;

    uint32_t slot = key & mTableMask;
    for (uint32_t i = 0; i < mTableSize; i++) {
        uint32_t s = (slot + i) & mTableMask;
        if (mTable[s].key == 0) return 0;
        if (mTable[s].key == key) return (int)mTable[s].src;
    }
    return 0;
}

void ContaminantDB::addSequence(const std::string& seq, int srcId) {
    if (srcId <= 0 || srcId >= MAX_CONTAM_SOURCES) return;
    int len = (int)seq.size();
    for (int i = 0; i + CONTAM_K <= len; i++) {
        uint64_t h = canonical_nthash(seq.c_str() + i, CONTAM_K);
        tableInsert(h, (uint8_t)srcId);
    }
}

void ContaminantDB::addFastaFile(const std::string& path,
                                  const std::string& srcPrefix,
                                  int firstSrcId) {
    std::ifstream f(path);
    if (!f.good()) return;

    std::string line, seqBuf;
    int curSrc = -1;
    int userSrcCount = 0;

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line[0] == '>') {
            // Flush previous sequence.
            if (curSrc > 0 && !seqBuf.empty())
                addSequence(seqBuf, curSrc);
            seqBuf.clear();
            // Register a new source name from the FASTA header.
            std::string hdr = line.substr(1);
            if (!hdr.empty() && hdr.back() == '\r') hdr.pop_back();
            std::string srcName = srcPrefix + ":" + hdr.substr(0, 60);
            int srcId = firstSrcId + userSrcCount;
            if (srcId < MAX_CONTAM_SOURCES) {
                // Ensure the mSourceNames vector is large enough.
                while ((int)mSourceNames.size() <= srcId)
                    mSourceNames.push_back("");
                mSourceNames[srcId] = srcName;
                curSrc = srcId;
                userSrcCount++;
            } else {
                curSrc = -1;
            }
        } else {
            if (curSrc > 0) seqBuf += line;
        }
    }
    // Flush last entry.
    if (curSrc > 0 && !seqBuf.empty())
        addSequence(seqBuf, curSrc);
}

void ContaminantDB::addBuiltinSources() {
    // Called from build(); kept as a no-op here since build() handles all.
}

// -------------------------------------------------------------------------
// Multi-source FASTA loader.
// Supports '# source: NAME' comment lines to start a new source group.
// All FASTA entries following such a line belong to that source.
// Returns the number of new sources registered.
// -------------------------------------------------------------------------
int ContaminantDB::addMultiSourceFasta(const std::string& path) {
    std::ifstream f(path);
    if (!f.good()) return 0;

    std::string line, seqBuf;
    int curSrc = -1;
    int newSources = 0;

    auto flush = [&]() {
        if (curSrc > 0 && !seqBuf.empty()) {
            addSequence(seqBuf, curSrc);
            seqBuf.clear();
        }
    };

    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        if (line[0] == '#') {
            // Check for '# source: NAME'
            const char* p = line.c_str() + 1;
            while (*p == ' ') p++;
            if (strncmp(p, "source:", 7) == 0) {
                flush();
                p += 7;
                while (*p == ' ') p++;
                std::string srcName(p);
                // Trim trailing whitespace.
                while (!srcName.empty() && (srcName.back() == ' ' ||
                                            srcName.back() == '\t'))
                    srcName.pop_back();
                if (!srcName.empty()) {
                    curSrc = registerSource(srcName);
                    if (curSrc > 0) newSources++;
                }
            }
            // Other comment lines are ignored.
            continue;
        }

        if (line[0] == '>') {
            // New FASTA entry: flush previous sequence (same source).
            flush();
            // No source change — sequence belongs to curSrc until next
            // '# source:' line.
        } else {
            if (curSrc > 0) seqBuf += line;
        }
    }
    flush();
    return newSources;
}

// -------------------------------------------------------------------------
// CPU scan
// -------------------------------------------------------------------------

int ContaminantDB::scanRead(const Read* read) const {
    if (mTableSize == 0 || mNumKmers == 0) return 0;

    int hits[MAX_CONTAM_SOURCES] = {};
    const std::string& seq = *read->mSeq;
    int len = (int)seq.size();

    for (int i = 0; i + CONTAM_K <= len; i++) {
        uint64_t h = canonical_nthash(seq.c_str() + i, CONTAM_K);
        int srcId = tableLookup(h);
        if (srcId > 0 && srcId < MAX_CONTAM_SOURCES)
            hits[srcId]++;
    }

    // Return source ID with most hits, if it meets the threshold.
    int bestSrc = 0, bestHits = 0;
    for (int s = 1; s < MAX_CONTAM_SOURCES; s++) {
        if (hits[s] > bestHits) {
            bestHits = hits[s];
            bestSrc  = s;
        }
    }
    return (bestHits >= mMinKmerHits) ? bestSrc : 0;
}

// -------------------------------------------------------------------------
// Source name lookup
// -------------------------------------------------------------------------

const std::string& ContaminantDB::sourceName(int srcId) const {
    if (srcId < 0 || srcId >= (int)mSourceNames.size()) {
        static const std::string unknown = "unknown";
        return unknown;
    }
    return mSourceNames[srcId];
}

// -------------------------------------------------------------------------
// Built-in unit test
// -------------------------------------------------------------------------

bool ContaminantDB::test() {
    // Test 1: k-mer from TruSeq adapter should be detected.
    // The built-in TruSeq R1 adapter is "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA" (33bp).
    // We construct a synthetic read: 60bp of random + adapter + 60bp random.
    // At least 3 canonical 31-mers from the adapter must hash to TruSeq (srcId=1).

    Options opt;
    opt.contaminant.enabled     = true;
    opt.contaminant.minKmerHits = 3;
    opt.contaminant.dbFile      = "";

    ContaminantDB db;
    db.build(&opt);

    // Verify that the table was populated.
    if (db.mNumKmers == 0) return false;

    // Construct a read with the TruSeq R1 adapter embedded.
    // Flanking regions use an ATCG/TACG repeat that produces no poly-run
    // or adapter k-mers, so the only hits come from the TruSeq adapter.
    // Use the const char* constructor so Read owns its string copies.
    const std::string randLeft  = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"; // 60 bp
    const std::string randRight = "TACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACG"; // 60 bp
    const std::string adapterSeq = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA";
    const std::string fullSeq = randLeft + adapterSeq + randRight;

    // Build a synthetic Read using const char* constructor (Read makes its own copies).
    const std::string qualStr(fullSeq.size(), 'I');
    Read r("@test_truseq", fullSeq.c_str(), "+", qualStr.c_str(), false);

    int srcId = db.scanRead(&r);
    if (srcId != 1) return false;  // must detect TruSeq (source 1)

    // Test 2: clean read (ACGT repeat, no contaminant k-mers) must NOT be flagged.
    // All-A or all-T reads now match PolyRuns; use a non-homopolymeric sequence.
    const std::string cleanSeq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                                 "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                                 "ACGTACGTACGTACGTACGTACGTACGTACGT"; // 136 bp, ACGT×34
    const std::string cqual(cleanSeq.size(), 'I');
    Read rClean("@test_clean", cleanSeq.c_str(), "+", cqual.c_str(), false);
    if (db.scanRead(&rClean) != 0) return false;  // must be clean

    // Test 3: verify source names.
    if (db.sourceName(0) != "none")   return false;
    if (db.sourceName(1) != "TruSeq") return false;
    if (db.sourceName(2) != "Nextera") return false;

    return true;
}
