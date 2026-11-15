#ifndef CONTAMINANT_DB_H
#define CONTAMINANT_DB_H

/*
 * contaminant_db.h — GPU-fastp k-mer contamination detection
 *
 * Builds a flat, open-addressed hash table of canonical 31-mer NTHashes
 * from built-in adapter sequences and an optional user-supplied FASTA file.
 * ContaminantDB::scanRead() performs a CPU-side scan per read; the GPU Phase 4
 * kernel (cuda_contaminant.cu) provides GPU-accelerated scanning on the same
 * table layout.
 *
 * Source IDs are 1-indexed; 0 is the "empty" sentinel.  Source names are
 * registered in the order sequences are added.
 *
 * Built-in sources (always present when contamination detection is enabled):
 *   1 — TruSeq (Illumina TruSeq universal + Read 2 adapter)
 *   2 — Nextera (Nextera transposase Read 1 + Read 2 adapter)
 * User FASTA sources start at ID 3.
 */

#include <string>
#include <vector>
#include <cstdint>

class Options;
class Read;

// k-mer length and maximum number of tracked contamination sources.
static const int CONTAM_K            = 31;
static const int MAX_CONTAM_SOURCES  = 64;

// Hash table entry: 8 bytes, load factor ≤ 0.5.
// key == 0 means the slot is empty.
struct KhtEntry {
    uint32_t key;   // upper 32 bits of canonical NTHash (0 = empty sentinel)
    uint8_t  src;   // source ID (1-indexed)
    uint8_t  pad[3];
};

class ContaminantDB {
public:
    ContaminantDB();
    ~ContaminantDB();

    // Build the hash table from built-in sequences and opt->contamination.dbFile.
    // Safe to call before GPU init; GPU upload is done separately if needed.
    void build(Options* opt);

    // Scan a read against the hash table.
    // Returns the source ID with the most k-mer hits if hits >= minKmerHits,
    // otherwise returns 0.
    int scanRead(const Read* read) const;

    // Source name for a given source ID (0 returns "none").
    const std::string& sourceName(int srcId) const;
    int numSources() const { return (int)mSourceNames.size(); }

    // Flat hash table accessors for GPU upload (cuda_contaminant.cu).
    const KhtEntry* table()     const { return mTable; }
    uint32_t        tableSize() const { return mTableSize; }
    uint32_t        tableMask() const { return mTableMask; }
    int             minHits()   const { return mMinKmerHits; }

    // Built-in unit test.
    static bool test();

private:
    KhtEntry*               mTable;
    uint32_t                mTableSize;
    uint32_t                mTableMask;
    int                     mMinKmerHits;
    std::vector<std::string> mSourceNames;  // index 0 = "none"
    int                     mNumKmers;

    // Source management
    int  registerSource(const std::string& name);

    // Table building
    void allocTable(uint32_t capacity);
    bool tableInsert(uint64_t hashVal, uint8_t srcId);
    int  tableLookup(uint64_t hashVal) const;  // returns srcId or 0

    // Sequence ingestion
    void addSequence(const std::string& seq, int srcId);
    void addFastaFile(const std::string& path, const std::string& srcPrefix,
                      int firstSrcId);
    // Multi-source FASTA: groups separated by '# source: NAME' comment lines.
    // Returns the number of sources registered from the file.
    int  addMultiSourceFasta(const std::string& path);

    // Built-in adapters
    void addBuiltinSources();
};

#endif // CONTAMINANT_DB_H
