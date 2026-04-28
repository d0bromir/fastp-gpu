#ifndef BGZF_WRITER_H
#define BGZF_WRITER_H

// BGZF (Blocked GZip Format) output helpers.
//
// BGZF is the standard variant of gzip used by samtools/htslib for indexable
// random-access compressed files.  Each compressed block:
//   • Is a complete gzip member with FLG.FEXTRA set.
//   • Carries a 6-byte extra field with subfield identifier "BC" containing
//     the total block size minus one as a little-endian uint16.
//   • Has uncompressed payload ≤ 65280 bytes (so total block ≤ 65536).
//
// A valid BGZF file ends with a 28-byte empty-block "EOF marker".
//
// Compression backend is libdeflate (raw deflate inside each BGZF gzip
// member).  CRC32 is computed via libdeflate's optimized routine.

#include <stddef.h>
#include <stdint.h>

// Largest uncompressed payload per BGZF block.
static const size_t BGZF_BLOCK_PAYLOAD = 65280;

// 28-byte empty BGZF block emitted at end-of-file.
extern const unsigned char BGZF_EOF_MARKER[28];

// Opaque compressor handle.  One per writer thread (NOT thread-safe).
typedef struct BgzfCompressor BgzfCompressor;

// Allocate a compressor for the given fastp-style level (1..9).
// Returns NULL on failure.
BgzfCompressor* bgzf_compressor_alloc(int fastp_level);
void            bgzf_compressor_free(BgzfCompressor* c);

// Worst-case bound on BGZF output size for a buffer of @in_bytes.
size_t bgzf_compress_bound(size_t in_bytes);

// Compress @in_bytes from @src as a sequence of BGZF blocks into @dst.
// Returns the number of bytes written, or 0 on failure.
size_t bgzf_compress(BgzfCompressor* comp,
                     const void* src, size_t in_bytes,
                     void*       dst, size_t out_capacity);

#endif // BGZF_WRITER_H
