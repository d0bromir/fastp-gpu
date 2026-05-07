#include "bgzf_writer.h"
#include "libdeflate.h"
#include <stdlib.h>
#include <string.h>

// Empty BGZF block (28 bytes): a complete gzip member with zero uncompressed
// data.  htslib uses this as the canonical end-of-file marker.
const unsigned char BGZF_EOF_MARKER[28] = {
    0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00,
    0x00, 0xff, 0x06, 0x00, 0x42, 0x43, 0x02, 0x00,
    0x1b, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00
};

// Compressor handle wraps a libdeflate raw-deflate compressor.  Each instance
// is reused across many BGZF blocks within a worker thread.
struct BgzfCompressor {
    struct libdeflate_compressor* dc;
};

BgzfCompressor* bgzf_compressor_alloc(int fastp_level) {
    BgzfCompressor* c = (BgzfCompressor*)calloc(1, sizeof(BgzfCompressor));
    if (!c) return NULL;
    c->dc = libdeflate_alloc_compressor(fastp_level);
    if (!c->dc) { free(c); return NULL; }
    return c;
}

void bgzf_compressor_free(BgzfCompressor* c) {
    if (!c) return;
    if (c->dc) libdeflate_free_compressor(c->dc);
    free(c);
}

static inline void bgzf_put_u16le(unsigned char* p, uint16_t v) {
    p[0] = (unsigned char)(v       & 0xff);
    p[1] = (unsigned char)((v >> 8) & 0xff);
}

static inline void bgzf_put_u32le(unsigned char* p, uint32_t v) {
    p[0] = (unsigned char)(v        & 0xff);
    p[1] = (unsigned char)((v >>  8) & 0xff);
    p[2] = (unsigned char)((v >> 16) & 0xff);
    p[3] = (unsigned char)((v >> 24) & 0xff);
}

size_t bgzf_compress_bound(size_t in_bytes) {
    if (in_bytes == 0) return 28;
    size_t nblocks = (in_bytes + BGZF_BLOCK_PAYLOAD - 1) / BGZF_BLOCK_PAYLOAD;
    size_t per_block_overhead = 18 + 8 + 64;
    return in_bytes + nblocks * per_block_overhead + 28;
}

// Write one BGZF block: 18B header + raw deflate + 8B trailer.
static size_t bgzf_write_block(BgzfCompressor* c,
                               const unsigned char* src, size_t in_sz,
                               unsigned char* dst, size_t cap) {
    if (in_sz > BGZF_BLOCK_PAYLOAD) return 0;
    if (cap < 18 + 8) return 0;

    dst[0]  = 0x1f; dst[1]  = 0x8b;
    dst[2]  = 0x08;
    dst[3]  = 0x04;                           // FLG = FEXTRA
    dst[4]  = dst[5] = dst[6] = dst[7] = 0;   // MTIME = 0
    dst[8]  = 0x00;                           // XFL
    dst[9]  = 0xff;                           // OS = unknown
    dst[10] = 0x06; dst[11] = 0x00;           // XLEN = 6
    dst[12] = 'B'; dst[13] = 'C';
    dst[14] = 0x02; dst[15] = 0x00;           // SLEN = 2
    // dst[16..17] = BSIZE, filled in below

    size_t payload_cap = cap - 18 - 8;
    size_t deflated = libdeflate_deflate_compress(c->dc, src, in_sz,
                                                  dst + 18, payload_cap);
    if (deflated == 0) return 0;

    uint32_t crc = libdeflate_crc32(0, src, in_sz);
    bgzf_put_u32le(dst + 18 + deflated,     crc);
    bgzf_put_u32le(dst + 18 + deflated + 4, (uint32_t)in_sz);

    size_t total = 18 + deflated + 8;
    bgzf_put_u16le(dst + 16, (uint16_t)(total - 1));
    return total;
}

size_t bgzf_compress(BgzfCompressor* comp,
                     const void* src_v, size_t in_bytes,
                     void* dst_v, size_t out_capacity) {
    const unsigned char* src = (const unsigned char*)src_v;
    unsigned char*       dst = (unsigned char*)dst_v;
    size_t off = 0;
    size_t consumed = 0;

    while (consumed < in_bytes) {
        size_t take = in_bytes - consumed;
        if (take > BGZF_BLOCK_PAYLOAD) take = BGZF_BLOCK_PAYLOAD;
        if (off >= out_capacity) return 0;
        size_t n = bgzf_write_block(comp, src + consumed, take,
                                    dst + off, out_capacity - off);
        if (n == 0) return 0;
        off      += n;
        consumed += take;
    }
    return off;
}
