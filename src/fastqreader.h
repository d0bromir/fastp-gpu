/*
MIT License

Copyright (c) 2021 Shifu Chen <chen@haplox.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef FASTQ_READER_H
#define FASTQ_READER_H

#include <stdio.h>
#include <stdlib.h>
#include "read.h"
#include "common.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "igzip_lib.h"
#include "readpool.h"
#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP)
#include "cuda_gzip.h"
#endif
#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP) && defined(HAVE_GDS)
#include "gds_pipeline.h"
#endif

class FastqReader{
public:
FastqReader(string filename, bool hasQuality = true, bool phred64=false, bool useGDS=false);
~FastqReader();
bool isZipped();

void getBytes(size_t& bytesRead, size_t& bytesTotal);

//this function is not thread-safe
//do not call read() of a same FastqReader object from different threads concurrently
Read* read();
bool eof();
bool hasNoLineBreakAtEnd();
void setReadPool(ReadPool* rp);
// Start async double-buffered decompression.  Must be called after
// setReadPool() and only when mOptions->thread > 1.  Safe to call
// multiple times (idempotent).  Never called for evaluator readers.
void enableAsyncDecomp();

public:
static bool isZipFastq(string filename);
static bool isFastq(string filename);
static bool test();

private:
void init();
void close();
void getLine(string* line);
void clearLineBreaks(char* line);
void readToBuf();
void readToBufIgzip();
bool bufferFinished();
// Async decompression thread
void decompressThreadFunc();
void startDecompressThread();
void stopDecompressThread();
#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP)
void readToBufBgzfGpu();
#ifdef ENABLE_GPU_GZIP_WHOLE
void readToBufGzipGpu();
#endif
#endif
#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP) && defined(HAVE_GDS)
void readToBufGds();
#endif

private:
string mFilename;
struct isal_gzip_header mGzipHeader;
struct inflate_state mGzipState;
unsigned char *mGzipInputBuffer;
unsigned char *mGzipOutputBuffer;
size_t mGzipInputBufferSize;
size_t mGzipOutputBufferSize;
size_t mGzipInputUsedBytes;
FILE* mFile;
bool mZipped;
char* mFastqBuf;
int mBufDataLen;
int mBufUsedLen;
bool mStdinMode;
bool mHasNoLineBreakAtEnd;
long mCounter;
bool mHasQuality;
bool mPhred64;
ReadPool* mReadPool;

// ── Async double-buffered decompression ──────────────────────────────
// A background thread decompresses into mDecompBuf[!mDecompFront] while
// the reader parses from mFastqBuf (== mDecompBuf[mDecompFront]).
bool mAsyncDecomp;              // true if async decompress is active
char* mDecompBuf[2];            // double-buffer pair
int   mDecompLen[2];            // bytes decompressed in each buffer
int   mDecompFront;             // which buffer getLine() reads from (0 or 1)
std::thread* mDecompThread;     // background decompress thread
std::mutex mDecompMtx;
std::condition_variable mDecompCv;
bool mDecompReady;              // back buffer has data ready
bool mDecompRequested;          // request for back buffer fill
bool mDecompFinished;           // decompress thread should exit
bool mDecompEof;                // decompression reached EOF

#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP)
// ── BGZF + nvCOMP GPU decompression path ─────────────────────────────
// Activated automatically when the input file is detected as BGZF on open.
// Each readToBufBgzfGpu() call decompresses up to BGZF_READER_BATCH blocks.
static const int BGZF_READER_BATCH = 256; // blocks per GPU dispatch
bool                  mBgzfGpuMode;
CudaGzipDecompressor* mGzipDecompressor;
unsigned char*        mRawGzBuf;          // raw compressed staging area
size_t                mRawGzBufSize;       // sizeof(mRawGzBuf)
size_t                mRawGzPos;           // read cursor in mRawGzBuf
size_t                mRawGzEnd;           // valid bytes end in mRawGzBuf
size_t  mBgzfChunkOffsets[BGZF_READER_BATCH];
size_t  mBgzfChunkSizes  [BGZF_READER_BATCH];
bool                  mBgzfEof;
// output buffer: BGZF_READER_BATCH * BGZF_DECOMP_BYTES = up to 16 MB
unsigned char*        mBgzfDecompBuf;
size_t  mBgzfDecompOutSizes[BGZF_READER_BATCH];

#ifdef ENABLE_GPU_GZIP_WHOLE
// ── Standard gzip GPU decompression path ─────────────────────────────
// DISABLED by default (define ENABLE_GPU_GZIP_WHOLE to enable).
// ~97× slower than CPU igzip for single-stream gzip — see paper §8.4.
// The entire file is decompressed on GPU at open time; readToBufGzipGpu()
// serves 16 MB chunks from the pre-decompressed buffer.
bool           mGzipGpuMode;
unsigned char* mGzipDecompData;     // entire decompressed file content
size_t         mGzipDecompSize;     // total decompressed bytes
size_t         mGzipDecompPos;      // current read position
#endif // ENABLE_GPU_GZIP_WHOLE
#endif // HAVE_CUDA && HAVE_NVCOMP

#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP) && defined(HAVE_GDS)
// ── GPU-Direct Storage NVMe → GPU DMA pipeline ──────────────────────
// Activated with --use_gds when the file is on a GDS-capable filesystem.
// The entire decompression-to-parsing pipeline runs on the GPU;
// only the decompressed FASTQ text is copied to mFastqBuf.
bool                  mGdsMode;
GdsPipeline*          mGdsPipeline;
#endif // HAVE_CUDA && HAVE_NVCOMP && HAVE_GDS
};

class FastqReaderPair{
public:
FastqReaderPair(FastqReader* left, FastqReader* right);
FastqReaderPair(string leftName, string rightName, bool hasQuality = true, bool phred64 = false, bool interleaved = false, bool useGDS = false);
~FastqReaderPair();
void read(ReadPair* pair);
bool eof();
void enableAsyncDecomp();
public:
FastqReader* mLeft;
FastqReader* mRight;
bool mInterleaved;
};

#endif
