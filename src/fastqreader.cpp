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

#include "fastqreader.h"
#include "util.h"
#include "profiling.h"
#include <string.h>
#include <cassert>
#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP) && defined(HAVE_GDS)
#include <cuda_runtime.h>
#endif

#define FQ_BUF_SIZE (1<<24)     // 16MB - increased from 8MB for better I/O
#define IGZIP_IN_BUF_SIZE (1<<23)  // 8MB - increased from 4MB for faster decompression
#define GZIP_HEADER_BYTES_REQ (1<<17)  // 128KB - increased for better block alignment

// BGZF-GPU path: raw compressed and output buffer sizes
// 256 blocks × 65536 bytes = 16 MB each direction
#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP)
#define BGZF_RAW_BUF_SIZE  ((size_t)FastqReader::BGZF_READER_BATCH * BGZF_BLOCK_BYTES)
#define BGZF_OUT_BUF_SIZE  ((size_t)FastqReader::BGZF_READER_BATCH * BGZF_DECOMP_BYTES)
#endif

FastqReader::FastqReader(string filename, bool hasQuality, bool phred64, bool useGDS){
	mFilename = filename;
	mZipped = false;
	mFile = NULL;
	mStdinMode = false;
	mDecompBuf[0] = new char[FQ_BUF_SIZE];
	mDecompBuf[1] = new char[FQ_BUF_SIZE];
	mDecompLen[0] = 0;
	mDecompLen[1] = 0;
	mDecompFront = 0;
	mFastqBuf = mDecompBuf[0];
	mBufDataLen = 0;
	mBufUsedLen = 0;
	mHasNoLineBreakAtEnd = false;
	mGzipInputBufferSize = IGZIP_IN_BUF_SIZE;
	mGzipInputBuffer = new unsigned char[mGzipInputBufferSize];
	mGzipOutputBufferSize = FQ_BUF_SIZE;
	mGzipOutputBuffer = (unsigned char*)mFastqBuf;
	mCounter = 0;
	mPhred64 = phred64;
	mHasQuality = hasQuality;
	mHasNoLineBreakAtEnd = false;
	mGzipInputUsedBytes = 0;
	mReadPool = NULL;
	mAsyncDecomp = false;
	mDecompThread = NULL;
	mDecompReady = false;
	mDecompRequested = false;
	mDecompFinished = false;
	mDecompEof = false;
#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP)
	mBgzfGpuMode      = false;
	mGzipDecompressor = NULL;
	mRawGzBuf         = NULL;
	mRawGzBufSize     = 0;
	mRawGzPos         = 0;
	mRawGzEnd         = 0;
	mBgzfEof          = false;
	mBgzfDecompBuf    = NULL;
	memset(mBgzfChunkOffsets,   0, sizeof(mBgzfChunkOffsets));
	memset(mBgzfChunkSizes,     0, sizeof(mBgzfChunkSizes));
	memset(mBgzfDecompOutSizes, 0, sizeof(mBgzfDecompOutSizes));
#ifdef ENABLE_GPU_GZIP_WHOLE
	mGzipGpuMode      = false;
	mGzipDecompData   = NULL;
	mGzipDecompSize   = 0;
	mGzipDecompPos    = 0;
#endif
#endif
#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP) && defined(HAVE_GDS)
	mGdsMode     = useGDS;
	mGdsPipeline = NULL;
#endif
	init();
}

FastqReader::~FastqReader(){
	stopDecompressThread();
	close();
	delete[] mDecompBuf[0];
	delete[] mDecompBuf[1];
	mFastqBuf = NULL;
	delete[] mGzipInputBuffer;
#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP)
	if (mGzipDecompressor) { delete mGzipDecompressor; mGzipDecompressor = NULL; }
	if (mRawGzBuf)         { delete[] mRawGzBuf;        mRawGzBuf = NULL; }
	if (mBgzfDecompBuf)    { delete[] mBgzfDecompBuf;   mBgzfDecompBuf = NULL; }
#ifdef ENABLE_GPU_GZIP_WHOLE
	if (mGzipDecompData)   { delete[] mGzipDecompData;  mGzipDecompData = NULL; }
#endif
#endif
#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP) && defined(HAVE_GDS)
	if (mGdsPipeline)      { delete mGdsPipeline;       mGdsPipeline = NULL; }
#endif
}

bool FastqReader::hasNoLineBreakAtEnd() {
	return mHasNoLineBreakAtEnd;
}

void FastqReader::setReadPool(ReadPool* rp) {
	mReadPool = rp;
}


bool FastqReader::bufferFinished() {
#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP) && defined(HAVE_GDS)
	if (mGdsMode && mGdsPipeline) {
		return mGdsPipeline->eof() && (mBufUsedLen >= mBufDataLen);
	}
#endif
#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP)
	if (mBgzfGpuMode) {
		return mBgzfEof && (mBufUsedLen >= mBufDataLen);
	}
#ifdef ENABLE_GPU_GZIP_WHOLE
	if (mGzipGpuMode) {
		return (mGzipDecompPos >= mGzipDecompSize) && (mBufUsedLen >= mBufDataLen);
	}
#endif
#endif
	if(mZipped) {
		if (mAsyncDecomp)
			return mDecompEof && (mBufUsedLen >= mBufDataLen) && !mDecompReady;
		return eof() && mGzipState.avail_in == 0;
	} else {
		return eof();
	}
}

void FastqReader::readToBufIgzip(){
	mBufDataLen = 0;
	while(mBufDataLen == 0) {
		if(eof() && mGzipState.avail_in==0)
			return;
		if (mGzipState.avail_in == 0) {
			mGzipState.next_in = mGzipInputBuffer;
			// Use larger fread to reduce system calls and improve cache efficiency
			size_t read_size = fread(mGzipState.next_in, 1, mGzipInputBufferSize, mFile);
			mGzipState.avail_in = read_size;
			mGzipInputUsedBytes += read_size;
			PROF_ADD(io_bytes, read_size);
		}
		mGzipState.next_out = mGzipOutputBuffer;
		mGzipState.avail_out = mGzipOutputBufferSize;

		PROF_START(_t_decomp_sync);
		int ret = isal_inflate(&mGzipState);
		PROF_END(_t_decomp_sync, decompress_ns);
		if (ret != ISAL_DECOMP_OK) {
			error_exit("igzip: encountered while decompressing file: " + mFilename);
		}
		mBufDataLen = mGzipState.next_out - mGzipOutputBuffer;
		// Optimize: break immediately when we have data to avoid extra decompression passes
		if(mBufDataLen > 0 || eof() || mGzipState.avail_in>0)
			break;
	}
	// this block is finished
	if(mGzipState.block_state == ISAL_BLOCK_FINISH) {
		// a new block begins
		if(!eof() || mGzipState.avail_in > 0) {
			if (mGzipState.avail_in == 0) {
				isal_inflate_reset(&mGzipState);
				mGzipState.next_in = mGzipInputBuffer;
				mGzipState.avail_in = fread(mGzipState.next_in, 1, mGzipInputBufferSize, mFile);
				mGzipInputUsedBytes += mGzipState.avail_in;
			} else if (mGzipState.avail_in >= GZIP_HEADER_BYTES_REQ){
				unsigned char* old_next_in = mGzipState.next_in;
				size_t old_avail_in = mGzipState.avail_in;
				isal_inflate_reset(&mGzipState);
				mGzipState.avail_in = old_avail_in;
				mGzipState.next_in = old_next_in;
			} else {
				size_t old_avail_in = mGzipState.avail_in;
				memmove(mGzipInputBuffer, mGzipState.next_in, mGzipState.avail_in);
				size_t added = 0;
				if(!eof()) {
					added = fread(mGzipInputBuffer + mGzipState.avail_in, 1, mGzipInputBufferSize - mGzipState.avail_in, mFile);
					mGzipInputUsedBytes += added;
				}
				isal_inflate_reset(&mGzipState);
				mGzipState.next_in = mGzipInputBuffer;
				mGzipState.avail_in = old_avail_in + added;
			}
			int ret = isal_read_gzip_header(&mGzipState, &mGzipHeader);
			if (ret != ISAL_DECOMP_OK) {
				error_exit("igzip: invalid gzip header found");
			}
		}
	}

	if(eof() && mGzipState.avail_in == 0) {
		// all data was processed - fail if not at logical end of zip file (truncated?)
		if (mGzipState.block_state != ISAL_BLOCK_FINISH || !mGzipState.bfinal) {
			error_exit("igzip: unexpected eof");
		}
	}
}

// ── Async double-buffered decompression ──────────────────────────────────────
// Background thread decompresses into the back buffer while parsing proceeds
// on the front buffer. When the parser needs a refill, it waits for the back
// buffer to be ready, swaps front/back, and kicks off the next decompression.

void FastqReader::decompressThreadFunc() {
	while (true) {
		std::unique_lock<std::mutex> lk(mDecompMtx);
		mDecompCv.wait(lk, [this]{ return mDecompRequested || mDecompFinished; });
		if (mDecompFinished) return;
		mDecompRequested = false;
		lk.unlock();

		// Point igzip output into the back buffer
		int back = 1 - mDecompFront;
		unsigned char* outBuf = (unsigned char*)mDecompBuf[back];
		size_t outBufSize = FQ_BUF_SIZE;

		// Inline igzip decompression — same logic as readToBufIgzip() but
		// using a local dataLen to avoid racing with the parser thread's mBufDataLen.
		int dataLen = 0;
		while (dataLen == 0) {
			if (eof() && mGzipState.avail_in == 0)
				break;
			if (mGzipState.avail_in == 0) {
				mGzipState.next_in = mGzipInputBuffer;
				size_t read_size = fread(mGzipState.next_in, 1, mGzipInputBufferSize, mFile);
				mGzipState.avail_in = read_size;
				mGzipInputUsedBytes += read_size;
				PROF_ADD(io_bytes, read_size);
			}
			mGzipState.next_out = outBuf;
			mGzipState.avail_out = outBufSize;

			PROF_START(_t_decomp_async);
			int ret = isal_inflate(&mGzipState);
			PROF_END(_t_decomp_async, decompress_ns);
			if (ret != ISAL_DECOMP_OK) {
				error_exit("igzip: encountered while decompressing file: " + mFilename);
			}
			dataLen = mGzipState.next_out - outBuf;
			if (dataLen > 0 || eof() || mGzipState.avail_in > 0)
				break;
		}
		if (mGzipState.block_state == ISAL_BLOCK_FINISH) {
			if (!eof() || mGzipState.avail_in > 0) {
				if (mGzipState.avail_in == 0) {
					isal_inflate_reset(&mGzipState);
					mGzipState.next_in = mGzipInputBuffer;
					mGzipState.avail_in = fread(mGzipState.next_in, 1, mGzipInputBufferSize, mFile);
					mGzipInputUsedBytes += mGzipState.avail_in;
				} else if (mGzipState.avail_in >= GZIP_HEADER_BYTES_REQ) {
					unsigned char* old_next_in = mGzipState.next_in;
					size_t old_avail_in = mGzipState.avail_in;
					isal_inflate_reset(&mGzipState);
					mGzipState.avail_in = old_avail_in;
					mGzipState.next_in = old_next_in;
				} else {
					size_t old_avail_in = mGzipState.avail_in;
					memmove(mGzipInputBuffer, mGzipState.next_in, mGzipState.avail_in);
					size_t added = 0;
					if (!eof()) {
						added = fread(mGzipInputBuffer + mGzipState.avail_in, 1, mGzipInputBufferSize - mGzipState.avail_in, mFile);
						mGzipInputUsedBytes += added;
					}
					isal_inflate_reset(&mGzipState);
					mGzipState.next_in = mGzipInputBuffer;
					mGzipState.avail_in = old_avail_in + added;
				}
				int ret = isal_read_gzip_header(&mGzipState, &mGzipHeader);
				if (ret != ISAL_DECOMP_OK) {
					error_exit("igzip: invalid gzip header found");
				}
			}
		}

		lk.lock();
		mDecompLen[back] = dataLen;
		if (dataLen == 0 && eof() && mGzipState.avail_in == 0)
			mDecompEof = true;
		mDecompReady = true;
		mDecompCv.notify_one();
	}
}

void FastqReader::startDecompressThread() {
	mAsyncDecomp = true;
	mDecompFinished = false;
	mDecompThread = new std::thread(&FastqReader::decompressThreadFunc, this);
}

void FastqReader::stopDecompressThread() {
	if (!mDecompThread) return;
	{
		std::lock_guard<std::mutex> lk(mDecompMtx);
		mDecompFinished = true;
		mDecompCv.notify_one();
	}
	mDecompThread->join();
	delete mDecompThread;
	mDecompThread = NULL;
	mAsyncDecomp = false;
}

// Start the async double-buffered decompression background thread.
// Idempotent: safe to call multiple times; only the first call has effect.
// Must be called after init() (which fills the front buffer synchronously)
// and only when the caller decides async is worthwhile (i.e. mOptions->thread > 1).
void FastqReader::enableAsyncDecomp() {
	if (mAsyncDecomp) return;           // already running
	if (!mZipped || mStdinMode) return; // plain / stdin: no async needed
	if (bufferFinished()) return;       // tiny/empty file already done
	bool useAsync = true;
#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP)
	if (mBgzfGpuMode) useAsync = false; // GPU decompressor owns the buffer
#endif
#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP) && defined(HAVE_GDS)
	if (mGdsMode && mGdsPipeline) useAsync = false; // GDS pipeline
#endif
	if (!useAsync) return;
	{
		std::lock_guard<std::mutex> lk(mDecompMtx);
		mDecompRequested = true;
	}
	startDecompressThread();
}

#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP)
// ─────────────────────────────────────────────────────────────────────────────
// readToBufBgzfGpu
//
// Reads up to BGZF_READER_BATCH complete BGZF blocks from mFile, decompresses
// them all in a single nvCOMP batch call, and places the concatenated
// decompressed text into mFastqBuf.
//
// Sets mBufDataLen to the total decompressed bytes on success.
// Sets mBgzfEof when no more data is available.
// ─────────────────────────────────────────────────────────────────────────────
void FastqReader::readToBufBgzfGpu() {
	mBufDataLen = 0;

	if (mBgzfEof) return;

	// ── 1. Refill raw compressed buffer ──────────────────────────────────
	// Slide remaining unconsumed data to the front, then top up from file.
	if (mRawGzPos > 0 && mRawGzEnd > mRawGzPos) {
		size_t remaining = mRawGzEnd - mRawGzPos;
		memmove(mRawGzBuf, mRawGzBuf + mRawGzPos, remaining);
		mRawGzEnd = remaining;
		mRawGzPos = 0;
	} else if (mRawGzPos >= mRawGzEnd) {
		mRawGzEnd = 0;
		mRawGzPos = 0;
	}

	// Top-up: fill the rest of the buffer from file
	if (mRawGzEnd < mRawGzBufSize && !feof(mFile)) {
		size_t space = mRawGzBufSize - mRawGzEnd;
		size_t got   = fread(mRawGzBuf + mRawGzEnd, 1, space, mFile);
		mRawGzEnd   += got;
	}

	if (mRawGzEnd == 0) {
		mBgzfEof = true;
		return;
	}

	// ── 2. Parse BGZF block headers to find chunk boundaries ─────────────
	int num_chunks = 0;
	size_t pos = 0;

	while (pos < mRawGzEnd && num_chunks < BGZF_READER_BATCH) {
		// Need at least BGZF_SIGNATURE_BYTES bytes to read the header
		if (mRawGzEnd - pos < BGZF_SIGNATURE_BYTES) break;

		size_t block_size = CudaGzipDecompressor::bgzfBlockSize(
			mRawGzBuf + pos, mRawGzEnd - pos);
		if (block_size == 0) break;  // malformed block or incomplete header
		if (pos + block_size > mRawGzEnd) break;  // block not fully buffered

		// Detect BGZF EOF marker: empty block has bsize exactly 28
		// (28 = 10-byte header + 2 xlen + 6 SI/SLEN + BSIZE + 8 deflate/CRC)
		if (block_size == 28) {
			// empty EOF block — done
			pos += block_size;
			mBgzfEof = true;
			break;
		}

		mBgzfChunkOffsets[num_chunks] = pos;
		mBgzfChunkSizes  [num_chunks] = block_size;
		num_chunks++;
		pos += block_size;
	}

	if (num_chunks == 0) {
		// No complete blocks available
		if (feof(mFile)) mBgzfEof = true;
		return;
	}

	// Advance raw buffer cursor
	mRawGzPos = pos;

	// ── 3. GPU batch decompression ────────────────────────────────────────
	int rc = mGzipDecompressor->decompress(
		mRawGzBuf,
		mBgzfChunkOffsets,
		mBgzfChunkSizes,
		(size_t)num_chunks,
		mBgzfDecompBuf,
		mBgzfDecompOutSizes);

	if (rc != 0) {
		error_exit("[cuda_gzip] GPU decompression failed for: " + mFilename);
	}

	// ── 4. Concatenate decompressed output into mFastqBuf ─────────────────
	size_t total_out = 0;
	for (int i = 0; i < num_chunks; i++) {
		size_t sz = mBgzfDecompOutSizes[i];
		if (total_out + sz > (size_t)FQ_BUF_SIZE) {
			// Safety: should never happen since 256 * 65536 == FQ_BUF_SIZE
			sz = (size_t)FQ_BUF_SIZE - total_out;
		}
		memcpy(mFastqBuf + total_out,
		       mBgzfDecompBuf + (size_t)i * BGZF_DECOMP_BYTES,
		       sz);
		total_out += sz;
	}
	mBufDataLen = (int)total_out;

	// If we hit EOF and there's no more data in the raw buffer beyond what
	// we just processed, mark done.
	if (feof(mFile) && mRawGzPos >= mRawGzEnd && !mBgzfEof) {
		// Let the next call detect EOF properly
	}
}

#ifdef ENABLE_GPU_GZIP_WHOLE
// ─────────────────────────────────────────────────────────────────────────────
// readToBufGzipGpu
//
// Serves 16 MB chunks from the pre-decompressed buffer (mGzipDecompData)
// which was filled during init() via GPU whole-file gzip decompression.
// NOTE: Disabled by default—~97× slower than CPU igzip.  Define
//       ENABLE_GPU_GZIP_WHOLE at compile time to include this path.
// ─────────────────────────────────────────────────────────────────────────────
void FastqReader::readToBufGzipGpu() {
	mBufDataLen = 0;
	if (mGzipDecompPos >= mGzipDecompSize) return;

	size_t remaining = mGzipDecompSize - mGzipDecompPos;
	size_t chunk = remaining;
	if (chunk > (size_t)FQ_BUF_SIZE) chunk = (size_t)FQ_BUF_SIZE;

	memcpy(mFastqBuf, mGzipDecompData + mGzipDecompPos, chunk);
	mGzipDecompPos += chunk;
	mBufDataLen = (int)chunk;
}
#endif // ENABLE_GPU_GZIP_WHOLE
#endif // HAVE_CUDA && HAVE_NVCOMP

#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP) && defined(HAVE_GDS)
// ─────────────────────────────────────────────────────────────────────────────
// readToBufGds
//
// Uses the GDS pipeline (NVMe → GPU DMA → GPU decompress → GPU parse) and
// copies only the final decompressed FASTQ text to mFastqBuf.
// CPU involvement is limited to this memcpy and orchestrating the pipeline.
// ─────────────────────────────────────────────────────────────────────────────
void FastqReader::readToBufGds() {
	mBufDataLen = 0;

	if (!mGdsPipeline || mGdsPipeline->eof()) return;

	ssize_t decomp = mGdsPipeline->readAndDecompress();
	if (decomp <= 0) return;

	size_t text_bytes = mGdsPipeline->decompressedBytes();
	if (text_bytes > (size_t)FQ_BUF_SIZE) text_bytes = (size_t)FQ_BUF_SIZE;

	/* D2H: copy decompressed FASTQ text from GPU to mFastqBuf */
	cudaMemcpy(mFastqBuf, mGdsPipeline->decompressedDevicePtr(),
	           text_bytes, cudaMemcpyDeviceToHost);

	mBufDataLen = (int)text_bytes;
}
#endif // HAVE_CUDA && HAVE_NVCOMP && HAVE_GDS

void FastqReader::readToBuf() {
	mBufDataLen = 0;
	PROF_START(_t_io);
#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP) && defined(HAVE_GDS)
	if (mGdsMode && mGdsPipeline) {
		readToBufGds();
		mBufUsedLen = 0;
		if (bufferFinished() && mBufDataLen > 0)
			if (mFastqBuf[mBufDataLen - 1] != '\n')
				mHasNoLineBreakAtEnd = true;
		PROF_END(_t_io, io_read_ns);
		return;
	}
#endif
#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP)
	if (mBgzfGpuMode) {
		readToBufBgzfGpu();
		mBufUsedLen = 0;
		if (bufferFinished() && mBufDataLen > 0)
			if (mFastqBuf[mBufDataLen - 1] != '\n')
				mHasNoLineBreakAtEnd = true;
		PROF_END(_t_io, io_read_ns);
		return;
	}
#ifdef ENABLE_GPU_GZIP_WHOLE
	if (mGzipGpuMode) {
		readToBufGzipGpu();
		mBufUsedLen = 0;
		if (bufferFinished() && mBufDataLen > 0)
			if (mFastqBuf[mBufDataLen - 1] != '\n')
				mHasNoLineBreakAtEnd = true;
		PROF_END(_t_io, io_read_ns);
		return;
	}
#endif
#endif
	if (mAsyncDecomp) {
		// Wait for back buffer to be filled by decompress thread
		std::unique_lock<std::mutex> lk(mDecompMtx);
		mDecompCv.wait(lk, [this]{ return mDecompReady || mDecompFinished; });
		if (!mDecompReady) {
			// Thread exited without data (shouldn't happen normally)
			mBufDataLen = 0;
			mBufUsedLen = 0;
			PROF_END(_t_io, io_read_ns);
			return;
		}
		// Swap: back buffer becomes front, old front becomes back
		mDecompFront = 1 - mDecompFront;
		mFastqBuf = mDecompBuf[mDecompFront];
		mBufDataLen = mDecompLen[mDecompFront];
		mDecompReady = false;

		if (!mDecompEof) {
			// Kick off next back-buffer fill
			mDecompRequested = true;
			mDecompCv.notify_one();
		}
		lk.unlock();
	} else if(mZipped) {
		readToBufIgzip();
	} else {
		if(!eof()) {
			mBufDataLen = fread(mFastqBuf, 1, FQ_BUF_SIZE, mFile);
			PROF_ADD(io_bytes, mBufDataLen);
		}
	}
	mBufUsedLen = 0;

	if(bufferFinished() && mBufDataLen>0) {
		if(mFastqBuf[mBufDataLen-1] != '\n')
			mHasNoLineBreakAtEnd = true;
	}
	PROF_END(_t_io, io_read_ns);
}

void FastqReader::init(){
	if (ends_with(mFilename, ".gz")){
		mFile = fopen(mFilename.c_str(), "rb");
		if(mFile == NULL) {
			error_exit("Failed to open file: " + mFilename);
		}
#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP) && defined(HAVE_GDS)
		// ── Try GPU-Direct Storage pipeline first ─────────────────────────
		if (mGdsMode && GdsPipeline::isAvailable()) {
			// Probe for BGZF signature
			unsigned char probe[18];
			size_t probe_n = fread(probe, 1, 18, mFile);
			rewind(mFile);
			if (probe_n >= 18 && probe[0] == 0x1f && probe[1] == 0x8b &&
			    probe[2] == 8 && (probe[3] & 0x04) &&
			    probe[12] == 'B' && probe[13] == 'C') {
				mGdsPipeline = new GdsPipeline(0);
				if (mGdsPipeline->valid() &&
				    mGdsPipeline->open(mFilename.c_str()) == 0) {
					mZipped = true;
					fputs("[fastp] GPU-Direct Storage pipeline activated "
					      "(NVMe -> GPU DMA)\n", stderr);
					readToBuf();
					return;
				} else {
					fputs("[fastp] GDS pipeline init failed, "
					      "falling back to standard GPU/CPU path\n",
					      stderr);
					delete mGdsPipeline;
					mGdsPipeline = NULL;
					mGdsMode = false;
				}
			}
		}
#endif
#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP)
		// ── Detect BGZF and set up GPU decompressor ───────────────────────
		// Read a header probe to check for BGZF signature.
		unsigned char probe[BGZF_SIGNATURE_BYTES];
		size_t probe_n = fread(probe, 1, BGZF_SIGNATURE_BYTES, mFile);
		if (probe_n == BGZF_SIGNATURE_BYTES &&
		    CudaGzipDecompressor::isBgzf(probe, probe_n)) {
			// Rewind so readToBufBgzfGpu can read from start
			rewind(mFile);
			// Allocate raw compressed buffer + output buffer
			mRawGzBufSize  = BGZF_RAW_BUF_SIZE;
			mRawGzBuf      = new unsigned char[mRawGzBufSize];
			mBgzfDecompBuf = new unsigned char[BGZF_OUT_BUF_SIZE];
			// Create decompressor on device 0 (primary)
			mGzipDecompressor = new CudaGzipDecompressor(0);
			if (!mGzipDecompressor->valid()) {
				// GPU init failed – fall back to CPU igzip
				fputs("[fastp] BGZF GPU decompressor init failed, using CPU\n", stderr);
				delete mGzipDecompressor; mGzipDecompressor = NULL;
				delete[] mRawGzBuf;       mRawGzBuf = NULL;
				delete[] mBgzfDecompBuf;  mBgzfDecompBuf = NULL;
				rewind(mFile);
			} else {
				mBgzfGpuMode = true;
				mZipped      = true;  // still a gzip file for bookkeeping
				readToBuf();
				return;
			}
		} else {
			// Not BGZF: seek back to start
			if (probe_n > 0) rewind(mFile);

			// NOTE: GPU decompression of standard (non-BGZF) gzip is NOT
			// enabled by default.  Standard gzip uses a single monolithic
			// DEFLATE stream which cannot be parallelised—nvCOMP processes it
			// sequentially on one GPU thread, making it ~100x slower than
			// ISA-L igzip on CPU.  Only BGZF (blocked gzip with independent
			// 64 KB blocks) benefits from GPU-parallel decompression.
			//
			// The implementation (CudaGzipDecompressor::decompressGzipWhole)
			// exists and is functionally correct; to benchmark it, uncomment
			// the block below or gate it behind a --gpu_decompress flag.
#ifdef ENABLE_GPU_GZIP_WHOLE  // disabled by default: ~97x slower than CPU igzip
			if (CudaGzipDecompressor::isGzip(probe, probe_n)) {
				// Get file size
				fseek(mFile, 0, SEEK_END);
				long fsize = ftell(mFile);
				rewind(mFile);

				if (fsize > 0) {
					size_t file_bytes = (size_t)fsize;
					unsigned char* file_buf = new unsigned char[file_bytes];
					size_t nread = fread(file_buf, 1, file_bytes, mFile);
					if (nread == file_bytes) {
						CudaGzipDecompressor decomp(0);
						if (decomp.valid()) {
							unsigned char* decomp_out = nullptr;
							size_t decomp_sz = 0;
							fprintf(stderr, "[fastp] Attempting GPU gzip decompression "
							        "of %s (%.1f MB)...\n",
							        mFilename.c_str(),
							        file_bytes / (1024.0 * 1024.0));
							int rc = decomp.decompressGzipWhole(
								file_buf, file_bytes,
								&decomp_out, &decomp_sz);
							if (rc == 0 && decomp_out && decomp_sz > 0) {
								fprintf(stderr, "[fastp] GPU gzip decompression OK: "
								        "%.1f MB -> %.1f MB\n",
								        file_bytes / (1024.0 * 1024.0),
								        decomp_sz / (1024.0 * 1024.0));
								mGzipDecompData = decomp_out;
								mGzipDecompSize = decomp_sz;
								mGzipDecompPos  = 0;
								mGzipGpuMode    = true;
								mZipped         = true;
								delete[] file_buf;
								readToBuf();
								return;
							} else {
								fprintf(stderr, "[fastp] GPU gzip decompression failed, "
								        "falling back to CPU igzip\n");
								if (decomp_out) delete[] decomp_out;
							}
						}
					}
					delete[] file_buf;
					rewind(mFile);
				}
			}
#endif
		}
#endif
		isal_gzip_header_init(&mGzipHeader);
		isal_inflate_init(&mGzipState);
		mGzipState.crc_flag = ISAL_GZIP_NO_HDR_VER;
		mGzipState.next_in = mGzipInputBuffer;
		mGzipState.avail_in = fread(mGzipState.next_in, 1, mGzipInputBufferSize, mFile);
		mGzipInputUsedBytes += mGzipState.avail_in;
		int ret = isal_read_gzip_header(&mGzipState, &mGzipHeader);
		if (ret != ISAL_DECOMP_OK) {
			error_exit("igzip: Error invalid gzip header found: "  + mFilename);
		}
		mZipped = true;
	}
	else {
		if(mFilename == "/dev/stdin") {
			mFile = stdin;
		}
		else
			mFile = fopen(mFilename.c_str(), "rb");
		if(mFile == NULL) {
			error_exit("Failed to open file: " + mFilename);
		}
		mZipped = false;
	}
	readToBuf();
	// Async decompression is NOT started here.  The main reader calls
	// enableAsyncDecomp() explicitly after construction so that evaluator
	// FastqReader instances (which read only a small sample) never spawn
	// a background thread, and so T=1 runs can avoid concurrent fread()
	// calls that hurt throughput on slow/network filesystems.
}

void FastqReader::getBytes(size_t& bytesRead, size_t& bytesTotal) {
#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP) && defined(HAVE_GDS)
	if (mGdsMode && mGdsPipeline) {
		// GDS tracks file offset internally; approximate progress
		bytesRead = 0;  // not directly accessible; use file size as proxy
		ifstream is(mFilename);
		is.seekg(0, is.end);
		bytesTotal = is.tellg();
		if (mGdsPipeline->eof()) bytesRead = bytesTotal;
		return;
	}
#endif
	if(mZipped) {
		bytesRead = mGzipInputUsedBytes - mGzipState.avail_in;
	} else {
		bytesRead = ftell(mFile);//mFile.tellg();
	}
	// use another ifstream to not affect current reader
	ifstream is(mFilename);
	is.seekg (0, is.end);
	bytesTotal = is.tellg();
}

void FastqReader::clearLineBreaks(char* line) {

	// trim \n, \r or \r\n in the tail
	int readed = strlen(line);
	if(readed >=2 ){
		if(line[readed-1] == '\n' || line[readed-1] == '\r'){
			line[readed-1] = '\0';
			if(line[readed-2] == '\r')
				line[readed-2] = '\0';
		}
	}
}

bool FastqReader::eof() {
#if defined(HAVE_CUDA) && defined(HAVE_NVCOMP) && defined(HAVE_GDS)
	if (mGdsMode && mGdsPipeline)
		return mGdsPipeline->eof();
#endif
	return feof(mFile);//mFile.eof();
}

// Optimized getLine: uses NEON-accelerated memchr() to scan for newlines
// ~16× faster than byte-by-byte on ARM aarch64 (16 bytes/cycle vs 1).
// Only called for cross-buffer records (~0.01% of reads); the fast path
// in read() handles the common case without calling getLine() at all.
void FastqReader::getLine(string* line){
	int start = mBufUsedLen;
	int remaining = mBufDataLen - start;

	// SIMD-accelerated newline search
	const char* nl = (const char*)memchr(mFastqBuf + start, '\n', remaining);

	if (nl) {
		// Newline found in current buffer — fast path
		int len = (int)(nl - (mFastqBuf + start));
		// Handle \r\n: strip trailing \r
		if (len > 0 && mFastqBuf[start + len - 1] == '\r')
			len--;
		line->assign(mFastqBuf + start, len);
		mBufUsedLen = (int)(nl + 1 - mFastqBuf);
		return;
	}

	// No newline found — check for bare \r (legacy Mac line endings)
	const char* cr = (const char*)memchr(mFastqBuf + start, '\r', remaining);
	if (cr) {
		int len = (int)(cr - (mFastqBuf + start));
		line->assign(mFastqBuf + start, len);
		mBufUsedLen = (int)(cr + 1 - mFastqBuf);
		return;
	}

	// No line ending in buffer at all
	if (bufferFinished()) {
		line->assign(mFastqBuf + start, remaining);
		mBufUsedLen = mBufDataLen;
		return;
	}

	// Line spans buffer boundary — save partial content and read more
	line->assign(mFastqBuf + start, remaining);

	while(true) {
		readToBuf();
		start = 0;
		// Skip leading line breaks if line is empty (boundary artifact)
		if(line->empty()) {
			while(start < mBufDataLen && (mFastqBuf[start] == '\r' || mFastqBuf[start] == '\n'))
				start++;
		}

		remaining = mBufDataLen - start;
		nl = (const char*)memchr(mFastqBuf + start, '\n', remaining);

		if (nl) {
			int len = (int)(nl - (mFastqBuf + start));
			if (len > 0 && mFastqBuf[start + len - 1] == '\r')
				len--;
			line->append(mFastqBuf + start, len);
			// Strip \r that may have been at end of previous buffer portion
			if (!line->empty() && line->back() == '\r')
				line->pop_back();
			mBufUsedLen = (int)(nl + 1 - mFastqBuf);
			return;
		}

		if (bufferFinished()) {
			line->append(mFastqBuf + start, remaining);
			if (!line->empty() && line->back() == '\r')
				line->pop_back();
			mBufUsedLen = mBufDataLen;
			return;
		}

		// Entire buffer consumed without newline (very rare)
		line->append(mFastqBuf + start, remaining);
	}
}

Read* FastqReader::read(){
	if(mBufUsedLen >= mBufDataLen && bufferFinished()) {
		return NULL;
	}

	string* name;
	string* sequence;
	string* strand;
	string* quality;

	Read* readInPool = NULL;
	if(mReadPool)
		readInPool = mReadPool->getOne();

	if(readInPool) {
		name = readInPool->mName;
		sequence = readInPool->mSeq;
		strand = readInPool->mStrand;
		quality = readInPool->mQuality;
	} else {
		name = new string();
		sequence = new string();
		strand = new string();
		quality = new string();
	}

	// ── Fast path: parse 4-line FASTQ record using NEON-accelerated memchr ──
	// Finds all 4 newlines in the current buffer without per-line function
	// call overhead. Handles >99.99% of reads (fails only at 16MB buffer
	// boundaries, where we fall through to the slow path).
	if (mBufUsedLen < mBufDataLen) {
		const char* p = mFastqBuf + mBufUsedLen;
		const char* bufEnd = mFastqBuf + mBufDataLen;
		const char* nl[4];
		const char* scan = p;
		int i;
		for (i = 0; i < 4; i++) {
			nl[i] = (const char*)memchr(scan, '\n', bufEnd - scan);
			if (!nl[i]) break;
			scan = nl[i] + 1;
		}

		if (i == 4) {
			// All 4 newlines found — compute line starts and lengths
			const char* starts[4] = { p, nl[0]+1, nl[1]+1, nl[2]+1 };
			int lens[4];
			for (int j = 0; j < 4; j++) {
				lens[j] = (int)(nl[j] - starts[j]);
				if (lens[j] > 0 && starts[j][lens[j]-1] == '\r')
					lens[j]--;  // strip \r from \r\n
			}

			// Validate FASTQ record structure
			if (lens[0] > 0 && starts[0][0] == '@' &&
			    lens[2] > 0 && starts[2][0] == '+' &&
			    lens[3] == lens[1]) {
				// H-09: reject quality scores outside valid Phred+33 range (ASCII
				// 33–126) before the data reaches SOUP compression libraries.
				for (int k = 0; k < lens[3]; k++) {
					unsigned char qc = (unsigned char)starts[3][k];
					if (qc < 33 || qc > 126)
						error_exit("Invalid quality score (ASCII " +
							to_string((int)qc) + ") at position " +
							to_string(k) + " in read " +
							string(starts[0], lens[0]) +
							" in " + mFilename +
							". Phred+33 quality must be ASCII 33-126.");
				}

				name->assign(starts[0], lens[0]);
				sequence->assign(starts[1], lens[1]);
				strand->assign(starts[2], lens[2]);
				quality->assign(starts[3], lens[3]);

				mBufUsedLen = (int)(nl[3] + 1 - mFastqBuf);

				if (readInPool) return readInPool;
				return new Read(name, sequence, strand, quality, mPhred64);
			}
		}
	}

	// ── Slow path: line-by-line parsing for cross-buffer records ─────────
	getLine(name);
	// name should start with @
	while((name->empty() && !(mBufUsedLen >= mBufDataLen && bufferFinished())) || (!name->empty() && (*name)[0]!='@')){
		getLine(name);
	}
	if(name->empty())
		return NULL;

	getLine(sequence);
	getLine(strand);
	getLine(quality);

	if (strand->empty() || (*strand)[0]!='+') {
		error_exit("Invalid FASTQ record in " + mFilename +
			": expected '+' strand line after read " + *name +
			", got: " + *strand);
	}

	if(quality->length() != sequence->length()) {
		error_exit("Invalid FASTQ record in " + mFilename +
			": sequence length (" + to_string(sequence->length()) +
			") != quality length (" + to_string(quality->length()) +
			") for read " + *name);
	}

	// H-09: reject quality scores outside valid Phred+33 range (slow path)
	for (size_t k = 0; k < quality->size(); k++) {
		unsigned char qc = (unsigned char)(*quality)[k];
		if (qc < 33 || qc > 126)
			error_exit("Invalid quality score (ASCII " +
				to_string((int)qc) + ") at position " +
				to_string(k) + " in read " + *name +
				" in " + mFilename +
				". Phred+33 quality must be ASCII 33-126.");
	}

	if(readInPool)
		return readInPool;
	else
		return new Read(name, sequence, strand, quality, mPhred64);
}

void FastqReader::close(){
	if (mFile){
		fclose(mFile);
		mFile = NULL;
	}
}

bool FastqReader::isZipFastq(string filename) {
	if (ends_with(filename, ".fastq.gz"))
		return true;
	else if (ends_with(filename, ".fq.gz"))
		return true;
	else if (ends_with(filename, ".fasta.gz"))
		return true;
	else if (ends_with(filename, ".fa.gz"))
		return true;
	else
		return false;
}

bool FastqReader::isFastq(string filename) {
	if (ends_with(filename, ".fastq"))
		return true;
	else if (ends_with(filename, ".fq"))
		return true;
	else if (ends_with(filename, ".fasta"))
		return true;
	else if (ends_with(filename, ".fa"))
		return true;
	else
		return false;
}

bool FastqReader::isZipped(){
	return mZipped;
}

bool FastqReader::test(){
	FastqReader reader1("testdata/R1.fq");
	FastqReader reader2("testdata/R1.fq");
	Read* r1 = NULL;
	Read* r2 = NULL;
	int i=0;
	while(true){
		i++;
		r1=reader1.read();
		r2=reader2.read();
		if(r1 == NULL || r2==NULL)
			break;
		r1->print();
		r2->print();
		delete r1;
		delete r2;
	}
	return true;
}

FastqReaderPair::FastqReaderPair(FastqReader* left, FastqReader* right){
	mLeft = left;
	mRight = right;
}

FastqReaderPair::FastqReaderPair(string leftName, string rightName, bool hasQuality, bool phred64, bool interleaved, bool useGDS){
	mInterleaved = interleaved;
	mLeft = new FastqReader(leftName, hasQuality, phred64, useGDS);
	if(mInterleaved)
		mRight = NULL;
	else
		mRight = new FastqReader(rightName, hasQuality, phred64, useGDS);
}

FastqReaderPair::~FastqReaderPair(){
	if(mLeft){
		delete mLeft;
		mLeft = NULL;
	}
	if(mRight){
		delete mRight;
		mRight = NULL;
	}
}

void FastqReaderPair::read(ReadPair* pair){
	Read* l = mLeft->read();
	Read* r = NULL;
	if(mInterleaved)
		r = mLeft->read();
	else
		r = mRight->read();
	pair->setPair(l, r);
}

void FastqReaderPair::enableAsyncDecomp() {
	if (mLeft)  mLeft->enableAsyncDecomp();
	if (mRight) mRight->enableAsyncDecomp();
}
