#include "writerthread.h"
#include "util.h"
#include "bgzf_writer.h"
#include "profiling.h"
#include <memory.h>
#include <stdlib.h>
#include <unistd.h>
#include <algorithm>

// Number of parallel compression threads per WriterThread.
//
// Fairness contract (vs. upstream OpenGene fastp):
//   The user's `-w N` flag is the total fastp worker budget.  Across all
//   gzipped main output writers, the SUM of compressor pool sizes must
//   equal N (matching opengene, where compression runs serially inside
//   each writer thread, so total compressor work = N processor threads
//   handing buffers to ~1 thread per writer).
//
// Implementation: divide N by the number of *main* gzipped output files
// (1 for SE, 2 for PE).  Auxiliary writers (--unpaired1/2, --merged_out,
// --failed_out, --overlapped_out) are best-effort and reuse the same
// per-writer share; they are not on the hot path.
//
// Override via FASTP_COMPRESSORS env var (sets the per-writer pool size
// directly, bypassing the fairness divider — for benchmarking only).
static int numCompressors(Options* opt) {
    const char* env = getenv("FASTP_COMPRESSORS");
    if (env && *env) {
        int v = atoi(env);
        if (v > 0) return v;
    }
    int processorThreads = opt->thread;
    int mainWriters = (opt->isPaired() && !opt->out2.empty()) ? 2 : 1;
    int n = processorThreads / mainWriters;
    if (n < 1) n = 1;
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu < 1) ncpu = 4;
    if (n > (int)ncpu) n = (int)ncpu;
    return n;
}

WriterThread::WriterThread(Options* opt, string filename, bool isSTDOUT){
    mOptions = opt;
    mWriter1 = NULL;
    mInputCompleted = false;
    mFilename = filename;
    mIsSTDOUT = isSTDOUT;
    mWorkingBufferList = 0;
    mBufferLength = 0;
    mNextSeqno = 0;
    mNextWriteSeqno = 0;
    mJobsDone = false;
    mOutDone = false;
    mActiveCompressors = 0;

    initWriter(filename, isSTDOUT);
    mIsGzip = mWriter1->isZipped();
    initBufferLists();

    if (mIsGzip && !mIsSTDOUT) {
        mNumCompressors = numCompressors(mOptions);
        startCompressPool();
    } else {
        mNumCompressors = 0;
    }
}

WriterThread::~WriterThread() {
    cleanup();
}

bool WriterThread::isCompleted()
{
    return mInputCompleted && (mBufferLength == 0);
}

bool WriterThread::setInputCompleted() {
    mInputCompleted = true;
    for(int t=0; t<mOptions->thread; t++) {
        mBufferLists[t]->setProducerFinished();
    }
    return true;
}

// Called by the writerTask loop in peprocessor/seprocessor.
// In parallel-compress mode this just drains the input SPSC lists and
// enqueues jobs for the compress pool; the actual write happens in
// writerWorker().  In plain (non-gz) mode it writes directly as before.
void WriterThread::output(){
    SingleProducerSingleConsumerList<string*>* list = mBufferLists[mWorkingBufferList];
    if (!list->canBeConsumed()) {
        usleep(100);
        return;
    }
    string* str = list->consume();
    mWorkingBufferList = (mWorkingBufferList + 1) % mOptions->thread;

    if (mNumCompressors > 0) {
        // Hand off to compress pool.
        // NOTE: do NOT decrement mBufferLength here — it stays elevated until
        // writerWorker() actually flushes the chunk to disk.  This keeps the
        // backpressure check in seprocessor/peprocessor accurate; without this
        // the processor threads can queue the entire file in memory before a
        // single byte is written, causing multi-GB heap growth.
        PendingJob job;
        job.seqno = mNextSeqno.fetch_add(1, std::memory_order_relaxed);
        job.data  = str;  // compress worker frees this
        {
            std::lock_guard<std::mutex> lk(mJobMtx);
            mJobQueue.push(job);
        }
        mJobCv.notify_one();
    } else {
        // Plain (non-gz) or stdout: write directly then release the counter.
        PROF_START(_t_write_direct);
        mWriter1->write(str->data(), str->length());
        PROF_END(_t_write_direct, write_ns);
        delete str;
        mBufferLength--;
    }
}

void WriterThread::input(int tid, string* data) {
    mBufferLists[tid]->produce(data);
    mBufferLength++;
}

void WriterThread::startCompressPool() {
    mActiveCompressors = mNumCompressors;
    for (int i = 0; i < mNumCompressors; i++) {
        mCompressThreads.emplace_back(&WriterThread::compressWorker, this);
    }
    mWriterThread = std::thread(&WriterThread::writerWorker, this);
}

void WriterThread::stopCompressPool() {
    // Signal compress workers that no more jobs will arrive
    {
        std::lock_guard<std::mutex> lk(mJobMtx);
        mJobsDone = true;
    }
    mJobCv.notify_all();
    for (auto& t : mCompressThreads)
        if (t.joinable()) t.join();
    mCompressThreads.clear();
    // writerWorker exits once all compressors are done and outQueue is drained
    if (mWriterThread.joinable())
        mWriterThread.join();
}

void WriterThread::compressWorker() {
    // Each worker has its own BGZF (ISA-L) compressor instance.
    BgzfCompressor* comp = bgzf_compressor_alloc(mOptions->compression);
    if (!comp) {
        error_exit("WriterThread: failed to allocate BGZF compressor (ISA-L)");
    }

    while (true) {
        PendingJob job;
        {
            std::unique_lock<std::mutex> lk(mJobMtx);
            mJobCv.wait(lk, [this]{ return !mJobQueue.empty() || mJobsDone; });
            if (mJobQueue.empty()) {
                // No more jobs and queue is empty
                break;
            }
            job = mJobQueue.front();
            mJobQueue.pop();
        }

        // Compress the string as a sequence of BGZF blocks.  The EOF
        // marker is appended once by Writer::close() at file end.
        const char* src  = job.data->data();
        size_t      srcSz = job.data->size();
        size_t      bound = bgzf_compress_bound(srcSz);
        char*       dst   = (char*)malloc(bound);
        PROF_START(_t_comp);
        size_t      dstSz = bgzf_compress(comp, src, srcSz, dst, bound);
        PROF_END(_t_comp, compress_ns);

        delete job.data;

        CompressedChunk chunk;
        chunk.seqno   = job.seqno;
        chunk.data    = dst;
        chunk.size    = dstSz;
        chunk.isPlain = false;
        {
            std::lock_guard<std::mutex> lk(mOutMtx);
            mOutQueue.push(chunk);
        }
        mOutCv.notify_one();
    }

    bgzf_compressor_free(comp);

    // When this compressor exits, decrement active count and wake writer
    if (mActiveCompressors.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        // Last compressor exiting
        std::lock_guard<std::mutex> lk(mOutMtx);
        mOutDone = true;
        mOutCv.notify_all();
    }
}

void WriterThread::writerWorker() {
    while (true) {
        std::unique_lock<std::mutex> lk(mOutMtx);
        mOutCv.wait(lk, [this]{
            return (!mOutQueue.empty() && mOutQueue.top().seqno == mNextWriteSeqno)
                   || (mOutDone && mOutQueue.empty());
        });

        if (mOutDone && mOutQueue.empty())
            break;

        // Drain all consecutive chunks that are ready
        while (!mOutQueue.empty() && mOutQueue.top().seqno == mNextWriteSeqno) {
            CompressedChunk chunk = mOutQueue.top();
            mOutQueue.pop();
            lk.unlock();

            PROF_START(_t_fwrite);
            fwrite(chunk.data, 1, chunk.size, mWriter1->getFP());
            PROF_END(_t_fwrite, write_ns);
            free(chunk.data);
            mNextWriteSeqno++;
            mBufferLength--;  // now the slot is truly consumed; activates backpressure

            lk.lock();
        }
    }
}

void WriterThread::cleanup() {
    if (mNumCompressors > 0) {
        stopCompressPool();
    }
    deleteWriter();
    if (mBufferLists) {
        for(int t=0; t<mOptions->thread; t++) {
            delete mBufferLists[t];
        }
        delete[] mBufferLists;
        mBufferLists = NULL;
    }
}

void WriterThread::deleteWriter() {
    if(mWriter1 != NULL) {
        delete mWriter1;
        mWriter1 = NULL;
    }
}

void WriterThread::initWriter(string filename1, bool isSTDOUT) {
    deleteWriter();
    mWriter1 = new Writer(mOptions, filename1, mOptions->compression, isSTDOUT);
}

void WriterThread::initBufferLists() {
    mBufferLists = new SingleProducerSingleConsumerList<string*>*[mOptions->thread];
    for(int t=0; t<mOptions->thread; t++) {
        mBufferLists[t] = new SingleProducerSingleConsumerList<string*>();
    }
}
