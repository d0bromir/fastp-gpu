#ifndef WRITER_THREAD_H
#define WRITER_THREAD_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "writer.h"
#include "options.h"
#include <atomic>
#include "singleproducersingleconsumerlist.h"

using namespace std;

// A compressed chunk ready to write, with sequence number for ordered output.
struct CompressedChunk {
    uint64_t seqno;
    char*    data;
    size_t   size;
    bool     isPlain; // true when output is not .gz (no compression needed)
};

// Min-heap comparator: lowest seqno at top
struct ChunkOrder {
    bool operator()(const CompressedChunk& a, const CompressedChunk& b) const {
        return a.seqno > b.seqno;
    }
};

class WriterThread{
public:
    // nThreads: number of processor threads feeding this writer (0 = use opt->thread).
    // Pass the adaptive effective thread count so the buffer-list pool and
    // compressor pool scale down for small inputs instead of always using opt->thread.
    WriterThread(Options* opt, string filename, bool isSTDOUT = false, int nThreads = 0);
    ~WriterThread();

    void initWriter(string filename1, bool isSTDOUT = false);

    void cleanup();

    bool isCompleted();
    void output();
    // seqno: the producing pack's global sequence number (ReadPack::seqno),
    // not a thread id — output() drains strictly in seqno order so that
    // packs processed by different worker threads (out of order, under the
    // shared work-stealing queue) still land in the file in original order.
    void input(long seqno, string* data);
    bool setInputCompleted();

    long bufferLength() {return mBufferLength;};
    string getFilename() {return mFilename;}

private:
    void deleteWriter();
    void startCompressPool();
    void stopCompressPool();
    void compressWorker();  // run by each compression thread
    void writerWorker();    // serialises compressed chunks to disk

    // A pending (uncompressed) job
    struct PendingJob {
        uint64_t seqno;
        string*  data;    // owned; freed after compression
    };

private:
    Writer* mWriter1;
    Options* mOptions;
    string mFilename;
    bool mIsGzip;
    bool mIsSTDOUT;

    // ── Input buffer (processor threads → compress pool) ─────────────────
    // Raw (uncompressed) string chunks arrive tagged with their producing
    // pack's seqno, from potentially any worker thread in any order; this
    // min-heap reorders them back into strict pack order before they are
    // handed to the compressor pool / written out, generalizing the same
    // reorder-by-seqno trick already used below for compressed chunks.
    struct RawChunk {
        long    seqno;
        string* data;
    };
    struct RawChunkOrder {
        bool operator()(const RawChunk& a, const RawChunk& b) const {
            return a.seqno > b.seqno;
        }
    };
    bool mInputCompleted;
    atomic_long mBufferLength;
    std::mutex mRawMtx;
    std::condition_variable mRawCv;
    std::priority_queue<RawChunk, std::vector<RawChunk>, RawChunkOrder> mRawQueue;
    long mNextRawSeqno;
    int mNThreads;  // effective thread count (compressor pool sizing)

    // ── Parallel compression pool ─────────────────────────────────────────
    int mNumCompressors;         // number of compression worker threads
    std::vector<std::thread> mCompressThreads;
    std::thread mWriterThread;   // single thread that writes to disk in order

    // job queue: processor threads push PendingJob objects here
    std::mutex              mJobMtx;
    std::condition_variable mJobCv;
    std::queue<PendingJob>  mJobQueue;
    bool                    mJobsDone;  // set when no more jobs will arrive
    atomic<uint64_t>        mNextSeqno; // monotonically increasing job counter

    // output queue: compress workers push CompressedChunk here (out of order)
    std::mutex                                               mOutMtx;
    std::condition_variable                                  mOutCv;
    std::priority_queue<CompressedChunk,
                        std::vector<CompressedChunk>,
                        ChunkOrder>                          mOutQueue;
    uint64_t                mNextWriteSeqno; // next seqno the writer expects
    bool                    mOutDone;        // set when all compress threads exit
    atomic<int>             mActiveCompressors; // count of live compress threads
};

#endif