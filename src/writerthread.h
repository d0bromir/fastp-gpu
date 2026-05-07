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
    WriterThread(Options* opt, string filename, bool isSTDOUT = false);
    ~WriterThread();

    void initWriter(string filename1, bool isSTDOUT = false);
    void initBufferLists();

    void cleanup();

    bool isCompleted();
    void output();
    void input(int tid, string* data);
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
    bool mInputCompleted;
    atomic_long mBufferLength;
    SingleProducerSingleConsumerList<string*>** mBufferLists;
    int mWorkingBufferList;

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