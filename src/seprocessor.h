#ifndef SE_PROCESSOR_H
#define SE_PROCESSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "read.h"
#include <cstdlib>
#include <condition_variable>
#include <mutex>
#include <thread>
#include "options.h"
#include "threadconfig.h"
#include "filter.h"
#include "umiprocessor.h"
#include "writerthread.h"
#include "duplicate.h"
#include "singleproducersingleconsumerlist.h"
#include "packqueue.h"
#include "contaminant_db.h"
#include "readpool.h"

using namespace std;

typedef struct ReadRepository ReadRepository;

class SingleEndProcessor{
public:
    SingleEndProcessor(Options* opt);
    ~SingleEndProcessor();
    bool process();

private:
    bool processSingleEnd(ReadPack* pack, ThreadConfig* config);
    void readerTask();
    void processorTask(ThreadConfig* config);
    void initConfig(ThreadConfig* config);
    void initOutput();
    void closeOutput();
    void writerTask(WriterThread* config);
    void recycleToPool(int tid, Read* r);

private:
    Options* mOptions;
    int mEffectiveThreads;  // adaptive worker count (≤ mOptions->thread) based on input size
    int mEffectivePackSize; // adaptive pack size (≤ MAX_PACK_SIZE) based on input size / threads
    atomic_bool mReaderFinished;
    atomic_int mFinishedThreads;
    Filter* mFilter;
    ContaminantDB* mContaminantDB;
    UmiProcessor* mUmiProcessor;
    WriterThread* mLeftWriter;
    WriterThread* mFailedWriter;
    Duplicate* mDuplicate;
    // --split path only (fixed pack-index-to-thread striping; ThreadConfig
    // relies on it). See PackQueue's comment in packqueue.h for why the
    // default path uses the shared queue below instead.
    SingleProducerSingleConsumerList<ReadPack*>** mInputLists;
    // Default (non-split) path: shared work queue, load-balanced across
    // worker threads instead of striped round-robin.
    PackQueue<ReadPack*>* mPackQueue;
    size_t mPackReadCounter;
    atomic_long mPackProcessedCounter;
    ReadPool* mReadPool;
};


#endif