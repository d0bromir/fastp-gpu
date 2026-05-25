#include "seprocessor.h"
#include "fastqreader.h"
#include <iostream>
#include <unistd.h>
#include <functional>
#include <thread>
#include <memory.h>
#include <chrono>
#include <sys/stat.h>
#include "util.h"
#include "jsonreporter.h"
#include "htmlreporter.h"
#include "adaptertrimmer.h"
#include "polyx.h"
#include "profiling.h"

// Compute an adaptive pack size so each worker thread sees at least
// MIN_PACKS_PER_THREAD packs regardless of input size, while staying within
// the GPU-optimal ceiling of MAX_PACK_SIZE.  160 bytes/read is a conservative
// underestimate of compressed-FASTQ density, biasing toward smaller packs.
static int effectivePackSize(int nThreads, const string& inFile) {
    static const int    MIN_PACKS_PER_THREAD = 16;
    static const size_t BYTES_PER_READ       = 160; // conservative underestimate
    static const int    MIN_PS               = 512; // floor: still GPU-useful
    if (inFile.empty() || nThreads <= 0) return MAX_PACK_SIZE;
    struct stat st;
    if (stat(inFile.c_str(), &st) != 0) return MAX_PACK_SIZE;
    size_t fileBytes = (size_t)st.st_size;
    if (fileBytes == 0) return MAX_PACK_SIZE;
    long estReads = (long)(fileBytes / BYTES_PER_READ);
    if (estReads < 1) return MAX_PACK_SIZE;
    long targetPacks = (long)nThreads * MIN_PACKS_PER_THREAD;
    long ps = estReads / targetPacks;
    if (ps < MIN_PS)        ps = MIN_PS;
    if (ps > MAX_PACK_SIZE) ps = MAX_PACK_SIZE;
    return (int)ps;
}

// Cap worker thread count so each thread processes at least MIN_PACKS_PER_THREAD
// packs.  Uses the same BYTES_PER_READ as effectivePackSize() for consistency.
static int effectiveWorkerThreads(int nThreads, int packSize, const string& inFile) {
    static const int    MIN_PACKS_PER_THREAD = 16;
    static const size_t BYTES_PER_READ       = 160;  // must match effectivePackSize
    if (inFile.empty() || nThreads <= 1) return nThreads;
    struct stat st;
    if (stat(inFile.c_str(), &st) != 0) return nThreads;   // stdin / error
    size_t fileBytes = (size_t)st.st_size;
    if (fileBytes == 0) return nThreads;
    long estPacks = (long)(fileBytes / BYTES_PER_READ) / packSize;
    if (estPacks < 1) estPacks = 1;
    int maxWorkers = (int)(estPacks / MIN_PACKS_PER_THREAD);
    if (maxWorkers < 1) maxWorkers = 1;
    return std::min(nThreads, maxWorkers);
}

SingleEndProcessor::SingleEndProcessor(Options* opt){
    mOptions = opt;
    mEffectiveThreads  = opt->thread;  // refined to adaptive value in process()
    mEffectivePackSize = MAX_PACK_SIZE; // refined to adaptive value in process()
    mReaderFinished = false;
    mFinishedThreads = 0;
    mFilter = new Filter(opt);
    mUmiProcessor = new UmiProcessor(opt);
    mLeftWriter =  NULL;
    mFailedWriter = NULL;

    mDuplicate = NULL;
    if(mOptions->duplicate.enabled) {
        mDuplicate = new Duplicate(mOptions);
    }

    mPackReadCounter = 0;
    mPackProcessedCounter = 0;

    mReadPool = new ReadPool(mOptions);
}

SingleEndProcessor::~SingleEndProcessor() {
    delete mFilter;
    if(mDuplicate) {
        delete mDuplicate;
        mDuplicate = NULL;
    }
    if(mReadPool) {
        delete mReadPool;
        mReadPool = NULL;
    }
    delete[] mInputLists;
}

void SingleEndProcessor::initOutput() {
    if(!mOptions->failedOut.empty())
        mFailedWriter = new WriterThread(mOptions, mOptions->failedOut);
    if(mOptions->out1.empty() && !mOptions->outputToSTDOUT)
        return;
    mLeftWriter = new WriterThread(mOptions, mOptions->out1, mOptions->outputToSTDOUT, mEffectiveThreads);
}

void SingleEndProcessor::closeOutput() {
    if(mLeftWriter) {
        delete mLeftWriter;
        mLeftWriter = NULL;
    }
    if(mFailedWriter) {
        delete mFailedWriter;
        mFailedWriter = NULL;
    }
}

void SingleEndProcessor::initConfig(ThreadConfig* config) {
    if(mOptions->out1.empty())
        return;

    if(mOptions->split.enabled) {
        config->initWriterForSplit();
    }
}

bool SingleEndProcessor::process(){
    // Adaptive pack size + thread count: computed together so they are
    // self-consistent.  Pack size is derived first (based on requested threads),
    // then the worker count is re-checked using the resulting pack size.
    mEffectivePackSize = effectivePackSize(mOptions->thread, mOptions->in1);
    mEffectiveThreads  = effectiveWorkerThreads(mOptions->thread, mEffectivePackSize, mOptions->in1);
    if (mEffectivePackSize < MAX_PACK_SIZE && mOptions->verbose) {
        loginfo("adaptive pack size: " + to_string(MAX_PACK_SIZE) +
                " → " + to_string(mEffectivePackSize) +
                " (small input, tuning pack granularity)");
    }
    if (mEffectiveThreads < mOptions->thread && mOptions->verbose) {
        loginfo("adaptive threads: " + to_string(mOptions->thread) +
                " → " + to_string(mEffectiveThreads) +
                " (small input, reducing overhead)");
    }

    if(!mOptions->split.enabled)
        initOutput();

    auto timeStageStart = std::chrono::system_clock::now();
    if(mOptions->verbose) {
        string msg = "[fastp " + string(FASTP_VER) + "] Stage: loading";
        loginfo(msg);
    }

    mInputLists = new SingleProducerSingleConsumerList<ReadPack*>*[mEffectiveThreads];

    ThreadConfig** configs = new ThreadConfig*[mEffectiveThreads];
    for(int t=0; t<mEffectiveThreads; t++){
        mInputLists[t] = new SingleProducerSingleConsumerList<ReadPack*>();
        configs[t] = new ThreadConfig(mOptions, t, false);
        configs[t]->setInputList(mInputLists[t]);
        initConfig(configs[t]);
    }

    // Overlap GPU context creation with reader start-up.  ensureGPUInit() is
    // mutex-protected and idempotent; worker threads that call it later either
    // find it already done (fast path) or block briefly until this finishes.
    std::thread gpuInitThread([this]{ mFilter->ensureGPUInit(); });

    std::thread readerThread(std::bind(&SingleEndProcessor::readerTask, this));

    std::thread** threads = new thread*[mEffectiveThreads];
    for(int t=0; t<mEffectiveThreads; t++){
        threads[t] = new std::thread(std::bind(&SingleEndProcessor::processorTask, this, configs[t]));
    }

    std::thread* leftWriterThread = NULL;
    std::thread* failedWriterThread = NULL;
    if(mLeftWriter)
        leftWriterThread = new std::thread(std::bind(&SingleEndProcessor::writerTask, this, mLeftWriter));
    if(mFailedWriter)
        failedWriterThread = new std::thread(std::bind(&SingleEndProcessor::writerTask, this, mFailedWriter));

    readerThread.join();
    gpuInitThread.join(); // already done in all practical cases; join for hygiene

    auto timeStageEnd = std::chrono::system_clock::now();
    auto durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(timeStageEnd - timeStageStart).count();
    if(mOptions->verbose) {
        string msg = "[fastp " + string(FASTP_VER) + "] Stage: loading (" + to_string(durationMs) + " ms)";
        loginfo(msg);
    }

    timeStageStart = std::chrono::system_clock::now();
    if(mOptions->verbose) {
        string msg = "[fastp " + string(FASTP_VER) + "] Stage: filtering";
        loginfo(msg);
    }

    for(int t=0; t<mEffectiveThreads; t++){
        threads[t]->join();
    }

    timeStageEnd = std::chrono::system_clock::now();
    durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(timeStageEnd - timeStageStart).count();
    if(mOptions->verbose) {
        string msg = "[fastp " + string(FASTP_VER) + "] Stage: filtering (" + to_string(durationMs) + " ms)";
        loginfo(msg);
    }

    timeStageStart = std::chrono::system_clock::now();
    if(mOptions->verbose) {
        string msg = "[fastp " + string(FASTP_VER) + "] Stage: writing";
        loginfo(msg);
    }

    if(!mOptions->split.enabled) {
        if(leftWriterThread)
            leftWriterThread->join();
        if(failedWriterThread)
            failedWriterThread->join();
    }

    timeStageEnd = std::chrono::system_clock::now();
    durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(timeStageEnd - timeStageStart).count();
    if(mOptions->verbose) {
        string msg = "[fastp " + string(FASTP_VER) + "] Stage: writing (" + to_string(durationMs) + " ms)";
        loginfo(msg);
    }

    // Print GPU vs CPU filter profiling stats (always in PROFILING builds)
#ifdef FASTP_PROFILING
    mFilter->printProfilingStats();
#else
    if(mOptions->verbose) {
        mFilter->printProfilingStats();
    }
#endif

    if(mOptions->verbose)
        loginfo("start to generate reports\n");

    // merge stats and read filter results
    vector<Stats*> preStats;
    vector<Stats*> postStats;
    vector<FilterResult*> filterResults;
    for(int t=0; t<mEffectiveThreads; t++){
        preStats.push_back(configs[t]->getPreStats1());
        postStats.push_back(configs[t]->getPostStats1());
        filterResults.push_back(configs[t]->getFilterResult());
    }
    Stats* finalPreStats = Stats::merge(preStats);
    Stats* finalPostStats = Stats::merge(postStats);
#ifdef FASTP_PROFILING
    g_profiling.total_reads.store(finalPreStats->getReads());
#endif
    FilterResult* finalFilterResult = FilterResult::merge(filterResults);

    // H-01 read conservation runtime check (IEC 62304 §5.5.2)
    {
        long readsIn = finalPreStats->getReads();
        long accounted = 0;
        long* stats = finalFilterResult->getFilterReadStats();
        for (int i = 0; i < FILTER_RESULT_TYPES; i++)
            accounted += stats[i];
        if (accounted != readsIn)
            error_exit("Read conservation violation: " + to_string(readsIn) +
                " reads entered the pipeline but " + to_string(accounted) +
                " are accounted for in filtering_result buckets."
                " This is a bug — please report it.");
    }

    // read filter results to the first thread's
    for(int t=1; t<mEffectiveThreads; t++){
        preStats.push_back(configs[t]->getPreStats1());
        postStats.push_back(configs[t]->getPostStats1());
    }

    cerr << "Read1 before filtering:"<<endl;
    finalPreStats->print();
    cerr << endl;
    cerr << "Read1 after filtering:"<<endl;
    finalPostStats->print();

    cerr << endl;
    cerr << "Filtering result:"<<endl;
    finalFilterResult->print();

    double dupRate = 0.0;
    if(mOptions->duplicate.enabled) {
        dupRate = mDuplicate->getDupRate();
        cerr << endl;
        cerr << "Duplication rate (may be overestimated since this is SE data): " << dupRate * 100.0 << "%" << endl;
    }

    // make JSON report
    JsonReporter jr(mOptions);
    jr.setDup(dupRate);
    jr.report(finalFilterResult, finalPreStats, finalPostStats);

    // make HTML report
    HtmlReporter hr(mOptions);
    hr.setDup(dupRate);
    hr.report(finalFilterResult, finalPreStats, finalPostStats);

    // clean up
    for(int t=0; t<mEffectiveThreads; t++){
        delete threads[t];
        threads[t] = NULL;
        delete configs[t];
        configs[t] = NULL;
    }

    delete finalPreStats;
    delete finalPostStats;
    delete finalFilterResult;

    delete[] threads;
    delete[] configs;

    if(leftWriterThread)
        delete leftWriterThread;
    if(failedWriterThread)
        delete failedWriterThread;

    if(!mOptions->split.enabled)
        closeOutput();

    return true;
}

void SingleEndProcessor::recycleToPool(int tid, Read* r) {
    // failed to recycle, then delete it
    if(!mReadPool->input(tid, r))
        delete r;
}

bool SingleEndProcessor::processSingleEnd(ReadPack* pack, ThreadConfig* config){
    string* outstr = new string();
    string* failedOut = new string();
    int tid = config->getThreadId();

    int readPassed = 0;
    
    GPU_FPRINTF("[GPU] processSingleEnd called with pack count: %d\n", pack->count);
    
    // Collect reads for batch GPU processing
    vector<Read*> readsForGPU;
    vector<Read*> originalReads;
    vector<bool> isDedupOut;
    vector<bool> isAdapterDimerVec;
    
    // First pass: prepare reads (trimming, adapter removal, etc.)
    for(int p=0;p<pack->count;p++){

        // original read1
        Read* or1 = pack->data[p];

        // handling the duplication profiling
        bool dedupOut = false;
        if(mDuplicate) {
            bool isDup = mDuplicate->checkRead(or1);
            if(mOptions->duplicate.dedup && isDup)
                dedupOut = true;
        }

        // filter by index
        if(mOptions->indexFilter.enabled && mFilter->filterByIndex(or1)) {
            recycleToPool(tid, or1);
            continue;
        }

        // fix MGI
        if(mOptions->fixMGI) {
            or1->fixMGI();
        }

        // umi processing
        if(mOptions->umi.enabled)
            mUmiProcessor->process(or1);

        // Pre-filter stats on raw read BEFORE any trimming modifies it in-place
        PROF_START(_t_stat);
        config->getPreStats1()->statReadBasic(or1);
        PROF_END(_t_stat, cpu_statread_ns);

        int frontTrimmed = 0;
        // trim in head and tail, and apply quality cut in sliding window
        // NOTE: Quality trimming and PolyG trimming now done on GPU for efficiency
        PROF_START(_t_trim);
        Read* r1 = mFilter->trimAndCut(or1, mOptions->trim.front1, mOptions->trim.tail1, frontTrimmed);

        // GPU will compute PolyG and quality trimming, so skip CPU versions
        // The GPU results will be applied during filtering phase

        bool isAdapterDimer = false;
        if(r1 != NULL && mOptions->adapter.enabled){
            bool trimmed = false;
            if(mOptions->adapter.hasSeqR1)
                trimmed = AdapterTrimmer::trimBySequence(r1, config->getFilterResult(), mOptions->adapter.sequence, false);
            if(mOptions->adapter.hasFasta) {
                trimmed |= AdapterTrimmer::trimByMultiSequences(r1, config->getFilterResult(), mOptions->adapter.seqsInFasta, false);
            }

            if(trimmed )
                config->getFilterResult()->incTrimmedAdapterRead(1);

            // Check for adapter dimer: read shorter than threshold after adapter trimming
            // AND adapter was detected (requires evidence)
            if(r1 != NULL && trimmed && r1->length() <= mOptions->adapter.dimerMaxLen) {
                isAdapterDimer = true;
            }
        }

        // PolyX trimming skipped - will be computed on GPU via PolyG detection
        // PolyG trimming skipped - will be computed on GPU

        if(r1 != NULL) {
            if( mOptions->trim.maxLen1 > 0 && mOptions->trim.maxLen1 < r1->length())
                r1->resize(mOptions->trim.maxLen1);
        }
        PROF_END(_t_trim, cpu_trim_adapter_ns);

        // Collect for GPU batch processing (NO speculative statRead here —
        // the GPU kernel computes post-filter stats for passing reads directly)
        readsForGPU.push_back(r1);
        originalReads.push_back(or1);
        isDedupOut.push_back(dedupOut);
        isAdapterDimerVec.push_back(isAdapterDimer);
        
        if (p < 3) GPU_FPRINTF("[GPU] Read %d: r1=%p or1=%p\n", p, r1, or1);
    }
    
    GPU_FPRINTF("[GPU] Total reads collected for GPU batch: %zu\n", readsForGPU.size());

    // Pre-filter stats already computed inline (before trimAndCut) for correctness.

    // GPU filtering + post-filter stats
    vector<int> filterResults;
    if(!readsForGPU.empty()) {
        GPU_FPRINTF("[GPU] processSingleEnd batch: %zu reads\n", readsForGPU.size());
        PROF_START(_t_gpu);
        mFilter->filterBatchGPUWithStats(readsForGPU, filterResults,
                                          config->getPostStats1());
        PROF_END(_t_gpu, cpu_filter_ns);
    }

    // Process filter results
    if(!filterResults.empty()) {
        for(size_t i = 0; i < readsForGPU.size(); i++) {
            int result = filterResults[i];
            Read* r1 = readsForGPU[i];
            Read* or1 = originalReads[i];
            bool dedupOut = isDedupOut[i];
            bool isAdapterDimer = isAdapterDimerVec[i];
            
            if(isAdapterDimer) {
                // GPU may have counted stats for this read — subtract if it passed GPU filter
                if(result == PASS_FILTER && r1 != NULL) {
                    PROF_START(_t_unstat);
                    config->getPostStats1()->unstatRead(r1);
                    PROF_END(_t_unstat, cpu_unstatread_ns);
                }
                result = FAIL_ADAPTER_DIMER;
            }

            config->addFilterResult(result, 1);

            if(!dedupOut) {
                if( r1 != NULL &&  result == PASS_FILTER) {
                    PROF_START(_t_out);
                    r1->appendToString(outstr);
                    PROF_END(_t_out, cpu_output_ns);
                    readPassed++;
                } else {
                    if(mFailedWriter)
                        or1->appendToStringWithTag(failedOut, FAILED_TYPES[result]);
                }
            } else {
                // GPU counted stats for this read (it passed filter), but it's a dedup → subtract
                if(r1 != NULL && result == PASS_FILTER) {
                    PROF_START(_t_unstat2);
                    config->getPostStats1()->unstatRead(r1);
                    PROF_END(_t_unstat2, cpu_unstatread_ns);
                }
            }

            recycleToPool(tid, or1);
            // if no trimming applied, r1 should be identical to or1
            if(r1 != or1 && r1 != NULL)
                recycleToPool(tid, r1);
        }
    } // end filter results

    if(mOptions->split.enabled) {
        // split output by each worker thread
        if(!mOptions->out1.empty())
            config->getWriter1()->writeString(outstr);
    }

    if(mLeftWriter) {
        mLeftWriter->input(tid, outstr);
        outstr = NULL;
    }
    if(mFailedWriter) {
        // write failed data
        mFailedWriter->input(tid, failedOut);
        failedOut = NULL;
    }

    if(mOptions->split.byFileLines)
        config->markProcessed(readPassed);
    else
        config->markProcessed(pack->count);

    if(outstr)
        delete outstr;
    if(failedOut)
        delete failedOut;

    delete pack->data;
    delete pack;

    mPackProcessedCounter++;

    return true;
}

void SingleEndProcessor::readerTask()
{
    if(mOptions->verbose)
        loginfo("start to load data");
    long lastReported = 0;
    int slept = 0;
    long readNum = 0;
    bool splitSizeReEvaluated = false;
    Read** data = new Read*[mEffectivePackSize];
    memset(data, 0, sizeof(Read*)*mEffectivePackSize);
    FastqReader reader(mOptions->in1, true, mOptions->phred64, mOptions->useGDS);
    reader.setReadPool(mReadPool);
    // Async double-buffered decompression overlaps I/O with compute, but
    // at T=1 the extra concurrent fread() hurts throughput on slow storage.
    if (mOptions->thread > 1) reader.enableAsyncDecomp();
    int count=0;
    bool needToBreak = false;
    while(true){
        Read* read = reader.read();
        if(!read || needToBreak){
            // the last pack
            ReadPack* pack = new ReadPack;
            pack->data = data;
            pack->count = count;
            mInputLists[mPackReadCounter % mEffectiveThreads]->produce(pack);
            mPackReadCounter++;
            data = NULL;
            if(read) {
                delete read;
                read = NULL;
            }
            break;
        }
        data[count] = read;
        count++;
        // configured to process only first N reads
        if(mOptions->readsToProcess >0 && count + readNum >= mOptions->readsToProcess) {
            needToBreak = true;
        }
        if(mOptions->verbose && count + readNum >= lastReported + 1000000) {
            lastReported = count + readNum;
            string msg = "loaded " + to_string((lastReported/1000000)) + "M reads";
            loginfo(msg);
        }
        // a full pack
        if(count == mEffectivePackSize || needToBreak){
            ReadPack* pack = new ReadPack;
            pack->data = data;
            pack->count = count;
            mInputLists[mPackReadCounter % mEffectiveThreads]->produce(pack);
            mPackReadCounter++;
            //re-initialize data for next pack
            data = new Read*[mEffectivePackSize];
            memset(data, 0, sizeof(Read*)*mEffectivePackSize);
            // if the processor is far behind this reader, sleep and wait to limit memory usage
            while( mPackReadCounter - mPackProcessedCounter > PACK_IN_MEM_LIMIT){
                //cerr<<"sleep"<<endl;
                slept++;
                usleep(100);
            }
            readNum += count;
            // if the writer threads are far behind this reader, sleep and wait
            // check this only when necessary
            if(readNum % (mEffectivePackSize * PACK_IN_MEM_LIMIT) == 0 && mLeftWriter) {
                while(mLeftWriter->bufferLength() > PACK_IN_MEM_LIMIT) {
                    slept++;
                    usleep(1000);
                }
            }
            // reset count to 0
            count = 0;
            // re-evaluate split size
            // TODO: following codes are commented since it may cause threading related conflicts in some systems
            /*if(mOptions->split.needEvaluation && !splitSizeReEvaluated && readNum >= mOptions->split.size) {
                splitSizeReEvaluated = true;
                // greater than the initial evaluation
                if(readNum >= 1024*1024) {
                    size_t bytesRead;
                    size_t bytesTotal;
                    reader.getBytes(bytesRead, bytesTotal);
                    mOptions->split.size *=  (double)bytesTotal / ((double)bytesRead * (double) mOptions->split.number);
                    if(mOptions->split.size <= 0)
                        mOptions->split.size = 1;
                }
            }*/
        }
    }

    for(int t=0; t<mEffectiveThreads; t++)
        mInputLists[t]->setProducerFinished();

    //std::unique_lock<std::mutex> lock(mRepo.readCounterMtx);
    mReaderFinished = true;
    if(mOptions->verbose) {
        loginfo("Loading completed with " + to_string(mPackReadCounter) + " packs");
    }
    //lock.unlock();

    // if the last data initialized is not used, free it
    if(data != NULL)
        delete[] data;
}

void SingleEndProcessor::processorTask(ThreadConfig* config)
{
    SingleProducerSingleConsumerList<ReadPack*>* input = config->getLeftInput();
    while(true) {
        if(config->canBeStopped()){
            break;
        }
        while(input->canBeConsumed()) {
            ReadPack* data = input->consume();
            processSingleEnd(data, config);
        }
        if(input->isProducerFinished()) {
            if(!input->canBeConsumed()) {
                if(mOptions->verbose) {
                    string msg = "thread " + to_string(config->getThreadId() + 1) + " data processing completed";
                    loginfo(msg);
                }
                break;
            }
        } else {
            usleep(100);
        }
    }
    input->setConsumerFinished();

    mFinishedThreads++;
    if(mFinishedThreads == mEffectiveThreads) {
        if(mLeftWriter)
            mLeftWriter->setInputCompleted();
        if(mFailedWriter)
            mFailedWriter->setInputCompleted();
    }

    if(mOptions->verbose) {
        string msg = "thread " + to_string(config->getThreadId() + 1) + " finished";
        loginfo(msg);
    }
}

void SingleEndProcessor::writerTask(WriterThread* config)
{
    while(true) {
        if(config->isCompleted()){
            // last check for possible threading related issue
            config->output();
            break;
        }
        config->output();
    }

    if(mOptions->verbose) {
        string msg = config->getFilename() + " writer finished";
        loginfo(msg);
    }
}
