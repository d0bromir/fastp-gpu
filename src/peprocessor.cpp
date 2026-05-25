#include "peprocessor.h"
#include "fastqreader.h"
#include <iostream>
#include <unistd.h>
#include <functional>
#include <thread>
#include <vector>
#include <memory.h>
#include <chrono>
#include <sys/stat.h>
#include "util.h"
#include "adaptertrimmer.h"
#include "basecorrector.h"
#include "jsonreporter.h"
#include "htmlreporter.h"
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

PairEndProcessor::PairEndProcessor(Options* opt){
    mOptions = opt;
    mEffectiveThreads  = opt->thread;  // refined to adaptive value in process()
    mEffectivePackSize = MAX_PACK_SIZE; // refined to adaptive value in process()
    mLeftReaderFinished = false;
    mRightReaderFinished = false;
    mFinishedThreads = 0;
    mFilter = new Filter(opt);
    mUmiProcessor = new UmiProcessor(opt);

    int isizeBufLen = mOptions->insertSizeMax + 1;
    mInsertSizeHist = new atomic_long[isizeBufLen];
    memset(mInsertSizeHist, 0, sizeof(atomic_long)*isizeBufLen);
    mLeftWriter =  NULL;
    mRightWriter = NULL;
    mUnpairedLeftWriter =  NULL;
    mUnpairedRightWriter = NULL;
    mMergedWriter = NULL;
    mFailedWriter = NULL;
    mOverlappedWriter = NULL;
    shouldStopReading = false;

    mDuplicate = NULL;
    if(mOptions->duplicate.enabled) {
        mDuplicate = new Duplicate(mOptions);
    }

    mLeftPackReadCounter = 0;
    mRightPackReadCounter = 0;
    mPackProcessedCounter = 0;

    mLeftReadPool = new ReadPool(mOptions);
    mRightReadPool = new ReadPool(mOptions);
}

PairEndProcessor::~PairEndProcessor() {
    delete mInsertSizeHist;
    if(mDuplicate) {
        delete mDuplicate;
        mDuplicate = NULL;
    }
    if(mLeftReadPool) {
        delete mLeftReadPool;
        mLeftReadPool = NULL;
    }
    if(mRightReadPool) {
        delete mRightReadPool;
        mRightReadPool = NULL;
    }
    delete[] mLeftInputLists;
    delete[] mRightInputLists;
}

void PairEndProcessor::initOutput() {
    if(!mOptions->unpaired1.empty())
        mUnpairedLeftWriter = new WriterThread(mOptions, mOptions->unpaired1);

    if(!mOptions->unpaired2.empty() && mOptions->unpaired2 != mOptions->unpaired1)
        mUnpairedRightWriter = new WriterThread(mOptions, mOptions->unpaired2);

    if(mOptions->merge.enabled) {
        if(!mOptions->merge.out.empty())
            mMergedWriter = new WriterThread(mOptions, mOptions->merge.out);
    }

    if(!mOptions->failedOut.empty())
        mFailedWriter = new WriterThread(mOptions, mOptions->failedOut);

    if(!mOptions->overlappedOut.empty())
        mOverlappedWriter = new WriterThread(mOptions, mOptions->overlappedOut);

    if(mOptions->out1.empty() && !mOptions->outputToSTDOUT)
        return;
    
    mLeftWriter = new WriterThread(mOptions, mOptions->out1, mOptions->outputToSTDOUT, mEffectiveThreads);
    if(!mOptions->out2.empty())
        mRightWriter = new WriterThread(mOptions, mOptions->out2, false, mEffectiveThreads);
}

void PairEndProcessor::closeOutput() {
    if(mLeftWriter) {
        delete mLeftWriter;
        mLeftWriter = NULL;
    }
    if(mRightWriter) {
        delete mRightWriter;
        mRightWriter = NULL;
    }
    if(mMergedWriter) {
        delete mMergedWriter;
        mMergedWriter = NULL;
    }
    if(mFailedWriter) {
        delete mFailedWriter;
        mFailedWriter = NULL;
    }
    if(mOverlappedWriter) {
        delete mOverlappedWriter;
        mOverlappedWriter = NULL;
    }
    if(mUnpairedLeftWriter) {
        delete mUnpairedLeftWriter;
        mUnpairedLeftWriter = NULL;
    }
    if(mUnpairedRightWriter) {
        delete mUnpairedRightWriter;
        mUnpairedRightWriter = NULL;
    }
}

void PairEndProcessor::initConfig(ThreadConfig* config) {
    if(mOptions->out1.empty())
        return;
    if(mOptions->split.enabled) {
        config->initWriterForSplit();
    }
}


bool PairEndProcessor::process(){
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

    std::thread readerLeft;
    std::thread readerRight;
    std::thread readerInterveleaved;
    bool hasReaderLeft = false;
    bool hasReaderRight = false;
    bool hasReaderInterleaved = false;

    mLeftInputLists = new SingleProducerSingleConsumerList<ReadPack*>*[mEffectiveThreads];
    mRightInputLists = new SingleProducerSingleConsumerList<ReadPack*>*[mEffectiveThreads];

    ThreadConfig** configs = new ThreadConfig*[mEffectiveThreads];
    for(int t=0; t<mEffectiveThreads; t++){
        mLeftInputLists[t] = new SingleProducerSingleConsumerList<ReadPack*>();
        mRightInputLists[t] = new SingleProducerSingleConsumerList<ReadPack*>();
        configs[t] = new ThreadConfig(mOptions, t, true);
        configs[t]->setInputListPair(mLeftInputLists[t], mRightInputLists[t]);
        initConfig(configs[t]);
    }

    // Overlap GPU context creation with reader start-up (see seprocessor.cpp
    // for rationale).  ensureGPUInit() is mutex-protected and idempotent.
    std::thread gpuInitThread([this]{ mFilter->ensureGPUInit(); });

    if(mOptions->interleavedInput) {
        readerInterveleaved = std::thread(&PairEndProcessor::interleavedReaderTask, this);
        hasReaderInterleaved = true;
    } else {
        readerLeft = std::thread(&PairEndProcessor::readerTask, this, true);
        readerRight = std::thread(&PairEndProcessor::readerTask, this, false);
        hasReaderLeft = true;
        hasReaderRight = true;
    }

    std::vector<std::thread> threads;
    threads.reserve(mEffectiveThreads);
    for(int t=0; t<mEffectiveThreads; t++){
        threads.emplace_back(&PairEndProcessor::processorTask, this, configs[t]);
    }

    std::thread leftWriterThread;
    std::thread rightWriterThread;
    std::thread unpairedLeftWriterThread;
    std::thread unpairedRightWriterThread;
    std::thread mergedWriterThread;
    std::thread failedWriterThread;
    std::thread overlappedWriterThread;
    bool hasLeftWriterThread = false;
    bool hasRightWriterThread = false;
    bool hasUnpairedLeftWriterThread = false;
    bool hasUnpairedRightWriterThread = false;
    bool hasMergedWriterThread = false;
    bool hasFailedWriterThread = false;
    bool hasOverlappedWriterThread = false;
    if(mLeftWriter) {
        leftWriterThread = std::thread(&PairEndProcessor::writerTask, this, mLeftWriter);
        hasLeftWriterThread = true;
    }
    if(mRightWriter) {
        rightWriterThread = std::thread(&PairEndProcessor::writerTask, this, mRightWriter);
        hasRightWriterThread = true;
    }
    if(mUnpairedLeftWriter) {
        unpairedLeftWriterThread = std::thread(&PairEndProcessor::writerTask, this, mUnpairedLeftWriter);
        hasUnpairedLeftWriterThread = true;
    }
    if(mUnpairedRightWriter) {
        unpairedRightWriterThread = std::thread(&PairEndProcessor::writerTask, this, mUnpairedRightWriter);
        hasUnpairedRightWriterThread = true;
    }
    if(mMergedWriter) {
        mergedWriterThread = std::thread(&PairEndProcessor::writerTask, this, mMergedWriter);
        hasMergedWriterThread = true;
    }
    if(mFailedWriter) {
        failedWriterThread = std::thread(&PairEndProcessor::writerTask, this, mFailedWriter);
        hasFailedWriterThread = true;
    }
    if(mOverlappedWriter) {
        overlappedWriterThread = std::thread(&PairEndProcessor::writerTask, this, mOverlappedWriter);
        hasOverlappedWriterThread = true;
    }

    if(hasReaderInterleaved) {
        if(readerInterveleaved.joinable()) readerInterveleaved.join();
    } else {
        if(hasReaderLeft && readerLeft.joinable()) readerLeft.join();
        if(hasReaderRight && readerRight.joinable()) readerRight.join();
    }
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

    for(auto &th : threads){ if(th.joinable()) th.join(); }

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
        if(hasLeftWriterThread && leftWriterThread.joinable()) leftWriterThread.join();
        if(hasRightWriterThread && rightWriterThread.joinable()) rightWriterThread.join();
        if(hasUnpairedLeftWriterThread && unpairedLeftWriterThread.joinable()) unpairedLeftWriterThread.join();
        if(hasUnpairedRightWriterThread && unpairedRightWriterThread.joinable()) unpairedRightWriterThread.join();
        if(hasMergedWriterThread && mergedWriterThread.joinable()) mergedWriterThread.join();
        if(hasFailedWriterThread && failedWriterThread.joinable()) failedWriterThread.join();
        if(hasOverlappedWriterThread && overlappedWriterThread.joinable()) overlappedWriterThread.join();
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

    // merge stats and filter results
    vector<Stats*> preStats1;
    vector<Stats*> postStats1;
    vector<Stats*> preStats2;
    vector<Stats*> postStats2;
    vector<FilterResult*> filterResults;
    for(int t=0; t<mEffectiveThreads; t++){
        preStats1.push_back(configs[t]->getPreStats1());
        postStats1.push_back(configs[t]->getPostStats1());
        preStats2.push_back(configs[t]->getPreStats2());
        postStats2.push_back(configs[t]->getPostStats2());
        filterResults.push_back(configs[t]->getFilterResult());
    }
    Stats* finalPreStats1 = Stats::merge(preStats1);
    Stats* finalPostStats1 = Stats::merge(postStats1);
    Stats* finalPreStats2 = Stats::merge(preStats2);
    Stats* finalPostStats2 = Stats::merge(postStats2);
    FilterResult* finalFilterResult = FilterResult::merge(filterResults);

    // H-01 read conservation runtime check (IEC 62304 §5.5.2): every read must
    // land in exactly one FilterResult bucket; a mismatch indicates a bug in the
    // filter or writer pipeline that silently dropped or double-counted reads.
    {
        long readsIn = finalPreStats1->getReads() + finalPreStats2->getReads();
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

#ifdef FASTP_PROFILING
    g_profiling.total_reads.store(finalPreStats1->getReads());
#endif

    cerr << "Read1 before filtering:"<<endl;
    finalPreStats1->print();
    cerr << endl;
    cerr << "Read2 before filtering:"<<endl;
    finalPreStats2->print();
    cerr << endl;
    if(!mOptions->merge.enabled) {
        cerr << "Read1 after filtering:"<<endl;
        finalPostStats1->print();
        cerr << endl;
        cerr << "Read2 after filtering:"<<endl;
        finalPostStats2->print();
    } else {
        cerr << "Merged and filtered:"<<endl;
        finalPostStats1->print();
    }

    cerr << endl;
    cerr << "Filtering result:"<<endl;
    finalFilterResult->print();

    double dupRate = 0.0;
    if(mOptions->duplicate.enabled) {
        dupRate = mDuplicate->getDupRate();
        cerr << endl;
        cerr << "Duplication rate: " << dupRate * 100.0 << "%" << endl;
    }

    // insert size distribution
    int peakInsertSize = getPeakInsertSize();
    cerr << endl;
    cerr << "Insert size peak (evaluated by paired-end reads): " << peakInsertSize << endl;

    if(mOptions->merge.enabled) {
        cerr << endl;
        cerr << "Read pairs merged: " << finalFilterResult->mMergedPairs << endl;
        if(finalPostStats1->getReads() > 0) {
            double postMergedPercent = 100.0 * finalFilterResult->mMergedPairs / finalPostStats1->getReads();
            double preMergedPercent = 100.0 * finalFilterResult->mMergedPairs / finalPreStats1->getReads();
            cerr << "% of original read pairs: " << preMergedPercent << "%" << endl;
            cerr << "% in reads after filtering: " << postMergedPercent << "%" << endl;
        }
        cerr << endl;
    }

    // make JSON report
    JsonReporter jr(mOptions);
    jr.setDup(dupRate);
    jr.setInsertHist(mInsertSizeHist, peakInsertSize);
    jr.report(finalFilterResult, finalPreStats1, finalPostStats1, finalPreStats2, finalPostStats2);

    // make HTML report
    HtmlReporter hr(mOptions);
    hr.setDup(dupRate);
    hr.setInsertHist(mInsertSizeHist, peakInsertSize);
    hr.report(finalFilterResult, finalPreStats1, finalPostStats1, finalPreStats2, finalPostStats2);

    // clean up
    for(int t=0; t<mEffectiveThreads; t++){
        delete configs[t];
        configs[t] = NULL;
    }

    delete finalPreStats1;
    delete finalPostStats1;
    delete finalPreStats2;
    delete finalPostStats2;
    delete finalFilterResult;

    delete[] configs;

    // thread objects are RAII-managed (joined above)

    if(!mOptions->split.enabled)
        closeOutput();

    return true;
}

int PairEndProcessor::getPeakInsertSize() {
    int peak = 0;
    long maxCount = -1;
    for(int i=0; i<mOptions->insertSizeMax; i++) {
        if(mInsertSizeHist[i] > maxCount) {
            peak = i;
            maxCount = mInsertSizeHist[i];
        }
    }
    return peak;
}

void PairEndProcessor::recycleToPool1(int tid, Read* r) {
    // failed to recycle, then delete it
    if(!mLeftReadPool->input(tid, r))
        delete r;
}

void PairEndProcessor::recycleToPool2(int tid, Read* r) {
    // failed to recycle, then delete it
    if(!mRightReadPool->input(tid, r))
        delete r;
}

bool PairEndProcessor::processPairEnd(ReadPack* leftPack, ReadPack* rightPack, ThreadConfig* config){
    if(leftPack->count != rightPack->count) {
        error_exit("Paired-end read count mismatch in pack " +
            to_string(mPackProcessedCounter) +
            ": Read1 has " + to_string(leftPack->count) +
            " reads, Read2 has " + to_string(rightPack->count) +
            " reads. Input files are not properly paired.");
    }
    int tid = config->getThreadId();

    string* outstr1 = new string();
    string* outstr2 = new string();
    string* unpairedOut1 = new string();
    string* unpairedOut2 = new string();
    string* singleOutput = new string();
    string* mergedOutput = new string();
    string* failedOut = new string();
    string* overlappedOut = new string();

    int readPassed = 0;
    int mergedCount = 0;
    
    // Pre-filtering stage: trimming and UMI processing
    vector<Read*> trimmedReads1;
    vector<Read*> trimmedReads2;
    vector<Read*> originalReads1;
    vector<Read*> originalReads2;
    vector<bool> isDedupOutVec;
    vector<bool> isAdapterDimerVec;
    
    GPU_FPRINTF("[GPU] processPairEnd called with pack count: %d\n", leftPack->count);
    
    // First pass: trim and preprocess reads
    for(int p=0;p<leftPack->count && p<rightPack->count;p++){
        Read* or1 = leftPack->data[p];
        Read* or2 = rightPack->data[p];

        // handling the duplication profiling
        bool dedupOut = false;
        if(mDuplicate) {
            bool isDup = mDuplicate->checkPair(or1, or2);
            if(mOptions->duplicate.dedup && isDup)
                dedupOut = true;
        }

        // filter by index
        if(mOptions->indexFilter.enabled && mFilter->filterByIndex(or1, or2)) {
            recycleToPool1(tid, or1);
            recycleToPool2(tid, or2);
            continue;
        }

        // fix MGI
        if(mOptions->fixMGI) {
            or1->fixMGI();
            or2->fixMGI();
        }
        // umi processing
        if(mOptions->umi.enabled)
            mUmiProcessor->process(or1, or2);

        // Pre-filter stats on raw reads BEFORE any trimming modifies them in-place
        PROF_START(_t_stat);
        config->getPreStats1()->statReadBasic(or1);
        config->getPreStats2()->statReadBasic(or2);
        PROF_END(_t_stat, cpu_statread_ns);

        // trim in head and tail, and apply quality cut in sliding window
        int frontTrimmed1 = 0;
        int frontTrimmed2 = 0;
        PROF_START(_t_trim);
        Read* r1 = mFilter->trimAndCutGPU(or1, mOptions->trim.front1, mOptions->trim.tail1, frontTrimmed1);
        Read* r2 = mFilter->trimAndCutGPU(or2, mOptions->trim.front2, mOptions->trim.tail2, frontTrimmed2);

        if(r1 != NULL && r2!=NULL) {
            if(mOptions->polyGTrim.enabled)
                mFilter->trimPolyGGPU(r1, r2);
        }
        
        bool isizeEvaluated = false;
        bool isAdapterDimer = false;
        if(r1 != NULL && r2!=NULL && (mOptions->adapter.enabled || mOptions->correction.enabled)){
            OverlapResult ov = OverlapAnalysis::analyze(r1, r2, mOptions->overlapDiffLimit, mOptions->overlapRequire, mOptions->overlapDiffPercentLimit/100.0, mOptions->adapter.allowGapOverlapTrimming);
            // we only use thread 0 to evaluae ISIZE
            if(config->getThreadId() == 0) {
                statInsertSize(r1, r2, ov, frontTrimmed1, frontTrimmed2);
                isizeEvaluated = true;
            }
            if(mOptions->correction.enabled && !ov.hasGap) {
                // no gap allowed for overlap correction
                BaseCorrector::correctByOverlapAnalysis(r1, r2, config->getFilterResult(), ov);
            }
            if(mOptions->adapter.enabled) {
                bool trimmed = AdapterTrimmer::trimByOverlapAnalysis(r1, r2, config->getFilterResult(), ov, frontTrimmed1, frontTrimmed2);
                bool trimmed1 = trimmed;
                bool trimmed2 = trimmed;
                if(!trimmed){
                    if(mOptions->adapter.hasSeqR1)
                        trimmed1 = AdapterTrimmer::trimBySequence(r1, config->getFilterResult(), mOptions->adapter.sequence, false);
                    if(mOptions->adapter.hasSeqR2)
                        trimmed2 = AdapterTrimmer::trimBySequence(r2, config->getFilterResult(), mOptions->adapter.sequenceR2, true);
                }
                if(mOptions->adapter.hasFasta) {
                    trimmed1 |= AdapterTrimmer::trimByMultiSequences(r1, config->getFilterResult(), mOptions->adapter.seqsInFasta, false);
                    trimmed2 |= AdapterTrimmer::trimByMultiSequences(r2, config->getFilterResult(), mOptions->adapter.seqsInFasta, true);
                }

                if(trimmed1 )
                    config->getFilterResult()->incTrimmedAdapterRead(1);
                if(trimmed2 )
                    config->getFilterResult()->incTrimmedAdapterRead(1);


                // Check for adapter dimer: both reads shorter than threshold after adapter trimming
                // AND adapters were detected in at least one of the reads (requires evidence)
                if(r1 != NULL && r2 != NULL && (trimmed1 || trimmed2) &&
                   r1->length() <= mOptions->adapter.dimerMaxLen &&
                   r2->length() <= mOptions->adapter.dimerMaxLen) {
                    isAdapterDimer = true;
                }
            }
        }

        if(r1 != NULL && r2!=NULL && mOverlappedWriter) {
            OverlapResult ov = OverlapAnalysis::analyze(r1, r2, mOptions->overlapDiffLimit, mOptions->overlapRequire, 0);
            if(ov.overlapped) {
                Read* overlappedRead = new Read(new string(*r1->mName), new string(r1->mSeq->substr(max(0,ov.offset)), ov.overlap_len), new string(*r1->mStrand), new string(r1->mQuality->substr(max(0,ov.offset)), ov.overlap_len));
                overlappedRead->appendToString(overlappedOut);
                recycleToPool1(tid, overlappedRead);
            }
        }

        if(config->getThreadId() == 0 && !isizeEvaluated && r1 != NULL && r2!=NULL) {
            OverlapResult ov = OverlapAnalysis::analyze(r1, r2, mOptions->overlapDiffLimit, mOptions->overlapRequire, mOptions->overlapDiffPercentLimit/100.0);
            statInsertSize(r1, r2, ov, frontTrimmed1, frontTrimmed2);
            isizeEvaluated = true;
        }

        if(r1 != NULL && r2!=NULL) {
            if(mOptions->polyXTrim.enabled)
                PolyX::trimPolyX(r1, r2, config->getFilterResult(), mOptions->polyXTrim.minLen);
        }

        if(r1 != NULL && r2!=NULL) {
            if( mOptions->trim.maxLen1 > 0 && mOptions->trim.maxLen1 < r1->length())
                r1->resize(mOptions->trim.maxLen1);
            if( mOptions->trim.maxLen2 > 0 && mOptions->trim.maxLen2 < r2->length())
                r2->resize(mOptions->trim.maxLen2);
        }
        PROF_END(_t_trim, cpu_trim_adapter_ns);
        
        // Collect for GPU batch processing (NO speculative statRead here —
        // GPU kernel computes post-filter stats for passing reads directly)
        trimmedReads1.push_back(r1);
        trimmedReads2.push_back(r2);
        originalReads1.push_back(or1);
        originalReads2.push_back(or2);
        isDedupOutVec.push_back(dedupOut);
        isAdapterDimerVec.push_back(isAdapterDimer);
    }
    
    int pack_size = trimmedReads1.size();
    GPU_FPRINTF("[GPU] processPairEnd batch: %d read pairs\n", pack_size);

    // Pre-filter stats already computed inline (before trimAndCut) for correctness.

    // GPU filtering + post-filter stats
    vector<int> filterResults1, filterResults2;
    if(pack_size > 0) {
        PROF_START(_t_gpu);
        if(!mOptions->merge.enabled) {
            mFilter->filterBatchGPUWithStats(trimmedReads1, trimmedReads2,
                                              filterResults1, filterResults2,
                                              config->getPostStats1(), config->getPostStats2());
        } else {
            mFilter->filterBatchGPU(trimmedReads1, trimmedReads2, filterResults1, filterResults2);
        }
        PROF_END(_t_gpu, cpu_filter_ns);
    }

    // Third pass: apply results and output
    PROF_START(_t_out_pe);
    for(int p=0; p<pack_size; p++){
        Read* or1 = originalReads1[p];
        Read* or2 = originalReads2[p];
        Read* r1 = trimmedReads1[p];
        Read* r2 = trimmedReads2[p];
        
        bool dedupOut = isDedupOutVec[p];
        bool isAdapterDimer = isAdapterDimerVec[p];

        Read* merged = NULL;
        // merging mode
        bool mergeProcessed = false;
        if(mOptions->merge.enabled && r1 && r2) {
            OverlapResult ov = OverlapAnalysis::analyze(r1, r2, mOptions->overlapDiffLimit, mOptions->overlapRequire, mOptions->overlapDiffPercentLimit/100.0);
            if(ov.overlapped) {
                merged = OverlapAnalysis::merge(r1, r2, ov);
                int result = mFilter->passFilter(merged);
                config->addFilterResult(result, 2);
                if(result == PASS_FILTER) {
                    merged->appendToString(mergedOutput);
                    config->getPostStats1()->statRead(merged);
                    readPassed++;
                    mergedCount++;
                }
                recycleToPool1(tid, merged);
                mergeProcessed = true;
            } else if(mOptions->merge.includeUnmerged){
                int result1 = filterResults1[p];
                int result2 = filterResults2[p];

                if(isAdapterDimer) {
                    result1 = FAIL_ADAPTER_DIMER;
                    result2 = FAIL_ADAPTER_DIMER;
                }

                config->addFilterResult(result1, 1);
                if(result1 == PASS_FILTER && !dedupOut) {
                    r1->appendToString(mergedOutput);
                    config->getPostStats1()->statRead(r1);
                }

                config->addFilterResult(result2, 1);
                if(result2 == PASS_FILTER && !dedupOut) {
                    r2->appendToString(mergedOutput);
                    config->getPostStats1()->statRead(r2);
                }
                if(result1 == PASS_FILTER && result2 == PASS_FILTER )
                    readPassed++;
                mergeProcessed = true;
            }
        }

        if(!mergeProcessed) {

            int result1 = filterResults1[p];
            int result2 = filterResults2[p];

            if(isAdapterDimer) {
                // GPU may have counted stats for these reads — subtract if they passed GPU filter
                if(!mOptions->merge.enabled) {
                    if(result1 == PASS_FILTER && r1 != NULL)
                        config->getPostStats1()->unstatRead(r1);
                    if(result2 == PASS_FILTER && r2 != NULL)
                        config->getPostStats2()->unstatRead(r2);
                }
                result1 = FAIL_ADAPTER_DIMER;
                result2 = FAIL_ADAPTER_DIMER;
            }

            config->addFilterResult(max(result1, result2), 2);

            if(!dedupOut) {

                if( r1 != NULL &&  result1 == PASS_FILTER && r2 != NULL && result2 == PASS_FILTER ) {
                    
                    if(mOptions->outputToSTDOUT && !mOptions->merge.enabled) {
                        r1->appendToString(singleOutput);
                        r2->appendToString(singleOutput);
                    } else {
                        r1->appendToString(outstr1);
                        r2->appendToString(outstr2);
                    }

                    readPassed++;
                } else {
                    if( r1 != NULL &&  result1 == PASS_FILTER) {
                        if(mUnpairedLeftWriter) {
                            r1->appendToString(unpairedOut1);
                            if(mFailedWriter)
                                or2->appendToStringWithTag(failedOut, FAILED_TYPES[result2]);
                        } else {
                            // r1 passed filter individually but its partner failed and there
                            // is no unpaired output — r1 will not be written.
                            // filterBatchGPUWithStats already counted r1 in postStats1, so
                            // we must remove it to keep after_filtering counts accurate.
                            config->getPostStats1()->unstatRead(r1);
                            if(mFailedWriter) {
                                or1->appendToStringWithTag(failedOut, "paired_read_is_failing");
                                or2->appendToStringWithTag(failedOut, FAILED_TYPES[result2]);
                            }
                        }
                    } else if( r2 != NULL && result2 == PASS_FILTER) {
                        if(mUnpairedRightWriter) {
                            r2->appendToString(unpairedOut2);
                            if(mFailedWriter)
                                or1->appendToStringWithTag(failedOut,FAILED_TYPES[result1]);
                        } else if(mUnpairedLeftWriter) {
                            r2->appendToString(unpairedOut1);
                            if(mFailedWriter)
                                or1->appendToStringWithTag(failedOut,FAILED_TYPES[result1]);
                        }  else {
                            // r2 passed filter individually but its partner failed and there
                            // is no unpaired output — r2 will not be written.
                            // filterBatchGPUWithStats already counted r2 in postStats2, so
                            // we must remove it to keep after_filtering counts accurate.
                            config->getPostStats2()->unstatRead(r2);
                            if(mFailedWriter) {
                                or1->appendToStringWithTag(failedOut, FAILED_TYPES[result1]);
                                or2->appendToStringWithTag(failedOut, "paired_read_is_failing");
                            }
                        }
                    } else {
                        // Both reads failed
                        if(mFailedWriter) {
                            or1->appendToStringWithTag(failedOut, FAILED_TYPES[result1]);
                            or2->appendToStringWithTag(failedOut, FAILED_TYPES[result2]);
                        }
                    }
                }
            } else {
                // dedupOut: GPU counted stats for passing reads, but they should be excluded
                if(!mOptions->merge.enabled) {
                    if(r1 != NULL && result1 == PASS_FILTER)
                        config->getPostStats1()->unstatRead(r1);
                    if(r2 != NULL && result2 == PASS_FILTER)
                        config->getPostStats2()->unstatRead(r2);
                }
            }
        }

        // if no trimming applied, r1 should be identical to or1
        if(r1 != or1 && r1 != NULL) {
            recycleToPool1(tid, r1);
            r1 = NULL;
        }
        // if no trimming applied, r1 should be identical to or1
        if(r2 != or2 && r2 != NULL) {
            recycleToPool2(tid, r2);
            r2 = NULL;
        }

        if(or1) {
            recycleToPool1(tid, or1);
            or1 = NULL;
        }
        if(or2) {
            recycleToPool2(tid, or2);
            or2 = NULL;
        }
    }  // End of third pass loop
    PROF_END(_t_out_pe, cpu_output_ns);

	if(mOptions->split.enabled) {
        // split output by each worker thread
        if(!mOptions->out1.empty()) 
            config->getWriter1()->writeString(outstr1);
        if(!mOptions->out2.empty())
            config->getWriter2()->writeString(outstr2);
    } 

    if(mMergedWriter) {
        // write merged data
        mMergedWriter->input(tid, mergedOutput);
        mergedOutput = NULL;
    }

    if(mFailedWriter) {
        // write failed data
        mFailedWriter->input(tid, failedOut);
        failedOut = NULL;
    }

    if(mOverlappedWriter) {
        // write failed data
        mOverlappedWriter->input(tid, overlappedOut);
        overlappedOut = NULL;
    }

    // normal output by left/right writer thread
    if(mRightWriter && mLeftWriter) {
        // write PE
        mLeftWriter->input(tid, outstr1);
        outstr1 = NULL;

        mRightWriter->input(tid, outstr2);
        outstr2 = NULL;
    } else if(mLeftWriter) {
        // write singleOutput
        mLeftWriter->input(tid, singleOutput);
        singleOutput = NULL;
    }
    // output unpaired reads
    if(mUnpairedLeftWriter && mUnpairedRightWriter) {
        // write PE
        mUnpairedLeftWriter->input(tid, unpairedOut1);
        unpairedOut1 = NULL;

        mUnpairedRightWriter->input(tid, unpairedOut2);
        unpairedOut2 = NULL;
    } else if(mUnpairedLeftWriter) {
        mUnpairedLeftWriter->input(tid, unpairedOut1);
        unpairedOut1 = NULL;
    }

    if(mOptions->split.byFileLines)
        config->markProcessed(readPassed);
    else
        config->markProcessed(leftPack->count);

    if(mOptions->merge.enabled) {
        config->addMergedPairs(mergedCount);
    }

    if(outstr1)
        delete outstr1;
    if(outstr2)
        delete outstr2;
    if(unpairedOut1)
        delete unpairedOut1;
    if(unpairedOut2)
        delete unpairedOut2;
    if(singleOutput)
        delete singleOutput;
    if(mergedOutput)
        delete mergedOutput;
    if(failedOut)
        delete failedOut;
    if(overlappedOut)
        delete overlappedOut;

    delete[] leftPack->data;
    delete[] rightPack->data;
    delete leftPack;
    delete rightPack;

    mPackProcessedCounter++;

    return true;
}
    
void PairEndProcessor::statInsertSize(Read* r1, Read* r2, OverlapResult& ov, int frontTrimmed1, int frontTrimmed2) {
    int isize = mOptions->insertSizeMax;
    if(ov.overlapped) {
        if(ov.offset > 0)
            isize = r1->length() + r2->length() - ov.overlap_len + frontTrimmed1 + frontTrimmed2;
        else
            isize = ov.overlap_len + frontTrimmed1 + frontTrimmed2;
    }

    if(isize > mOptions->insertSizeMax)
        isize = mOptions->insertSizeMax;

    mInsertSizeHist[isize]++;
}

void PairEndProcessor::readerTask(bool isLeft)
{
    if(mOptions->verbose) {
        if(isLeft)
            loginfo("start to load data of read1");
        else
            loginfo("start to load data of read2");
    }
    long lastReported = 0;
    int slept = 0;
    long readNum = 0;
    bool splitSizeReEvaluated = false;
    Read** data = new Read*[mEffectivePackSize];
    memset(data, 0, sizeof(Read*)*mEffectivePackSize);
    FastqReader* reader = NULL;
    if(isLeft) {
        reader = new FastqReader(mOptions->in1, true, mOptions->phred64, mOptions->useGDS);
        reader->setReadPool(mLeftReadPool);
    }
    else {
        reader = new FastqReader(mOptions->in2, true, mOptions->phred64, mOptions->useGDS);
        reader->setReadPool(mRightReadPool);
    }
    // Enable async read-ahead only when multiple worker threads justify it.
    // At T=1 the extra concurrent fread() on the same file hurts slow storage.
    if (mEffectiveThreads > 1) reader->enableAsyncDecomp();

    int count=0;
    bool needToBreak = false;
    while(true){
        if(shouldStopReading)
            break;
        Read* read = reader->read();
        if(!read || needToBreak){
            // the last pack
            ReadPack* pack = new ReadPack;
            pack->data = data;
            pack->count = count;

            if(isLeft) {
                mLeftInputLists[mLeftPackReadCounter % mEffectiveThreads]->produce(pack);
                mLeftPackReadCounter++;
            } else {
                mRightInputLists[mRightPackReadCounter % mEffectiveThreads]->produce(pack);
                mRightPackReadCounter++;
            }
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
            string msg;
            if(isLeft)
                msg = "Read1: ";
            else
                msg = "Read2: ";
            msg += "loaded " + to_string((lastReported/1000000)) + "M reads";
            loginfo(msg);
        }
        // a full pack
        if(count == mEffectivePackSize || needToBreak){
            ReadPack* pack = new ReadPack;
            pack->data = data;
            pack->count = count;
            
            if(isLeft) {
                mLeftInputLists[mLeftPackReadCounter % mEffectiveThreads]->produce(pack);
                mLeftPackReadCounter++;
            } else {
                mRightInputLists[mRightPackReadCounter % mEffectiveThreads]->produce(pack);
                mRightPackReadCounter++;
            }

            //re-initialize data for next pack
            data = new Read*[mEffectivePackSize];
            memset(data, 0, sizeof(Read*)*mEffectivePackSize);
            // if the processor is far behind this reader, sleep and wait to limit memory usage
            if(isLeft) {
                while(mLeftPackReadCounter - mPackProcessedCounter > PACK_IN_MEM_LIMIT){
                    //cerr<<"sleep"<<endl;
                    slept++;
                    usleep(100);
                }
            } else {
                while(mRightPackReadCounter - mPackProcessedCounter > PACK_IN_MEM_LIMIT){
                    //cerr<<"sleep"<<endl;
                    slept++;
                    usleep(100);
                }
            }
            readNum += count;
            // if the writer threads are far behind this producer, sleep and wait
            // check this only when necessary
            if(readNum % (mEffectivePackSize * PACK_IN_MEM_LIMIT) == 0 && mLeftWriter) {
                while( (mLeftWriter && mLeftWriter->bufferLength() > PACK_IN_MEM_LIMIT) || (mRightWriter && mRightWriter->bufferLength() > PACK_IN_MEM_LIMIT) ){
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

    for(int t=0; t<mEffectiveThreads; t++) {
        if(isLeft)
            mLeftInputLists[t]->setProducerFinished();
        else
            mRightInputLists[t]->setProducerFinished();
    }

    if(mOptions->verbose) {
        if(isLeft) {
            mLeftReaderFinished = true;
            loginfo("Read1: loading completed with " + to_string(mLeftPackReadCounter) + " packs");
        }
        else {
            mRightReaderFinished = true;
            loginfo("Read2: loading completed with " + to_string(mRightPackReadCounter) + " packs");
        }
    }
    

    // if the last data initialized is not used, free it
    if(data != NULL)
        delete[] data;
    if(reader != NULL)
        delete reader;
}

void PairEndProcessor::interleavedReaderTask()
{
    if(mOptions->verbose)
        loginfo("start to load data");
    long lastReported = 0;
    int slept = 0;
    long readNum = 0;
    bool splitSizeReEvaluated = false;
    Read** dataLeft = new Read*[mEffectivePackSize];
    Read** dataRight = new Read*[mEffectivePackSize];
    memset(dataLeft, 0, sizeof(Read*)*mEffectivePackSize);
    memset(dataRight, 0, sizeof(Read*)*mEffectivePackSize);
    FastqReaderPair reader(mOptions->in1, mOptions->in2, true, mOptions->phred64, true, mOptions->useGDS);
    if (mEffectiveThreads > 1) reader.enableAsyncDecomp();
    int count=0;
    bool needToBreak = false;
    ReadPair* pair = new ReadPair();
    while(true){
        reader.read(pair);
        // TODO: put needToBreak here is just a WAR for resolve some unidentified dead lock issue 
        if(pair->eof() || needToBreak){
            // the last pack
            ReadPack* packLeft = new ReadPack;
            ReadPack* packRight = new ReadPack;
            packLeft->data = dataLeft;
            packRight->data = dataRight;
            packLeft->count = count;
            packRight->count = count;

            mLeftInputLists[mLeftPackReadCounter % mEffectiveThreads]->produce(packLeft);
            mLeftPackReadCounter++;

            mRightInputLists[mRightPackReadCounter % mEffectiveThreads]->produce(packRight);
            mRightPackReadCounter++;

            dataLeft = NULL;
            dataRight = NULL;
            break;
        }
        dataLeft[count] = pair->mLeft;
        dataRight[count] = pair->mRight;
        count++;
        // configured to process only first N reads
        if(mOptions->readsToProcess >0 && count + readNum >= mOptions->readsToProcess) {
            needToBreak = true;
        }
        if(mOptions->verbose && count + readNum >= lastReported + 1000000) {
            lastReported = count + readNum;
            string msg = "loaded " + to_string((lastReported/1000000)) + "M read pairs";
            loginfo(msg);
        }
        // a full pack
        if(count == mEffectivePackSize || needToBreak){
            ReadPack* packLeft = new ReadPack;
            ReadPack* packRight = new ReadPack;
            packLeft->data = dataLeft;
            packRight->data = dataRight;
            packLeft->count = count;
            packRight->count = count;

            mLeftInputLists[mLeftPackReadCounter % mEffectiveThreads]->produce(packLeft);
            mLeftPackReadCounter++;

            mRightInputLists[mRightPackReadCounter % mEffectiveThreads]->produce(packRight);
            mRightPackReadCounter++;

            //re-initialize data for next pack
            dataLeft = new Read*[mEffectivePackSize];
            dataRight = new Read*[mEffectivePackSize];
            memset(dataLeft, 0, sizeof(Read*)*mEffectivePackSize);
            memset(dataRight, 0, sizeof(Read*)*mEffectivePackSize);
            // if the consumer is far behind this producer, sleep and wait to limit memory usage
            while(mLeftPackReadCounter - mPackProcessedCounter > PACK_IN_MEM_LIMIT){
                //cerr<<"sleep"<<endl;
                slept++;
                usleep(100);
            }
            readNum += count;
            // if the writer threads are far behind this producer, sleep and wait
            // check this only when necessary
            if(readNum % (mEffectivePackSize * PACK_IN_MEM_LIMIT) == 0 && mLeftWriter) {
                while( (mLeftWriter && mLeftWriter->bufferLength() > PACK_IN_MEM_LIMIT) || (mRightWriter && mRightWriter->bufferLength() > PACK_IN_MEM_LIMIT) ){
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
                    reader.mLeft->getBytes(bytesRead, bytesTotal);
                    mOptions->split.size *=  (double)bytesTotal / ((double)bytesRead * (double) mOptions->split.number);
                    if(mOptions->split.size <= 0)
                        mOptions->split.size = 1;
                }
            }*/
        }
    }

    delete pair;

    for(int t=0; t<mEffectiveThreads; t++) {
        mLeftInputLists[t]->setProducerFinished();
        mRightInputLists[t]->setProducerFinished();
    }

    if(mOptions->verbose) {
        loginfo("interleaved: loading completed with " + to_string(mLeftPackReadCounter) + " packs");
    }

    mLeftReaderFinished = true;
    mRightReaderFinished = true;

    // if the last data initialized is not used, free it
    if(dataLeft != NULL)
        delete[] dataLeft;
    if(dataRight != NULL)
        delete[] dataRight;
}

void PairEndProcessor::processorTask(ThreadConfig* config)
{
    SingleProducerSingleConsumerList<ReadPack*>* inputLeft = config->getLeftInput();
    SingleProducerSingleConsumerList<ReadPack*>* inputRight = config->getRightInput();
    while(true) {
        if(config->canBeStopped()){
            break;
        }
        while(inputLeft->canBeConsumed() && inputRight->canBeConsumed()) {
            ReadPack* dataLeft = inputLeft->consume();
            ReadPack* dataRight = inputRight->consume();
            processPairEnd(dataLeft, dataRight, config);
        }
        if(inputLeft->isProducerFinished() && !inputLeft->canBeConsumed()) {
            break;
        } else if(inputRight->isProducerFinished() && !inputRight->canBeConsumed()) {
            break;
        } else {
            usleep(100);
        }
    }
    inputLeft->setConsumerFinished();
    inputRight->setConsumerFinished();

    mFinishedThreads++;
    if(mOptions->verbose) {
        string msg = "thread " + to_string(config->getThreadId() + 1) + " data processing completed";
        loginfo(msg);
    }

    if(mFinishedThreads == mEffectiveThreads) {
        if(mLeftWriter)
            mLeftWriter->setInputCompleted();
        if(mRightWriter)
            mRightWriter->setInputCompleted();
        if(mUnpairedLeftWriter)
            mUnpairedLeftWriter->setInputCompleted();
        if(mUnpairedRightWriter)
            mUnpairedRightWriter->setInputCompleted();
        if(mMergedWriter)
            mMergedWriter->setInputCompleted();
        if(mFailedWriter)
            mFailedWriter->setInputCompleted();
        if(mOverlappedWriter)
            mOverlappedWriter->setInputCompleted();
    }
    
    if(mOptions->verbose) {
        string msg = "thread " + to_string(config->getThreadId() + 1) + " finished";
        loginfo(msg);
    }
}

void PairEndProcessor::writerTask(WriterThread* config)
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
