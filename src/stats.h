#ifndef STATS_H
#define STATS_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <map>
#include "read.h"
#include "options.h"

struct GpuBatchPostStats;

using namespace std;

struct RepeatMatchInfo {
    string seq;
    size_t length;
    size_t pos1;
    size_t pos2;
};

class Stats{
public:
    // this @guessedCycles parameter should be calculated using the first several records
    Stats(Options* opt, bool isRead2 = false, int guessedCycles = 0, int bufferMargin = 1024);
    ~Stats();
    int getCycles();
    long getReads();
    long getBases();
    long getQ20();
    long getQ30();
    long getQ40();
    long getGCNumber();
    long* getQualHist();
    // by default the qualified qual score is Q20 ('5')
    void statRead(Read* r);
    // Lightweight version of statRead that skips kmer, overrep, and long repeat analysis.
    // Matches GPU pre-stats behavior (processBatchStatsOnly computes only per-position stats).
    void statReadBasic(Read* r);
    // Reverse the per-base array updates of statRead (decrements counters).
    // Used to speculatively accumulate post-filter stats, then subtract failed reads.
    void unstatRead(Read* r);
    // Merge GPU-computed batch post-filter statistics into this Stats object.
    void mergeBatchStats(const struct GpuBatchPostStats& batchStats);

    static Stats* merge(vector<Stats*>& list);
    void print();
    void summarize(bool forced = false);
    // a port of JSON report
    void reportJson(ofstream& ofs, string padding);
    // a port of HTML report
    void reportHtml(ofstream& ofs, string filteringType, string readName);
    void reportHtmlQuality(ofstream& ofs, string filteringType, string readName);
    void reportHtmlContents(ofstream& ofs, string filteringType, string readName);
    void reportHtmlKMER(ofstream& ofs, string filteringType, string readName);
    void reportHtmlORA(ofstream& ofs, string filteringType, string readName);
    void reportHtmlLongRepeats(ofstream& ofs, string filteringType, string readName);
    bool isLongRead();
    void initOverRepSeq();
    int getMeanLength();
    void setLongRepeatEnabled(bool enabled);

public:
    static string list2string(double* list, int size);
    static string list2string(double* list, int size, long* coords);
    static string list2string(long* list, int size);
    static int base2val(char base);

private:
    void extendBuffer(int newBufLen);
    string makeKmerTD(int i, int j);
    string kmer3(int val);
    string kmer2(int val);
    void deleteOverRepSeqDist();
    bool overRepPassed(string& seq, long count);

private:
    Options* mOptions;
    bool mIsRead2;
    long mReads;
    int mEvaluatedSeqLen;
    /* 
    why we use 8 here?
    map A/T/C/G/N to 0~7 by their ASCII % 8:
    'A' % 8 = 1
    'T' % 8 = 4
    'C' % 8 = 3
    'G' % 8 = 7
    'N' % 8 = 6
    */
    long *mCycleQ30Bases[8];
    long *mCycleQ20Bases[8];
    long *mCycleBaseContents[8];
    long *mCycleBaseQual[8];
    long *mCycleTotalBase;
    long *mCycleTotalQual;
    long *mKmer;
    long mBaseQualHistogram[128];

    map<string, double*> mQualityCurves;
    map<string, double*> mContentCurves;
    map<string, long> mOverRepSeq;
    map<string, long*> mOverRepSeqDist;
    vector<RepeatMatchInfo> mLongRepeats;
    RepeatMatchInfo mLongestRepeat;
    bool mLongRepeatEnabled;


    int mCycles;
    int mBufLen;
    long mBases;
    long mQ20Bases[8];
    long mQ30Bases[8];
    long mBaseContents[8];
    long mQ20Total;
    long mQ30Total;
    long mQ40Total;
    bool summarized;
    long mKmerMax;
    long mKmerMin;
    int mKmerBufLen;
    long mLengthSum;
};

#endif