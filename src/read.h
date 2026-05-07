#ifndef READ_H
#define READ_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include "sequence.h"
#include <vector>

using namespace std;

class Read{
public:
	Read(string* name, string* seq, string* strand, string* quality, bool phred64=false);
    Read(const char* name, const char* seq, const char* strand, const char* quality, bool phred64=false);
    ~Read();
	void print();
    void printFile(ofstream& file);
    Read* reverseComplement();
    string firstIndex();
    string lastIndex();
    // default is Q20
    int lowQualCount(int qual=20);
    int length();
    string toString();
    string toStringWithTag(string tag);
    void appendToString(string* target);
    void appendToStringWithTag(string* target, string tag);
    void resize(int len);
    void convertPhred64To33();
    void trimFront(int len);
    bool fixMGI();
    
    // Strategy 1: Lazy trimming helpers
    void setLazyTrim(int offset, int length);
    int getEffectiveLength() const;
    int getLazyTrimOffset() const { return mLazyTrimOffset; }
    int getLazyTrimLength() const { return mLazyTrimLength; }
    void clearLazyTrim() { mLazyTrimOffset = 0; mLazyTrimLength = 0; }

public:
    static bool test();

private:


public:
	string* mName;
	string* mSeq;
	string* mStrand;
	string* mQuality;
	
	// Strategy 1: Lazy trimming fields - store trim offsets without actual string modification
	int mLazyTrimOffset;  // Offset into sequence where actual read starts (0 = no lazy trim)
	int mLazyTrimLength;  // Length of actual sequence (0 = use full from offset to end)
};

class ReadPair{
public:
    ReadPair();
    ~ReadPair();
    void setPair(Read* left, Read* right);
    bool eof();

    // merge a pair, without consideration of seq error caused false INDEL
    Read* fastMerge();
public:
    Read* mLeft;
    Read* mRight;

public:
    static bool test();
};

struct ReadPack {
    Read** data;
    int count;
};

typedef struct ReadPack ReadPack;

#endif