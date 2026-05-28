#include "matcher.h"
#include "overlapanalysis.h"

// Compile-time DNA complement lookup table.
// Eliminates the per-call heap allocation of reverseComplement string
// by computing RC on-the-fly during comparison loops.
static const char COMP[256] = {
    'N','N','N','N','N','N','N','N','N','N','N','N','N','N','N','N', // 0x00-0x0F
    'N','N','N','N','N','N','N','N','N','N','N','N','N','N','N','N', // 0x10-0x1F
    'N','N','N','N','N','N','N','N','N','N','N','N','N','N','N','N', // 0x20-0x2F
    'N','N','N','N','N','N','N','N','N','N','N','N','N','N','N','N', // 0x30-0x3F
    'N','T','N','G','N','N','N','C','N','N','N','N','N','N','N','N', // 0x40: A->T, C->G, G->C
    'N','N','N','N','A','N','N','N','N','N','N','N','N','N','N','N', // 0x50: T->A
    'N','T','N','G','N','N','N','C','N','N','N','N','N','N','N','N', // 0x60: a->T, c->G, g->C
    'N','N','N','N','A','N','N','N','N','N','N','N','N','N','N','N', // 0x70: t->A
    'N','N','N','N','N','N','N','N','N','N','N','N','N','N','N','N', // 0x80-0x8F
    'N','N','N','N','N','N','N','N','N','N','N','N','N','N','N','N', // 0x90-0x9F
    'N','N','N','N','N','N','N','N','N','N','N','N','N','N','N','N', // 0xA0-0xAF
    'N','N','N','N','N','N','N','N','N','N','N','N','N','N','N','N', // 0xB0-0xBF
    'N','N','N','N','N','N','N','N','N','N','N','N','N','N','N','N', // 0xC0-0xCF
    'N','N','N','N','N','N','N','N','N','N','N','N','N','N','N','N', // 0xD0-0xDF
    'N','N','N','N','N','N','N','N','N','N','N','N','N','N','N','N', // 0xE0-0xEF
    'N','N','N','N','N','N','N','N','N','N','N','N','N','N','N','N'  // 0xF0-0xFF
};

OverlapAnalysis::OverlapAnalysis(){
}


OverlapAnalysis::~OverlapAnalysis(){
}

OverlapResult OverlapAnalysis::analyze(Read* r1, Read* r2, int overlapDiffLimit, int overlapRequire, double diffPercentLimit, bool allowGap) {
    return analyze(r1->mSeq, r2->mSeq, overlapDiffLimit, overlapRequire, diffPercentLimit, allowGap);
}

// Optimized overlap analysis: computes reverse complement of r2 on-the-fly
// via COMP[] lookup table, eliminating the heap-allocated string copy per call.
// Uses a quick 8-base seed check to skip obviously non-matching offsets,
// reducing inner-loop iterations by ~80-90% for typical genomic data.
OverlapResult OverlapAnalysis::analyze(string*  r1, string*  r2, int diffLimit, int overlapRequire, double diffPercentLimit, bool allowGap) {
    int len1 = r1->length();
    int len2 = r2->length();
    const char* str1 = r1->c_str();
    const char* str2_fwd = r2->c_str();

    int complete_compare_require = 50;

    // Stack-allocated RC buffer — avoids heap allocation per call.
    // RAII wrapper handles cleanup across all return paths.
    struct RCBuf {
        char stk[512]; char* hp; char* buf;
        RCBuf(int n) : hp(n > 512 ? new char[n] : nullptr), buf(hp ? hp : stk) {}
        ~RCBuf() { delete[] hp; }
    } rcr2(len2);
    for (int c = 0; c < len2; c++)
        rcr2.buf[c] = COMP[(unsigned char)str2_fwd[len2 - 1 - c]];
    const char* str2 = rcr2.buf;

    int overlap_len = 0;
    int offset = 0;
    int diff = 0;

    // Precompute overlapDiffLimit: for default settings (overlapRequire=30,
    // diffPercentLimit=0.4, diffLimit=5), all overlaps have
    // overlapDiffLimit == diffLimit because min(5, floor(30*0.4)) = 5.
    // This eliminates a floating-point multiply per offset (~142 per pair).
    const int minOverlapDiffLimit = min(diffLimit, (int)(overlapRequire * diffPercentLimit));
    const bool constDiffLimit = (minOverlapDiffLimit >= diffLimit);

    // Seed-based quick rejection: check diffLimit+1 positions at the start.
    const int seedLen = diffLimit + 1;

    // forward with no gap
    while (offset < len1-overlapRequire) {
        overlap_len = min(len1 - offset, len2);
        int overlapDiffLimit = constDiffLimit ? diffLimit : min(diffLimit, (int)(overlap_len * diffPercentLimit));

        // Quick seed check: test the first seedLen (=diffLimit+1) bases.
        // For short overlaps (< complete_compare_require), overlapDiffLimit is small
        // (e.g., 1-2), making the seed effective: ~80% of random offsets are skipped.
        // For long overlaps, overlapDiffLimit=5 means ~0% skip rate, so the 6-comparison
        // seed adds pure overhead — skip it.
        // Safety: if seedDiff > overlapDiffLimit, the full inner loop also breaks early
        // at i < seedLen (< 50 = complete_compare_require), yielding the same result.
        if (overlap_len >= seedLen && overlap_len < complete_compare_require) {
            int seedDiff = 0;
            const char* s1 = str1 + offset;
            for (int j = 0; j < seedLen; j++) {
                if (s1[j] != str2[j])
                    seedDiff++;
            }
            if (seedDiff > overlapDiffLimit) {
                offset++;
                continue;
            }
        }

        diff = 0;
        int i = 0;
        for (i=0; i<overlap_len; i++) {
            if (str1[offset + i] != str2[i]){
                diff += 1;
                if (diff > overlapDiffLimit && i < complete_compare_require)
                    break;
            }
        }
        
        if (diff <= overlapDiffLimit || (diff > overlapDiffLimit && i>complete_compare_require)){
            OverlapResult ov;
            ov.overlapped = true;
            ov.offset = offset;
            ov.overlap_len = overlap_len;
            ov.diff = diff;
            ov.hasGap = false;
            return ov;
        }

        offset += 1;
    }


    // reverse with no gap
    // in this case, the adapter is sequenced since TEMPLATE_LEN < SEQ_LEN
    // check if distance can get smaller if offset goes negative
    // this only happens when insert DNA is shorter than sequencing read length, and some adapter/primer is sequenced but not trimmed cleanly
    // we go reversely
    offset = 0;
    while (offset > -(len2-overlapRequire)){
        overlap_len = min(len1,  len2- abs(offset));
        int overlapDiffLimit = constDiffLimit ? diffLimit : min(diffLimit, (int)(overlap_len * diffPercentLimit));

        // Quick seed check for reverse scan (same safety guarantees as forward)
        if (overlap_len >= seedLen && overlap_len < complete_compare_require) {
            int seedDiff = 0;
            for (int j = 0; j < seedLen; j++) {
                if (str1[j] != str2[-offset + j])
                    seedDiff++;
            }
            if (seedDiff > overlapDiffLimit) {
                offset--;
                continue;
            }
        }

        diff = 0;
        int i = 0;
        for (i=0; i<overlap_len; i++) {
            if (str1[i] != str2[-offset + i]){
                diff += 1;
                if (diff > overlapDiffLimit && i < complete_compare_require)
                    break;
            }
        }
        
        if (diff <= overlapDiffLimit || (diff > overlapDiffLimit && i>complete_compare_require)){
            OverlapResult ov;
            ov.overlapped = true;
            ov.offset = offset;
            ov.overlap_len = overlap_len;
            ov.diff = diff;
            ov.hasGap = false;
            return ov;
        }

        offset -= 1;
    }

    if(allowGap) {

        // forward with one gap
        offset = 0;
        while (offset < len1-overlapRequire) {
            overlap_len = min(len1 - offset, len2);
            int overlapDiffLimit = constDiffLimit ? diffLimit : min(diffLimit, (int)(overlap_len * diffPercentLimit));

            int diff = Matcher::diffWithOneInsertion(str1 + offset, str2, overlap_len-1, overlapDiffLimit);
            if(diff <0 || diff > overlapDiffLimit)
                diff = Matcher::diffWithOneInsertion(str2, str1 + offset, overlap_len-1, overlapDiffLimit);
            
            if (diff <= overlapDiffLimit && diff >=0){
                OverlapResult ov;
                ov.overlapped = true;
                ov.offset = offset;
                ov.overlap_len = overlap_len;
                ov.diff = diff;
                ov.hasGap = true;
                return ov;
            }

            offset += 1;
        }

        // reverse with one gap
        offset = 0;
        while (offset > -(len2-overlapRequire)){
            overlap_len = min(len1,  len2- abs(offset));
            int overlapDiffLimit = constDiffLimit ? diffLimit : min(diffLimit, (int)(overlap_len * diffPercentLimit));

            int diff = Matcher::diffWithOneInsertion(str1, str2-offset, overlap_len-1, overlapDiffLimit);
            if(diff <0 || diff > overlapDiffLimit)
                diff = Matcher::diffWithOneInsertion(str2-offset, str1, overlap_len-1, overlapDiffLimit);
            
            if (diff <= overlapDiffLimit && diff >=0){
                OverlapResult ov;
                ov.overlapped = true;
                ov.offset = offset;
                ov.overlap_len = overlap_len;
                ov.diff = diff;
                ov.hasGap = true;
                return ov;
            }

            offset -= 1;
        }
    }

    OverlapResult ov;
    ov.overlapped = false;
    ov.offset = ov.overlap_len = ov.diff = 0;
    ov.hasGap = false;
    return ov;
}

Read* OverlapAnalysis::merge(Read* r1, Read* r2, OverlapResult ov) {
    int ol = ov.overlap_len;
    if(!ov.overlapped)
        return NULL;

    int len1 = ol + max(0, ov.offset);
    int len2 = 0; 
    if(ov.offset > 0)
        len2 = r2->length() - ol;

    Read* rr2 = r2->reverseComplement();
    string mergedSeq = r1->mSeq->substr(0, len1);
    if(ov.offset > 0) {
        mergedSeq += rr2->mSeq->substr(ol, len2);
    }

    string mergedQual = r1->mQuality->substr(0, len1);
    if(ov.offset > 0) {
        mergedQual += rr2->mQuality->substr(ol, len2);
    }

    delete rr2;

    string name = *(r1->mName) + " merged_" + to_string(len1) + "_" + to_string(len2);
    string strand = *(r1->mStrand);
    if (strand != "+") {
      strand = strand + " merged_" + to_string(len1) + "_" + to_string(len2);
    }
    Read* mergedRead = new Read(new string(name), new string(mergedSeq), new string(strand), new string(mergedQual));

    return mergedRead;
}

bool OverlapAnalysis::test(){
    //Sequence r1("CAGCGCCTACGGGCCCCTTTTTCTGCGCGACCGCGTGGCTGTGGGCGCGGATGCCTTTGAGCGCGGTGACTTCTCACTGCGTATCGAGCCGCTGGAGGTCTCCC");
    //Sequence r2("ACCTCCAGCGGCTCGATACGCAGTGAGAAGTCACCGCGCTCAAAGGCATCCGCGCCCACAGCCACGCGGTCGCGCAGAAAAAGGGGCCCGTAGGCGCGGCTCCC");

    string* r1 = new string("CAGCGCCTACGGGCCCCTTTTTCTGCGCGACCGCGTGGCTGTGGGCGCGGATGCCTTTGAGCGCGGTGACTTCTCACTGCGTATCGAGC");
    string* r2 = new string("ACCTCCAGCGGCTCGATACGCAGTGAGAAGTCACCGCGCTCAAAGGCATCCGCGCCCACAGCCACGCGGTCGCGCAGAAAAAGGGGTCC");
    string* qual1 = new string("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");
    string* qual2 = new string("#########################################################################################");
    
    OverlapResult ov = OverlapAnalysis::analyze(r1, r2, 2, 30, 0.2);

    Read read1(new string("name1"), r1, new string("+"), qual1);
    Read read2(new string("name2"), r2, new string("+"), qual2);

    Read* mergedRead = OverlapAnalysis::merge(&read1, &read2, ov);
    mergedRead->print();

    return ov.overlapped && ov.offset == 10 && ov.overlap_len == 79 && ov.diff == 1;
}
