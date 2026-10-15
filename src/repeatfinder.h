#ifndef REPEATFINDER_H
#define REPEATFINDER_H

#include <string>
#include <vector>
#include <cstddef>

struct RepeatMatch {
    size_t length;
    size_t pos1;
    size_t pos2;
};

class RepeatFinder {
public:
    static RepeatMatch findLongestRepeat(const std::string& seq, size_t minLen = 1);
    static std::vector<RepeatMatch> findLongRepeats(const std::string& seq, size_t minLen, size_t maxResults = 1000);

private:
    static void buildSuffixArray(const std::string& s, std::vector<int>& sa);
    static void buildLCP(const std::string& s, const std::vector<int>& sa, std::vector<int>& lcp);
};

#endif
