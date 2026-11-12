#include "repeatfinder.h"

#include <algorithm>

RepeatMatch RepeatFinder::findLongestRepeat(const std::string& seq, size_t minLen) {
    RepeatMatch best{0, 0, 0};
    if(seq.size() < 2) {
        return best;
    }
    if(minLen < 1) {
        minLen = 1;
    }

    std::vector<int> sa;
    std::vector<int> lcp;
    buildSuffixArray(seq, sa);
    buildLCP(seq, sa, lcp);

    for(size_t i = 1; i < lcp.size(); i++) {
        size_t len = static_cast<size_t>(lcp[i]);
        if(len >= minLen && len > best.length) {
            best.length = len;
            best.pos1 = static_cast<size_t>(sa[i]);
            best.pos2 = static_cast<size_t>(sa[i - 1]);
        }
    }

    return best;
}

std::vector<RepeatMatch> RepeatFinder::findLongRepeats(const std::string& seq, size_t minLen, size_t maxResults) {
    std::vector<RepeatMatch> results;
    if(seq.size() < 2) {
        return results;
    }
    if(minLen < 1) {
        minLen = 1;
    }

    std::vector<int> sa;
    std::vector<int> lcp;
    buildSuffixArray(seq, sa);
    buildLCP(seq, sa, lcp);

    for(size_t i = 1; i < lcp.size(); i++) {
        size_t len = static_cast<size_t>(lcp[i]);
        if(len >= minLen) {
            RepeatMatch match;
            match.length = len;
            match.pos1 = static_cast<size_t>(sa[i]);
            match.pos2 = static_cast<size_t>(sa[i - 1]);
            results.push_back(match);
        }
    }

    std::sort(results.begin(), results.end(), [](const RepeatMatch& a, const RepeatMatch& b) {
        return a.length > b.length;
    });

    if(maxResults > 0 && results.size() > maxResults) {
        results.resize(maxResults);
    }

    return results;
}

void RepeatFinder::buildSuffixArray(const std::string& s, std::vector<int>& sa) {
    int n = static_cast<int>(s.size());
    sa.resize(n);
    if(n == 0) {
        return;
    }

    std::vector<int> rank(n);
    std::vector<int> tmp(n);

    for(int i = 0; i < n; i++) {
        sa[i] = i;
        rank[i] = static_cast<unsigned char>(s[i]);
    }

    for(int k = 1; k < n; k <<= 1) {
        auto cmp = [&](int i, int j) {
            if(rank[i] != rank[j]) {
                return rank[i] < rank[j];
            }
            int ri = (i + k < n) ? rank[i + k] : -1;
            int rj = (j + k < n) ? rank[j + k] : -1;
            return ri < rj;
        };

        std::sort(sa.begin(), sa.end(), cmp);

        tmp[sa[0]] = 0;
        for(int i = 1; i < n; i++) {
            tmp[sa[i]] = tmp[sa[i - 1]] + (cmp(sa[i - 1], sa[i]) ? 1 : 0);
        }
        rank.swap(tmp);

        if(rank[sa[n - 1]] == n - 1) {
            break;
        }
    }
}

void RepeatFinder::buildLCP(const std::string& s, const std::vector<int>& sa, std::vector<int>& lcp) {
    int n = static_cast<int>(s.size());
    lcp.assign(n, 0);
    if(n == 0) {
        return;
    }

    std::vector<int> rank(n);
    for(int i = 0; i < n; i++) {
        rank[sa[i]] = i;
    }

    int k = 0;
    for(int i = 0; i < n; i++) {
        int r = rank[i];
        if(r == 0) {
            continue;
        }
        int j = sa[r - 1];
        while(i + k < n && j + k < n && s[i + k] == s[j + k]) {
            k++;
        }
        lcp[r] = k;
        if(k > 0) {
            k--;
        }
    }
}
