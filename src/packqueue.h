#ifndef PACK_QUEUE_H
#define PACK_QUEUE_H

#include <mutex>
#include <condition_variable>
#include <deque>
#include <chrono>
#include "read.h"

// Thread-safe multi-producer/multi-consumer queue used to hand ReadPack*
// packs from the reader thread(s) to the whole worker pool, instead of the
// legacy design where the reader striped packs round-robin into a fixed
// per-thread queue that only its owning worker could drain.
//
// That striping meant a worker stuck on an expensive pack head-of-line
// blocked the writer (which drains output in the same round-robin order)
// even while every other worker sat idle. Pulling from one shared queue
// lets whichever worker is free next take the next pack, so load balances
// naturally under uneven per-pack cost. Locking is per-pack (up to
// MAX_PACK_SIZE reads), not per-read, so mutex overhead is negligible.
//
// Used only on the non-split output path; --split output depends on a
// fixed pack-index-to-thread mapping (see ThreadConfig::markProcessed) and
// keeps the legacy SingleProducerSingleConsumerList mechanism.
template<typename T>
class PackQueue {
public:
    PackQueue() : mProducerFinished(false) {}

    void produce(T val) {
        std::lock_guard<std::mutex> lk(mMutex);
        mQueue.push_back(val);
        mCv.notify_one();
    }

    // Blocks up to timeoutUs for an item. Returns false if none arrived in
    // that window (caller should check isDrained() to decide whether to
    // keep retrying or stop).
    bool consume(T& out, int timeoutUs) {
        std::unique_lock<std::mutex> lk(mMutex);
        if (mQueue.empty()) {
            mCv.wait_for(lk, std::chrono::microseconds(timeoutUs),
                [this]{ return !mQueue.empty() || mProducerFinished; });
        }
        if (mQueue.empty())
            return false;
        out = mQueue.front();
        mQueue.pop_front();
        return true;
    }

    void setProducerFinished() {
        std::lock_guard<std::mutex> lk(mMutex);
        mProducerFinished = true;
        mCv.notify_all();
    }

    bool isDrained() {
        std::lock_guard<std::mutex> lk(mMutex);
        return mProducerFinished && mQueue.empty();
    }

private:
    std::mutex mMutex;
    std::condition_variable mCv;
    std::deque<T> mQueue;
    bool mProducerFinished;
};

// Paired-end variant: reader threads for R1 and R2 each push independently,
// but workers must consume one R1 pack together with the matching R2 pack.
// Both producers push in strict pack-index order, so the Nth item popped
// from the left side always pairs correctly with the Nth item popped from
// the right side, as long as the pop of both sides happens under one lock.
class PairedPackQueue {
public:
    PairedPackQueue() : mProducerFinished(false) {}

    void produceLeft(ReadPack* p) {
        std::lock_guard<std::mutex> lk(mMutex);
        mLeft.push_back(p);
        mCv.notify_all();
    }

    void produceRight(ReadPack* p) {
        std::lock_guard<std::mutex> lk(mMutex);
        mRight.push_back(p);
        mCv.notify_all();
    }

    bool consumePair(ReadPack*& left, ReadPack*& right, int timeoutUs) {
        std::unique_lock<std::mutex> lk(mMutex);
        if (mLeft.empty() || mRight.empty()) {
            mCv.wait_for(lk, std::chrono::microseconds(timeoutUs),
                [this]{ return (!mLeft.empty() && !mRight.empty()) || mProducerFinished; });
        }
        if (mLeft.empty() || mRight.empty())
            return false;
        left = mLeft.front();  mLeft.pop_front();
        right = mRight.front(); mRight.pop_front();
        return true;
    }

    void setProducerFinished() {
        std::lock_guard<std::mutex> lk(mMutex);
        mProducerFinished = true;
        mCv.notify_all();
    }

    // Matches a pre-existing property of the legacy per-thread-list design:
    // on malformed (mismatched R1/R2 count) input, the side that runs out
    // first stops the pipeline; processPairEnd()'s count check is what
    // actually flags that case as a fatal error.
    bool isDrained() {
        std::lock_guard<std::mutex> lk(mMutex);
        return mProducerFinished && (mLeft.empty() || mRight.empty());
    }

private:
    std::mutex mMutex;
    std::condition_variable mCv;
    std::deque<ReadPack*> mLeft;
    std::deque<ReadPack*> mRight;
    bool mProducerFinished;
};

#endif
