#pragma once

struct Score {
    public:

    Score() {}
    Score(float score, bool highestIsBest) : score(score), highestIsBest(highestIsBest) {}

    inline float GetScore() const { return score; }
    inline bool IsBetterThan(float other) const { return highestIsBest ? score > other : score < other; }
    inline bool IsBetterThan(const Score& other) const { return highestIsBest ? score > other.score : score < other.score; }

    protected:
    float score;
    bool highestIsBest;
};

struct Dropout {
    public:

    inline void Define(float rate, size_t nodes) {
        assert(!defined);
        this->rate = rate;
        bytes = (nodes+7)/8;

        dist = std::bernoulli_distribution(1.0f - rate);
        defined = true;
    }

    inline bool IsDefined() const { return defined; }    

    private:
    bool defined = false;

    float rate;
    size_t bytes;
    uint8_t* mask;
    std::bernoulli_distribution dist;
};

struct SkipConn {

    inline bool IsDefined() const { return defined; }    

    private:
    bool defined = false;
};