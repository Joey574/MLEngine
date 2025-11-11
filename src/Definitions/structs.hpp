#pragma once

struct Score {
    public:

    Score() {}
    Score(float score, bool highestIsBest) : score(score), highestIsBest(highestIsBest) {}

    inline float GetScore() const { return score; }
    inline bool IsBetterThan(float other) const { return highestIsBest ? score > other : score < other; }
    inline bool IsBetterThan(const Score& other) const { return highestIsBest ? score > other.score : score < other.score; }

    private:
    float score;
    bool highestIsBest;
};

struct TrainingConfig {
    public:

    size_t epochs;
    size_t batchSize;
    size_t testSize;
    int scoreFrequency;
};