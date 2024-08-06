#ifndef LEARNING_DECAY_HPP
#define LEARNING_DECAY_HPP

#include <unordered_map>
#include "Common.hpp"

class LearningDecay
{
public:
    LearningDecay(LearningDecayType decayType, const std::unordered_map<std::string, double> &params = {});
    double computeLearningRate(double initial_learning_rate, int epoch) const;

private:
    LearningDecayType decayType;
    std::unordered_map<std::string, double> params;
};

#endif // LEARNING_DECAY_HPP
