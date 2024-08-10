#ifndef LEARNING_DECAY_HPP
#define LEARNING_DECAY_HPP

#include <unordered_map>
#include "Common.hpp"

/**
 * @brief The LearningDecay class manages the decay of the learning rate during training.
 *
 * This class provides several decay strategies such as exponential, step, polynomial, inverse time, and cosine decay.
 */
class LearningDecay
{
public:
    /**
     * @brief Constructor for the LearningDecay class.
     *
     * Initializes the learning decay with the specified decay type and parameters.
     * If parameters are not provided, default values are used based on the decay type.
     *
     * @param decayType The type of learning rate decay to be applied.
     * @param params A map containing parameters for the specified decay type.
     */
    LearningDecay(LearningDecayType decayType,
                  const std::unordered_map<std::string, double> &params = {});

    /**
     * @brief Computes the learning rate based on the specified decay type and epoch.
     *
     * @param initial_learning_rate The initial learning rate before applying decay.
     * @param epoch The current training epoch.
     * @return The computed learning rate after applying decay.
     */
    double computeLearningRate(double initial_learning_rate,
                               int epoch) const;

private:
    LearningDecayType decayType;                    ///< The type of learning decay being used.
    std::unordered_map<std::string, double> params; ///< Parameters for the learning decay strategy.
};

#endif // LEARNING_DECAY_HPP
