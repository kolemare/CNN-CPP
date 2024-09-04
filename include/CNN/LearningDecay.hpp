/*
MIT License
Copyright (c) 2024 Marko Kostić

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

This project is the CNN-CPP Framework. Usage of this code is free, and 
uploading and using the code is also free, with a humble request to mention 
the origin of the implementation, the author Marko Kostić, and the repository 
link: https://github.com/kolemare/CNN-CPP.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
*/

#ifndef LEARNING_DECAY_HPP
#define LEARNING_DECAY_HPP

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
