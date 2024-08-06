#include "LearningDecay.hpp"
#include <cmath>
#include <stdexcept>

LearningDecay::LearningDecay(LearningDecayType decayType, const std::unordered_map<std::string, double> &params)
    : decayType(decayType), params(params)
{
    // Set default parameters if they are not provided
    switch (decayType)
    {
    case LearningDecayType::EXPONENTIAL:
        if (this->params.find("decay_rate") == this->params.end())
        {
            this->params["decay_rate"] = 0.96; // Default decay rate for exponential decay
        }
        break;

    case LearningDecayType::STEP:
        if (this->params.find("step_size") == this->params.end())
        {
            this->params["step_size"] = 10; // Default step size
        }
        if (this->params.find("decay_factor") == this->params.end())
        {
            this->params["decay_factor"] = 0.5; // Default decay factor for step decay
        }
        break;

    case LearningDecayType::POLYNOMIAL:
        if (this->params.find("end_learning_rate") == this->params.end())
        {
            this->params["end_learning_rate"] = 0.001; // Default end learning rate
        }
        if (this->params.find("decay_steps") == this->params.end())
        {
            this->params["decay_steps"] = 1000; // Default decay steps
        }
        if (this->params.find("power") == this->params.end())
        {
            this->params["power"] = 2.0; // Default power for polynomial decay
        }
        break;

    case LearningDecayType::INVERSE_TIME:
        if (this->params.find("decay_rate") == this->params.end())
        {
            this->params["decay_rate"] = 0.1; // Default decay rate for inverse time decay
        }
        if (this->params.find("decay_steps") == this->params.end())
        {
            this->params["decay_steps"] = 1000; // Default decay steps
        }
        break;

    case LearningDecayType::COSINE:
        if (this->params.find("decay_steps") == this->params.end())
        {
            this->params["decay_steps"] = 1000; // Default decay steps for cosine decay
        }
        if (this->params.find("alpha") == this->params.end())
        {
            this->params["alpha"] = 0.0; // Default alpha for cosine decay
        }
        break;

    case LearningDecayType::NONE:
    default:
        break;
    }
}

double LearningDecay::computeLearningRate(double initial_learning_rate, int epoch) const
{
    switch (decayType)
    {
    case LearningDecayType::EXPONENTIAL:
    {
        double decay_rate = params.at("decay_rate");
        return initial_learning_rate * std::pow(decay_rate, epoch);
    }
    case LearningDecayType::STEP:
    {
        double step_size = params.at("step_size");
        double decay_factor = params.at("decay_factor");
        return initial_learning_rate * std::pow(decay_factor, epoch / step_size);
    }
    case LearningDecayType::POLYNOMIAL:
    {
        double end_learning_rate = params.at("end_learning_rate");
        double decay_steps = params.at("decay_steps");
        double power = params.at("power");
        return (initial_learning_rate - end_learning_rate) * std::pow(1 - static_cast<double>(epoch) / decay_steps, power) + end_learning_rate;
    }
    case LearningDecayType::INVERSE_TIME:
    {
        double decay_rate = params.at("decay_rate");
        double decay_steps = params.at("decay_steps");
        return initial_learning_rate / (1 + decay_rate * static_cast<double>(epoch) / decay_steps);
    }
    case LearningDecayType::COSINE:
    {
        double decay_steps = params.at("decay_steps");
        double alpha = params.at("alpha");
        return initial_learning_rate * (1 + alpha * std::cos(M_PI * static_cast<double>(epoch) / decay_steps)) / 2;
    }
    case LearningDecayType::NONE:
    default:
        return initial_learning_rate;
    }
}
