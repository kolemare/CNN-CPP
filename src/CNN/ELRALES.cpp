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

#include "ELRALES.hpp"

ELRALES::ELRALES(double learning_rate_coef,
                 int maxSuccessiveEpochFailures,
                 int maxEpochFailures,
                 double tolerance,
                 const std::vector<std::shared_ptr<Layer>> &layers)
{
    if (learning_rate_coef > 1.0 || learning_rate_coef < 0.0)
    {
        throw std::runtime_error("ELRALES => Learning rate coefficient must be between 0 and 1.");
    }
    if (tolerance > 1.0 || tolerance < 0.0)
    {
        throw std::runtime_error("ELRALES => Tolerance must be between 0 and 1.");
    }
    this->learning_rate_coef = learning_rate_coef;
    this->maxSuccessiveEpochFailures = maxSuccessiveEpochFailures;
    this->maxEpochFailures = maxEpochFailures;
    this->tolerance = tolerance;
    this->best_loss = std::numeric_limits<double>::max();
    this->previous_loss = std::numeric_limits<double>::max();
    this->successiveEpochFailures = 0;
    this->totalEpochFailures = 0;
    saveState(layers);
}

ELRALES_Retval ELRALES::updateState(double current_loss,
                                    std::vector<std::shared_ptr<Layer>> &layers,
                                    double &learning_rate,
                                    ELRALES_StateMachine &mode)
{
    ELRALES_Retval elralesRetval;
    if (current_loss < best_loss)
    {
        best_loss = current_loss;
        successiveEpochFailures = 0;
        saveState(layers);
        mode = ELRALES_StateMachine::NORMAL;
        elralesRetval = ELRALES_Retval::SUCCESSFUL_EPOCH;
        std::cout << "New best model saved with loss: " << best_loss << std::endl;
    }
    else if (current_loss < previous_loss)
    {
        successiveEpochFailures++;
        totalEpochFailures++;
        if (successiveEpochFailures <= maxSuccessiveEpochFailures && totalEpochFailures <= maxEpochFailures)
        {
            mode = ELRALES_StateMachine::RECOVERY;
            elralesRetval = ELRALES_Retval::WASTED_EPOCH;
        }
        else
        {
            mode = ELRALES_StateMachine::EARLY_STOPPING;
            elralesRetval = ELRALES_Retval::END_LEARNING;
            restoreState(layers);
        }
    }
    else if (current_loss < best_loss * (1 + tolerance))
    {
        successiveEpochFailures++;
        totalEpochFailures++;
        if (successiveEpochFailures <= maxSuccessiveEpochFailures && totalEpochFailures <= maxEpochFailures)
        {
            mode = ELRALES_StateMachine::LOSING;
            elralesRetval = ELRALES_Retval::WASTED_EPOCH;
        }
        else
        {
            mode = ELRALES_StateMachine::EARLY_STOPPING;
            elralesRetval = ELRALES_Retval::END_LEARNING;
            restoreState(layers);
        }
        double percentageToReachThreshold = ((best_loss * (1 + tolerance) - current_loss) / (best_loss * tolerance)) * 100;
        std::cout << "Epoch loss within tolerance, percentage left to reach tolerance: "
                  << percentageToReachThreshold
                  << " => "
                  << tolerance * 100
                  << "%"
                  << std::endl;
    }
    else
    {
        successiveEpochFailures++;
        totalEpochFailures++;
        if (successiveEpochFailures <= maxSuccessiveEpochFailures && totalEpochFailures <= maxEpochFailures)
        {
            mode = ELRALES_StateMachine::RECOVERY;
            elralesRetval = ELRALES_Retval::WASTED_EPOCH;
        }
        else
        {
            mode = ELRALES_StateMachine::EARLY_STOPPING;
            elralesRetval = ELRALES_Retval::END_LEARNING;
        }
        restoreState(layers);
        learning_rate *= learning_rate_coef;
        std::cout << "Restored to best learning epoch with loss " << best_loss << std::endl;
        std::cout << "Learning rate reduced to " << learning_rate << std::endl;
    }

    previous_loss = current_loss;
    return elralesRetval;
}

void ELRALES::saveState(const std::vector<std::shared_ptr<Layer>> &layers)
{
    savedConvLayerStates.clear();
    savedFCLayerStates.clear();
    savedBatchNormStates.clear();

    for (const auto &layer : layers)
    {
        if (auto conv_layer = std::dynamic_pointer_cast<ConvolutionLayer>(layer))
        {
            ConvolutionLayerState state;
            state.kernels = conv_layer->getKernels();
            state.biases = conv_layer->getBiases();

            if (layer->needsOptimizer())
            {
                auto optimizer = layer->getOptimizer();
                if (auto adam = std::dynamic_pointer_cast<Adam>(optimizer))
                {
                    state.optimizer_state.m_weights = adam->getMWeights();
                    state.optimizer_state.v_weights = adam->getVWeights();
                    state.optimizer_state.m_biases = adam->getMBiases();
                    state.optimizer_state.v_biases = adam->getVBiases();
                    state.optimizer_state.beta1 = adam->getBeta1();
                    state.optimizer_state.beta2 = adam->getBeta2();
                    state.optimizer_state.epsilon = adam->getEpsilon();
                    state.optimizer_state.t = adam->getT();
                }
                else if (auto rmsprop = std::dynamic_pointer_cast<RMSprop>(optimizer))
                {
                    state.optimizer_state.s_weights = rmsprop->getSWeights();
                    state.optimizer_state.s_biases = rmsprop->getSBiases();
                    state.optimizer_state.beta = rmsprop->getBeta();
                    state.optimizer_state.epsilon = rmsprop->getEpsilon();
                }
                else if (auto sgd_momentum = std::dynamic_pointer_cast<SGDWithMomentum>(optimizer))
                {
                    state.optimizer_state.v_weights = sgd_momentum->getVWeights();
                    state.optimizer_state.v_biases = sgd_momentum->getVBiases();
                    state.optimizer_state.momentum = sgd_momentum->getMomentum();
                }
            }

            savedConvLayerStates.push_back(state);
        }
        else if (auto fc_layer = std::dynamic_pointer_cast<FullyConnectedLayer>(layer))
        {
            FullyConnectedLayerState state;
            state.weights = fc_layer->getWeights();
            state.biases = fc_layer->getBiases();

            if (layer->needsOptimizer())
            {
                auto optimizer = layer->getOptimizer();
                if (auto adam = std::dynamic_pointer_cast<Adam>(optimizer))
                {
                    state.optimizer_state.m_weights = adam->getMWeights();
                    state.optimizer_state.v_weights = adam->getVWeights();
                    state.optimizer_state.m_biases = adam->getMBiases();
                    state.optimizer_state.v_biases = adam->getVBiases();
                    state.optimizer_state.beta1 = adam->getBeta1();
                    state.optimizer_state.beta2 = adam->getBeta2();
                    state.optimizer_state.epsilon = adam->getEpsilon();
                    state.optimizer_state.t = adam->getT();
                }
                else if (auto rmsprop = std::dynamic_pointer_cast<RMSprop>(optimizer))
                {
                    state.optimizer_state.s_weights = rmsprop->getSWeights();
                    state.optimizer_state.s_biases = rmsprop->getSBiases();
                    state.optimizer_state.beta = rmsprop->getBeta();
                    state.optimizer_state.epsilon = rmsprop->getEpsilon();
                }
                else if (auto sgd_momentum = std::dynamic_pointer_cast<SGDWithMomentum>(optimizer))
                {
                    state.optimizer_state.v_weights = sgd_momentum->getVWeights();
                    state.optimizer_state.v_biases = sgd_momentum->getVBiases();
                    state.optimizer_state.momentum = sgd_momentum->getMomentum();
                }
            }

            savedFCLayerStates.push_back(state);
        }
        else if (auto batch_norm_layer = std::dynamic_pointer_cast<BatchNormalizationLayer>(layer))
        {
            BatchNormalizationLayerState state;
            state.gamma = batch_norm_layer->getGamma();
            state.beta = batch_norm_layer->getBeta();
            savedBatchNormStates.push_back(state);
        }
    }
}

void ELRALES::restoreState(std::vector<std::shared_ptr<Layer>> &layers)
{
    size_t conv_index = 0;
    size_t fc_index = 0;
    size_t batch_norm_index = 0;

    for (const auto &layer : layers)
    {
        if (auto conv_layer = std::dynamic_pointer_cast<ConvolutionLayer>(layer))
        {
            const auto &state = savedConvLayerStates[conv_index++];
            conv_layer->setKernels(state.kernels);
            conv_layer->setBiases(state.biases);

            if (layer->needsOptimizer())
            {
                auto optimizer = layer->getOptimizer();
                if (auto adam = std::dynamic_pointer_cast<Adam>(optimizer))
                {
                    adam->setMWeights(state.optimizer_state.m_weights);
                    adam->setVWeights(state.optimizer_state.v_weights);
                    adam->setMBiases(state.optimizer_state.m_biases);
                    adam->setVBiases(state.optimizer_state.v_biases);
                    adam->setBeta1(state.optimizer_state.beta1);
                    adam->setBeta2(state.optimizer_state.beta2);
                    adam->setEpsilon(state.optimizer_state.epsilon);
                    adam->setT(state.optimizer_state.t);
                }
                else if (auto rmsprop = std::dynamic_pointer_cast<RMSprop>(optimizer))
                {
                    rmsprop->setSWeights(state.optimizer_state.s_weights);
                    rmsprop->setSBiases(state.optimizer_state.s_biases);
                    rmsprop->setBeta(state.optimizer_state.beta);
                    rmsprop->setEpsilon(state.optimizer_state.epsilon);
                }
                else if (auto sgd_momentum = std::dynamic_pointer_cast<SGDWithMomentum>(optimizer))
                {
                    sgd_momentum->setVWeights(state.optimizer_state.v_weights);
                    sgd_momentum->setVBiases(state.optimizer_state.v_biases);
                    sgd_momentum->setMomentum(state.optimizer_state.momentum);
                }
            }
        }
        else if (auto fc_layer = std::dynamic_pointer_cast<FullyConnectedLayer>(layer))
        {
            const auto &state = savedFCLayerStates[fc_index++];
            fc_layer->setWeights(state.weights);
            fc_layer->setBiases(state.biases);

            if (layer->needsOptimizer())
            {
                auto optimizer = layer->getOptimizer();
                if (auto adam = std::dynamic_pointer_cast<Adam>(optimizer))
                {
                    adam->setMWeights(state.optimizer_state.m_weights);
                    adam->setVWeights(state.optimizer_state.v_weights);
                    adam->setMBiases(state.optimizer_state.m_biases);
                    adam->setVBiases(state.optimizer_state.v_biases);
                    adam->setBeta1(state.optimizer_state.beta1);
                    adam->setBeta2(state.optimizer_state.beta2);
                    adam->setEpsilon(state.optimizer_state.epsilon);
                    adam->setT(state.optimizer_state.t);
                }
                else if (auto rmsprop = std::dynamic_pointer_cast<RMSprop>(optimizer))
                {
                    rmsprop->setSWeights(state.optimizer_state.s_weights);
                    rmsprop->setSBiases(state.optimizer_state.s_biases);
                    rmsprop->setBeta(state.optimizer_state.beta);
                    rmsprop->setEpsilon(state.optimizer_state.epsilon);
                }
                else if (auto sgd_momentum = std::dynamic_pointer_cast<SGDWithMomentum>(optimizer))
                {
                    sgd_momentum->setVWeights(state.optimizer_state.v_weights);
                    sgd_momentum->setVBiases(state.optimizer_state.v_biases);
                    sgd_momentum->setMomentum(state.optimizer_state.momentum);
                }
            }
        }
        else if (auto batch_norm_layer = std::dynamic_pointer_cast<BatchNormalizationLayer>(layer))
        {
            const auto &state = savedBatchNormStates[batch_norm_index++];
            batch_norm_layer->setGamma(state.gamma);
            batch_norm_layer->setBeta(state.beta);
        }
    }
}
