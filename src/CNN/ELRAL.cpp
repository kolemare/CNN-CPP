#include "ELRAL.hpp"
#include <iostream>

ELRAL::ELRAL(double learning_rate_coef, int maxSuccessiveEpochFailures, int maxEpochFails, double tolerance, const std::vector<std::shared_ptr<Layer>> &layers)
    : learning_rate_coef(learning_rate_coef), maxSuccessiveEpochFailures(maxSuccessiveEpochFailures), maxEpochFails(maxEpochFails), tolerance(tolerance),
      best_loss(std::numeric_limits<double>::max()), successiveEpochFailures(0), totalEpochFailures(0)
{
    if (1 < learning_rate_coef || 0 > learning_rate_coef)
    {
        throw std::runtime_error("ELRAL => Learning rate coefficient must be between 0 and 1.");
    }
    if (0 > tolerance || 1 < tolerance)
    {
        throw std::runtime_error("ELRAL => Tolerance must be between 0 and 1.");
    }
    saveState(layers);
}

bool ELRAL::updateState(double current_loss, std::vector<std::shared_ptr<Layer>> &layers, double &learning_rate, ELRALMode &mode)
{
    if (current_loss < best_loss)
    {
        best_loss = current_loss;
        successiveEpochFailures = 0;
        saveState(layers);
        std::cout << "New best model saved with loss: " << best_loss << std::endl;
        mode = ELRALMode::NORMAL;
    }
    else if (current_loss < best_loss * (1 + tolerance))
    {
        double thresholdToBreak = best_loss * (1 + tolerance) - current_loss;
        std::cout << "Epoch loss within tolerance, threshold left to break: " << thresholdToBreak << std::endl;
        successiveEpochFailures++;
        mode = ELRALMode::LOSING;
    }
    else
    {
        successiveEpochFailures++;
        totalEpochFailures++;

        if (successiveEpochFailures <= maxSuccessiveEpochFailures)
        {
            if (totalEpochFailures <= maxEpochFails)
            {
                restoreState(layers);
                learning_rate *= learning_rate_coef;
                successiveEpochFailures = 0;
                std::cout << "Restored to best learning epoch. Learning rate reduced to: " << learning_rate << std::endl;
                mode = ELRALMode::RECOVERY;
            }
            else
            {
                mode = ELRALMode::DISABLED;
                throw std::runtime_error("ELRAL => Maximum number of failures reached.");
            }
        }
        else
        {
            mode = ELRALMode::DISABLED;
            throw std::runtime_error("ELRAL => Maximum number of successive failures reached.");
        }
    }
    if (ELRALMode::RECOVERY == mode || ELRALMode::LOSING == mode)
    {
        return false;
    }
    else
    {
        return true;
    }
}

void ELRAL::saveState(const std::vector<std::shared_ptr<Layer>> &layers)
{
    savedConvLayerStates.clear();
    savedFCLayerStates.clear();

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
                if (auto sgd_momentum = std::dynamic_pointer_cast<SGDWithMomentum>(optimizer))
                {
                    state.optimizer_state.v_weights_2d = sgd_momentum->getVWeights2D();
                    state.optimizer_state.v_biases_2d = sgd_momentum->getVBiases2D();
                    state.optimizer_state.v_weights_4d = sgd_momentum->getVWeights4D();
                    state.optimizer_state.v_biases_4d = sgd_momentum->getVBiases4D();
                }
                else if (auto adam = std::dynamic_pointer_cast<Adam>(optimizer))
                {
                    state.optimizer_state.m_weights_2d = adam->getMWeights2D();
                    state.optimizer_state.v_weights_2d = adam->getVWeights2D();
                    state.optimizer_state.m_biases_2d = adam->getMBiases2D();
                    state.optimizer_state.v_biases_2d = adam->getVBiases2D();
                    state.optimizer_state.m_weights_4d = adam->getMWeights4D();
                    state.optimizer_state.v_weights_4d = adam->getVWeights4D();
                    state.optimizer_state.m_biases_4d = adam->getMBiases4D();
                    state.optimizer_state.v_biases_4d = adam->getVBiases4D();
                }
                else if (auto rmsprop = std::dynamic_pointer_cast<RMSprop>(optimizer))
                {
                    state.optimizer_state.s_weights_2d = rmsprop->getSWeights2D();
                    state.optimizer_state.s_biases_2d = rmsprop->getSBiases2D();
                    state.optimizer_state.s_weights_4d = rmsprop->getSWeights4D();
                    state.optimizer_state.s_biases_4d = rmsprop->getSBiases4D();
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
                if (auto sgd_momentum = std::dynamic_pointer_cast<SGDWithMomentum>(optimizer))
                {
                    state.optimizer_state.v_weights_2d = sgd_momentum->getVWeights2D();
                    state.optimizer_state.v_biases_2d = sgd_momentum->getVBiases2D();
                    state.optimizer_state.v_weights_4d = sgd_momentum->getVWeights4D();
                    state.optimizer_state.v_biases_4d = sgd_momentum->getVBiases4D();
                }
                else if (auto adam = std::dynamic_pointer_cast<Adam>(optimizer))
                {
                    state.optimizer_state.m_weights_2d = adam->getMWeights2D();
                    state.optimizer_state.v_weights_2d = adam->getVWeights2D();
                    state.optimizer_state.m_biases_2d = adam->getMBiases2D();
                    state.optimizer_state.v_biases_2d = adam->getVBiases2D();
                    state.optimizer_state.m_weights_4d = adam->getMWeights4D();
                    state.optimizer_state.v_weights_4d = adam->getVWeights4D();
                    state.optimizer_state.m_biases_4d = adam->getMBiases4D();
                    state.optimizer_state.v_biases_4d = adam->getVBiases4D();
                }
                else if (auto rmsprop = std::dynamic_pointer_cast<RMSprop>(optimizer))
                {
                    state.optimizer_state.s_weights_2d = rmsprop->getSWeights2D();
                    state.optimizer_state.s_biases_2d = rmsprop->getSBiases2D();
                    state.optimizer_state.s_weights_4d = rmsprop->getSWeights4D();
                    state.optimizer_state.s_biases_4d = rmsprop->getSBiases4D();
                }
            }

            savedFCLayerStates.push_back(state);
        }
    }
}

void ELRAL::restoreState(std::vector<std::shared_ptr<Layer>> &layers)
{
    size_t conv_index = 0;
    size_t fc_index = 0;

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
                if (auto sgd_momentum = std::dynamic_pointer_cast<SGDWithMomentum>(optimizer))
                {
                    sgd_momentum->setVWeights2D(state.optimizer_state.v_weights_2d);
                    sgd_momentum->setVBiases2D(state.optimizer_state.v_biases_2d);
                    sgd_momentum->setVWeights4D(state.optimizer_state.v_weights_4d);
                    sgd_momentum->setVBiases4D(state.optimizer_state.v_biases_4d);
                }
                else if (auto adam = std::dynamic_pointer_cast<Adam>(optimizer))
                {
                    adam->setMWeights2D(state.optimizer_state.m_weights_2d);
                    adam->setVWeights2D(state.optimizer_state.v_weights_2d);
                    adam->setMBiases2D(state.optimizer_state.m_biases_2d);
                    adam->setVBiases2D(state.optimizer_state.v_biases_2d);
                    adam->setMWeights4D(state.optimizer_state.m_weights_4d);
                    adam->setVWeights4D(state.optimizer_state.v_weights_4d);
                    adam->setMBiases4D(state.optimizer_state.m_biases_4d);
                    adam->setVBiases4D(state.optimizer_state.v_biases_4d);
                }
                else if (auto rmsprop = std::dynamic_pointer_cast<RMSprop>(optimizer))
                {
                    rmsprop->setSWeights2D(state.optimizer_state.s_weights_2d);
                    rmsprop->setSBiases2D(state.optimizer_state.s_biases_2d);
                    rmsprop->setSWeights4D(state.optimizer_state.s_weights_4d);
                    rmsprop->setSBiases4D(state.optimizer_state.s_biases_4d);
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
                if (auto sgd_momentum = std::dynamic_pointer_cast<SGDWithMomentum>(optimizer))
                {
                    sgd_momentum->setVWeights2D(state.optimizer_state.v_weights_2d);
                    sgd_momentum->setVBiases2D(state.optimizer_state.v_biases_2d);
                    sgd_momentum->setVWeights4D(state.optimizer_state.v_weights_4d);
                    sgd_momentum->setVBiases4D(state.optimizer_state.v_biases_4d);
                }
                else if (auto adam = std::dynamic_pointer_cast<Adam>(optimizer))
                {
                    adam->setMWeights2D(state.optimizer_state.m_weights_2d);
                    adam->setVWeights2D(state.optimizer_state.v_weights_2d);
                    adam->setMBiases2D(state.optimizer_state.m_biases_2d);
                    adam->setVBiases2D(state.optimizer_state.v_biases_2d);
                    adam->setMWeights4D(state.optimizer_state.m_weights_4d);
                    adam->setVWeights4D(state.optimizer_state.v_weights_4d);
                    adam->setMBiases4D(state.optimizer_state.m_biases_4d);
                    adam->setVBiases4D(state.optimizer_state.v_biases_4d);
                }
                else if (auto rmsprop = std::dynamic_pointer_cast<RMSprop>(optimizer))
                {
                    rmsprop->setSWeights2D(state.optimizer_state.s_weights_2d);
                    rmsprop->setSBiases2D(state.optimizer_state.s_biases_2d);
                    rmsprop->setSWeights4D(state.optimizer_state.s_weights_4d);
                    rmsprop->setSBiases4D(state.optimizer_state.s_biases_4d);
                }
            }
        }
    }
}
