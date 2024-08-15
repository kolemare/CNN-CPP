#include "ELRALES.hpp"

ELRALES::ELRALES(double learning_rate_coef,
                 int maxSuccessiveEpochFailures,
                 int maxEpochFailures,
                 double tolerance,
                 const std::vector<std::shared_ptr<Layer>> &layers)
{
    if (1 < learning_rate_coef || 0 > learning_rate_coef)
    {
        throw std::runtime_error("ELRALES => Learning rate coefficient must be between 0 and 1.");
    }
    if (1 < tolerance || 0 > tolerance)
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
        else if (auto batch_norm_layer = std::dynamic_pointer_cast<BatchNormalizationLayer>(layer))
        {
            const auto &state = savedBatchNormStates[batch_norm_index++];
            batch_norm_layer->setGamma(state.gamma);
            batch_norm_layer->setBeta(state.beta);
        }
    }
}
