#include "NeuralNetwork.hpp"
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <filesystem>

NeuralNetwork::NeuralNetwork() : flattenAdded(false),
                                 clippingSet(false),
                                 elralesSet(false),
                                 currentDepth(3),
                                 logLevel(LogLevel::None),
                                 progressLevel(ProgressLevel::None) {}

void NeuralNetwork::setImageSize(const int targetWidth,
                                 const int targetHeight)
{
    inputHeight = targetHeight;
    inputWidth = targetWidth;
}

void NeuralNetwork::setLogLevel(LogLevel level)
{
    logLevel = level;
}

void NeuralNetwork::setProgressLevel(ProgressLevel level)
{
    progressLevel = level;
}

void NeuralNetwork::addConvolutionLayer(int filters,
                                        int kernel_size,
                                        int stride,
                                        int padding,
                                        ConvKernelInitialization kernel_init,
                                        ConvBiasInitialization bias_init)
{
    layers.push_back(std::make_shared<ConvolutionLayer>(filters, kernel_size, stride, padding, kernel_init, bias_init));
    if (logLevel == LogLevel::All)
    {
        std::cout << "Added Convolution Layer with " << filters << " filters, kernel size " << kernel_size << ", stride " << stride << ", padding " << padding << std::endl;
    }
}

void NeuralNetwork::addMaxPoolingLayer(int pool_size,
                                       int stride)
{
    layers.push_back(std::make_shared<MaxPoolingLayer>(pool_size, stride));
    if (logLevel == LogLevel::All)
    {
        std::cout << "Added Max Pooling Layer with pool size " << pool_size << ", stride " << stride << std::endl;
    }
}

void NeuralNetwork::addAveragePoolingLayer(int pool_size,
                                           int stride)
{
    layers.push_back(std::make_shared<AveragePoolingLayer>(pool_size, stride));
    if (logLevel == LogLevel::All)
    {
        std::cout << "Added Average Pooling Layer with pool size " << pool_size << ", stride " << stride << std::endl;
    }
}

void NeuralNetwork::addFlattenLayer()
{
    if (!flattenAdded)
    {
        layers.push_back(std::make_shared<FlattenLayer>());
        flattenAdded = true;
        if (logLevel == LogLevel::All)
        {
            std::cout << "Added Flatten Layer" << std::endl;
        }
    }
    else
    {
        std::cerr << "Flatten layer already added." << std::endl;
    }
}

void NeuralNetwork::addFullyConnectedLayer(int output_size,
                                           DenseWeightInitialization weight_init,
                                           DenseBiasInitialization bias_init)
{
    layers.push_back(std::make_shared<FullyConnectedLayer>(output_size, weight_init, bias_init));
    if (logLevel == LogLevel::All)
    {
        std::cout << "Added Fully Connected Layer with output size " << output_size << std::endl;
    }
}

void NeuralNetwork::addActivationLayer(ActivationType type)
{
    layers.push_back(std::make_shared<ActivationLayer>(type));
    if (logLevel == LogLevel::All)
    {
        std::cout << "Added Activation Layer of type " << static_cast<int>(type) << std::endl;
    }
}

void NeuralNetwork::setLossFunction(LossType type)
{
    lossFunction = LossFunction::create(type);
    if (logLevel == LogLevel::All)
    {
        std::cout << "Set Loss Function of type " << static_cast<int>(type) << std::endl;
    }
}

void NeuralNetwork::compile(OptimizerType optimizerType,
                            const std::unordered_map<std::string, double> &optimizer_params)
{
    std::unordered_map<std::string, double> default_params;

    switch (optimizerType)
    {
    case OptimizerType::SGD:
        // No parameters needed for SGD, empty map
        break;
    case OptimizerType::SGDWithMomentum:
        default_params = {{"momentum", 0.9}};
        break;
    case OptimizerType::Adam:
        default_params = {{"beta1", 0.9}, {"beta2", 0.999}, {"epsilon", 1e-7}};
        break;
    case OptimizerType::RMSprop:
        default_params = {{"beta", 0.9}, {"epsilon", 1e-7}};
        break;
    default:
        throw std::invalid_argument("Unknown optimizer type");
    }

    // Combine provided params with defaults, preferring provided params
    for (const auto &param : optimizer_params)
    {
        default_params[param.first] = param.second;
    }

    optimizer = Optimizer::create(optimizerType, default_params);

    int height = inputHeight;
    int width = inputWidth;
    int input_size = -1;

    if (!clippingSet)
    {
        // Default => GradientClipping DISABLED
        this->enableGradientClipping(0, GradientClippingMode::DISABLED);
    }

    if (!elralesSet)
    {
        // Default => ELRALES DISABLED
        this->enableELRALES(0.0, 0, 0, 0.0, ELRALES_Mode::DISABLED);
    }

    for (size_t i = 0; i < layers.size(); ++i)
    {
        if (auto conv_layer = dynamic_cast<ConvolutionLayer *>(layers[i].get()))
        {
            conv_layer->setInputDepth(currentDepth);
            currentDepth = conv_layer->getFilters();
            height = (height - conv_layer->getKernelSize() + 2 * conv_layer->getPadding()) / conv_layer->getStride() + 1;
            width = (width - conv_layer->getKernelSize() + 2 * conv_layer->getPadding()) / conv_layer->getStride() + 1;
            conv_layer->setOptimizer(optimizer);
        }
        else if (auto pool_layer = dynamic_cast<MaxPoolingLayer *>(layers[i].get()))
        {
            height = (height - pool_layer->getPoolSize()) / pool_layer->getStride() + 1;
            width = (width - pool_layer->getPoolSize()) / pool_layer->getStride() + 1;
        }
        else if (auto pool_layer = dynamic_cast<AveragePoolingLayer *>(layers[i].get()))
        {
            height = (height - pool_layer->getPoolSize()) / pool_layer->getStride() + 1;
            width = (width - pool_layer->getPoolSize()) / pool_layer->getStride() + 1;
        }
        else if (auto fc_layer = dynamic_cast<FullyConnectedLayer *>(layers[i].get()))
        {
            if (input_size == -1)
            {
                throw std::runtime_error("Input size for FullyConnectedLayer cannot be determined.");
            }
            fc_layer->setInputSize(input_size);
            input_size = fc_layer->getOutputSize();
            fc_layer->setOptimizer(optimizer);
        }
        else if (dynamic_cast<FlattenLayer *>(layers[i].get()))
        {
            input_size = height * width * currentDepth;
        }
    }
}

Eigen::Tensor<double, 4> NeuralNetwork::forward(const Eigen::Tensor<double, 4> &input)
{
    if (logLevel == LogLevel::All || logLevel == LogLevel::LayerOutputs)
    {
        NNLogger::printTensorSummary(input, "INPUT", PropagationType::FORWARD);
    }

    Eigen::Tensor<double, 4> output = input;
    layerInputs.clear();

    for (size_t i = 0; i < layers.size(); ++i)
    {
        layerInputs.push_back(output);
        output = layers[i]->forward(output);

        if (logLevel == LogLevel::All || logLevel == LogLevel::LayerOutputs)
        {
            std::string layerType;
            if (dynamic_cast<ConvolutionLayer *>(layers[i].get()))
            {
                layerType = "Convolution Layer";
            }
            else if (dynamic_cast<MaxPoolingLayer *>(layers[i].get()))
            {
                layerType = "Max Pooling Layer";
            }
            else if (dynamic_cast<AveragePoolingLayer *>(layers[i].get()))
            {
                layerType = "Average Pooling Layer";
            }
            else if (dynamic_cast<FlattenLayer *>(layers[i].get()))
            {
                layerType = "Flatten Layer";
            }
            else if (dynamic_cast<FullyConnectedLayer *>(layers[i].get()))
            {
                layerType = "Fully Connected Layer";
            }
            else if (dynamic_cast<ActivationLayer *>(layers[i].get()))
            {
                layerType = "Activation Layer";
            }

            if (logLevel == LogLevel::All)
            {
                NNLogger::printFullTensor(output, layerType, PropagationType::FORWARD);
            }
            else
            {
                NNLogger::printTensorSummary(output, layerType, PropagationType::FORWARD);
            }
        }
    }

    return output;
}

void NeuralNetwork::backward(const Eigen::Tensor<double, 4> &d_output,
                             double learning_rate)
{
    if (logLevel == LogLevel::All || logLevel == LogLevel::LayerOutputs)
    {
        NNLogger::printTensorSummary(d_output, "OUTPUT", PropagationType::BACK);
    }

    Eigen::Tensor<double, 4> d_input = d_output;

    for (int i = layers.size() - 1; i >= 0; --i)
    {
        std::string layerType;
        if (dynamic_cast<ConvolutionLayer *>(layers[i].get()))
        {
            layerType = "Convolution Layer";
        }
        else if (dynamic_cast<MaxPoolingLayer *>(layers[i].get()))
        {
            layerType = "Max Pooling Layer";
        }
        else if (dynamic_cast<AveragePoolingLayer *>(layers[i].get()))
        {
            layerType = "Average Pooling Layer";
        }
        else if (dynamic_cast<FlattenLayer *>(layers[i].get()))
        {
            layerType = "Flatten Layer";
        }
        else if (dynamic_cast<FullyConnectedLayer *>(layers[i].get()))
        {
            layerType = "Fully Connected Layer";
        }
        else if (dynamic_cast<ActivationLayer *>(layers[i].get()))
        {
            layerType = "Activation Layer";
        }

        d_input = layers[i]->backward(d_input, layerInputs[i], learning_rate);

        if (GradientClippingMode::ENABLED == clippingMode)
        {
            GradientClipping::clipGradients(d_input, clipValue);
        }

        if (logLevel == LogLevel::All || logLevel == LogLevel::LayerOutputs)
        {
            if (logLevel == LogLevel::All)
            {
                NNLogger::printFullTensor(d_input, layerType, PropagationType::BACK);
            }
            else
            {
                NNLogger::printTensorSummary(d_input, layerType, PropagationType::BACK);
            }
        }
    }
}

void NeuralNetwork::train(const ImageContainer &imageContainer,
                          int epochs,
                          int batch_size,
                          double learning_rate)
{
    if (!lossFunction)
    {
        throw std::runtime_error("Loss function must be set before training.");
    }

    NNLogger::initializeCSV("cnn.csv");

    BatchManager batchManager(imageContainer, batch_size, BatchType::Training);
    std::cout << "Training started..." << std::endl;
    auto start = std::chrono::steady_clock::now();
    double current_learning_rate = learning_rate;

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        Eigen::Tensor<double, 4> batch_input;
        Eigen::Tensor<int, 2> batch_label;
        int totalBatches = batchManager.getTotalBatches();
        int batchCounter = 0;
        double total_epoch_loss = 0.0;
        int correct_predictions = 0;
        int num_epoch_samples = 0;

        while (batchManager.getNextBatch(batch_input, batch_label))
        {
            // Forward pass
            Eigen::Tensor<double, 4> predictions = forward(batch_input);

            // Compute loss
            double batch_loss = lossFunction->compute(predictions, batch_label);
            total_epoch_loss += batch_loss * batch_input.dimension(0);

            // Count correct predictions
            for (int i = 0; i < predictions.dimension(0); ++i)
            {
                int predicted_label;
                int true_label;

                if (predictions.dimension(3) == 1) // Binary classification
                {
                    predicted_label = (predictions(i, 0, 0, 0) >= 0.5) ? 1 : 0;
                    true_label = batch_label(i, 0);
                }
                else // Multi-class classification
                {
                    predicted_label = 0;
                    double max_value = predictions(i, 0, 0, 0);

                    for (int j = 1; j < predictions.dimension(3); ++j)
                    {
                        if (predictions(i, 0, 0, j) > max_value)
                        {
                            max_value = predictions(i, 0, 0, j);
                            predicted_label = j;
                        }
                    }

                    true_label = 0;
                    for (int j = 0; j < batch_label.dimension(1); ++j)
                    {
                        if (batch_label(i, j) == 1)
                        {
                            true_label = j;
                            break;
                        }
                    }
                }

                if (predicted_label == true_label)
                {
                    correct_predictions++;
                }
                num_epoch_samples++;
            }

            // Backward pass
            Eigen::Tensor<double, 4> d_output = lossFunction->derivative(predictions, batch_label);
            backward(d_output, current_learning_rate);

            NNLogger::printProgress(epoch, epochs, batchCounter, totalBatches, start, batch_loss, progressLevel);
            batchCounter++;
        }

        double average_loss = total_epoch_loss / num_epoch_samples;
        double accuracy = static_cast<double>(correct_predictions) / num_epoch_samples;

        std::cout << "Evaluating..." << std::endl;
        std::tuple<double, double> evaluation = evaluate(imageContainer);

        if (ELRALES_Mode::ENABLED == elralesMode)
        {
            ELRALES_Retval elralesEvaluation = elrales->updateState(average_loss, layers, current_learning_rate, elralesStateMachine);
            std::string elralesState = toString(elralesStateMachine);

            if (ELRALES_Retval::SUCCESSFUL_EPOCH == elralesEvaluation)
            {
                std::cout << "Epoch " << epoch + 1 << " completed." << std::endl;
                std::cout << "Training Accuracy: " << accuracy << std::endl;
                std::cout << "Training Loss: " << average_loss << std::endl;
                std::cout << "Testing Accuracy: " << std::get<0>(evaluation) << std::endl;
                std::cout << "Testing Loss: " << std::get<1>(evaluation) << std::endl;
                std::cout << "ELRALES: " << elralesState << std::endl;
                NNLogger::appendToCSV("cnn.csv", epoch + 1, accuracy, average_loss, std::get<0>(evaluation), std::get<1>(evaluation), elralesState);
            }
            else if (ELRALES_Retval::WASTED_EPOCH == elralesEvaluation)
            {
                std::cout << "Wasted Epoch " << epoch + 1 << " completed." << std::endl;
                std::cout << "Wasted Training Accuracy: " << accuracy << std::endl;
                std::cout << "Wasted Training Loss: " << average_loss << std::endl;
                std::cout << "Wasted Testing Accuracy: " << std::get<0>(evaluation) << std::endl;
                std::cout << "Wasted Testing Loss: " << std::get<1>(evaluation) << std::endl;
                std::cout << "ELRALES: " << elralesState << std::endl;
                NNLogger::appendToCSV("cnn.csv", epoch + 1, accuracy, average_loss, std::get<0>(evaluation), std::get<1>(evaluation), elralesState);
                ++epochs; // This ensures the number of successful epochs remains constant
            }
            else if (ELRALES_Retval::END_LEARNING == elralesEvaluation)
            {
                std::cout << "EarlyStopping Epoch " << epoch + 1 << " completed." << std::endl;
                std::cout << "EarlyStopping Training Accuracy: " << accuracy << std::endl;
                std::cout << "EarlyStopping Training Loss: " << average_loss << std::endl;
                std::cout << "EarlyStopping Testing Accuracy: " << std::get<0>(evaluation) << std::endl;
                std::cout << "EarlyStopping Testing Loss: " << std::get<1>(evaluation) << std::endl;
                std::cout << "ELRALES: " << elralesState << std::endl;
                NNLogger::appendToCSV("cnn.csv", epoch + 1, accuracy, average_loss, std::get<0>(evaluation), std::get<1>(evaluation), elralesState);
                break;
            }
            elralesStateMachineTimeLine.push_back(static_cast<ELRALES_StateMachine>(elralesStateMachine));
        }
        else
        {
            std::cout << "Epoch " << epoch + 1 << " completed." << std::endl;
            std::cout << "Training Accuracy: " << accuracy << std::endl;
            std::cout << "Training Loss: " << average_loss << std::endl;
            std::cout << "Testing Accuracy: " << std::get<0>(evaluation) << std::endl;
            std::cout << "Testing Loss: " << std::get<1>(evaluation) << std::endl;
            std::cout << "ELRALES: OFF" << std::endl;
            NNLogger::appendToCSV("cnn.csv", epoch + 1, accuracy, average_loss, std::get<0>(evaluation), std::get<1>(evaluation), "OFF");
        }
    }

    // After the training loop
    std::cout << "Final Evaluation" << std::endl;
    std::tuple<double, double> finalEvaluation = evaluate(imageContainer);
    std::cout << "Final Testing Accuracy: " << std::get<0>(finalEvaluation) << std::endl;
    std::cout << "Final Testing Loss: " << std::get<1>(finalEvaluation) << std::endl;
    std::cout << "Training ended!" << std::endl;
}

std::tuple<double, double> NeuralNetwork::evaluate(const ImageContainer &imageContainer)
{
    if (!lossFunction)
    {
        throw std::runtime_error("Loss function must be set before evaluation.");
    }

    BatchManager batchManager(imageContainer, imageContainer.getTestImages().size(), BatchType::Testing);
    Eigen::Tensor<double, 4> batch_input;
    Eigen::Tensor<int, 2> batch_label;

    double total_loss = 0.0;
    int correct_predictions = 0;
    int num_samples = 0;

    while (batchManager.getNextBatch(batch_input, batch_label))
    {
        Eigen::Tensor<double, 4> predictions = forward(batch_input);

        double batch_loss = lossFunction->compute(predictions, batch_label);
        total_loss += batch_loss * batch_input.dimension(0);

        // Count correct predictions
        for (int i = 0; i < predictions.dimension(0); ++i)
        {
            int predicted_label;
            int true_label;

            if (predictions.dimension(3) == 1) // Binary classification
            {
                predicted_label = (predictions(i, 0, 0, 0) >= 0.5) ? 1 : 0;
                true_label = batch_label(i, 0);
            }
            else // Multi-class classification
            {
                predicted_label = 0;
                double max_value = predictions(i, 0, 0, 0);

                for (int j = 1; j < predictions.dimension(3); ++j)
                {
                    if (predictions(i, 0, 0, j) > max_value)
                    {
                        max_value = predictions(i, 0, 0, j);
                        predicted_label = j;
                    }
                }

                true_label = 0;
                for (int j = 0; j < batch_label.dimension(1); ++j)
                {
                    if (batch_label(i, j) == 1)
                    {
                        true_label = j;
                        break;
                    }
                }
            }

            if (predicted_label == true_label)
            {
                correct_predictions++;
            }
            num_samples++;
        }
    }

    double average_loss = total_loss / num_samples;
    double accuracy = static_cast<double>(correct_predictions) / num_samples;

    return std::make_tuple(accuracy, average_loss);
}

void NeuralNetwork::enableGradientClipping(double value,
                                           GradientClippingMode mode)
{
    clippingMode = mode;
    clipValue = value;
    clippingSet = true;
    if (GradientClippingMode::ENABLED == mode)
    {
        std::cout << "|Gradient Clipping: " << value << "|" << std::endl;
    }
    else
    {
        std::cout << "|Gradient Clipping Disabled|" << std::endl;
    }
}

void NeuralNetwork::enableELRALES(double learning_rate_coef,
                                  int maxSuccessiveEpochFailures,
                                  int maxEpochFailures,
                                  double tolerance,
                                  ELRALES_Mode mode)
{
    this->elralesMode = mode;
    this->elralesSet = true;
    this->elralesStateMachine = ELRALES_StateMachine::NORMAL;
    this->learning_rate_coef = learning_rate_coef;
    this->maxSuccessiveEpochFailures = maxSuccessiveEpochFailures;
    this->maxEpochFailures = maxEpochFailures;
    this->tolerance = tolerance;

    if (ELRALES_Mode::ENABLED == mode)
    {
        this->elrales = std::make_unique<ELRALES>(learning_rate_coef, maxSuccessiveEpochFailures, maxEpochFailures, tolerance, layers);
        std::cout << "|ELRALES Enabled with LRC: " << learning_rate_coef
                  << ", MSEF: " << maxSuccessiveEpochFailures
                  << ", MEF: " << maxEpochFailures
                  << ", TOL: " << tolerance
                  << "|" << std::endl;
    }
    else if (ELRALES_Mode::DISABLED == mode)
    {
        std::cout << "|Epoch Loss Recovery Adaptive Learning Early Stopping Disabled|" << std::endl;
    }
    else
    {
        throw std::runtime_error("Unknown ELRALES mode.");
    }
    elralesStateMachineTimeLine.push_back(static_cast<ELRALES_StateMachine>(elralesStateMachine));
}
