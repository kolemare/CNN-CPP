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

#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork()
{
    this->batchSize = 0;
    this->currentDepth = 3;
    this->trained = false;
    this->compiled = false;
    this->elralesSet = false;
    this->clippingSet = false;
    this->flattenAdded = false;
    this->adamIncrement = false;
    this->logLevel = LogLevel::None;
    this->progressLevel = ProgressLevel::None;
    this->batchMode = BatchMode::ShuffleOnly;
    this->lossType = LossType::MEAN_SQUARED_ERROR;
    outputCSV = "logs/cnn.csv";
}

void NeuralNetwork::setImageSize(const int targetWidth,
                                 const int targetHeight)
{
    inputHeight = targetHeight;
    inputWidth = targetWidth;
    this->hardReset();
}

void NeuralNetwork::setCSVPath(std::string outputCSV)
{
    this->outputCSV = outputCSV;
    std::cout << "Output CSV relative path => " << outputCSV << std::endl;
}

void NeuralNetwork::setLogLevel(LogLevel level)
{
    logLevel = level;
    this->hardReset();
}

void NeuralNetwork::setProgressLevel(ProgressLevel level)
{
    progressLevel = level;
    this->hardReset();
}

void NeuralNetwork::setBatchSize(const int size)
{
    this->batchSize = size;
}

void NeuralNetwork::addConvolutionLayer(int filters,
                                        int kernel_size,
                                        int stride,
                                        int padding,
                                        ConvKernelInitialization kernel_init,
                                        ConvBiasInitialization bias_init)
{
    layers.push_back(std::make_shared<ConvolutionLayer>(filters, kernel_size, stride, padding, kernel_init, bias_init));
    if (LogLevel::LayerSummary == logLevel)
    {
        std::cout << "Added Convolution Layer with " << filters << " filters, kernel size " << kernel_size << ", stride " << stride << ", padding " << padding << std::endl;
    }
    this->hardReset();
}

void NeuralNetwork::addMaxPoolingLayer(int pool_size,
                                       int stride)
{
    layers.push_back(std::make_shared<MaxPoolingLayer>(pool_size, stride));
    if (LogLevel::LayerSummary == logLevel)
    {
        std::cout << "Added Max Pooling Layer with pool size " << pool_size << ", stride " << stride << std::endl;
    }
    this->hardReset();
}

void NeuralNetwork::addAveragePoolingLayer(int pool_size,
                                           int stride)
{
    layers.push_back(std::make_shared<AveragePoolingLayer>(pool_size, stride));
    if (LogLevel::LayerSummary == logLevel)
    {
        std::cout << "Added Average Pooling Layer with pool size " << pool_size << ", stride " << stride << std::endl;
    }
    this->hardReset();
}

void NeuralNetwork::addFlattenLayer()
{
    if (!flattenAdded)
    {
        layers.push_back(std::make_shared<FlattenLayer>());
        flattenAdded = true;
        if (LogLevel::LayerSummary == logLevel)
        {
            std::cout << "Added Flatten Layer" << std::endl;
        }
    }
    else
    {
        std::cerr << "Flatten layer already added." << std::endl;
    }
    this->hardReset();
}

void NeuralNetwork::addFullyConnectedLayer(int output_size,
                                           DenseWeightInitialization weight_init,
                                           DenseBiasInitialization bias_init)
{
    layers.push_back(std::make_shared<FullyConnectedLayer>(output_size, weight_init, bias_init));
    if (LogLevel::LayerSummary == logLevel)
    {
        std::cout << "Added Fully Connected Layer with output size " << output_size << std::endl;
    }
    this->hardReset();
}

void NeuralNetwork::addActivationLayer(ActivationType type)
{
    layers.push_back(std::make_shared<ActivationLayer>(type));
    if (LogLevel::LayerSummary == logLevel)
    {
        std::cout << "Added Activation Layer of type " << static_cast<int>(type) << std::endl;
    }
    this->hardReset();
}

void NeuralNetwork::addBatchNormalizationLayer(double epsilon,
                                               double momentum)
{
    layers.push_back(std::make_shared<BatchNormalizationLayer>(epsilon, momentum));
    if (LogLevel::LayerSummary == logLevel)
    {
        std::cout << "Added Batch Normalization Layer" << std::endl;
    }
    this->hardReset();
}

void NeuralNetwork::setLossFunction(LossType type)
{
    this->lossType = type;
    lossFunction = LossFunction::create(type);
    if (LogLevel::LayerSummary == logLevel)
    {
        std::cout << "Set Loss Function of type " << static_cast<int>(type) << std::endl;
    }
    this->hardReset();
}

void NeuralNetwork::compile(OptimizerType optimizerType,
                            const std::unordered_map<std::string, double> &optimizer_params,
                            bool initializeWeights)
{
    std::unordered_map<std::string, double> default_params;
    BNTarget batchNormTarget = BNTarget::None;

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
        this->adamIncrement = true;
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

    int height = inputHeight;
    int width = inputWidth;
    int input_size = -1;

    if (!lossFunction && initializeWeights)
    {
        // Loss function must be set before compilation
        throw std::runtime_error("Loss function must be set before compiling.");
    }

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

    if (ELRALES_Mode::ENABLED == elralesMode && LearningDecayType::NONE != learningDecayMode)
    {
        throw std::runtime_error("Cannot use both ELRALES and LearningDecay simultaneously.");
    }

    for (size_t i = 0; i < layers.size(); ++i)
    {
        if (auto conv_layer = dynamic_cast<ConvolutionLayer *>(layers[i].get()))
        {
            conv_layer->setInputDepth(currentDepth, initializeWeights);
            currentDepth = conv_layer->getFilters();
            height = (height - conv_layer->getKernelSize() + 2 * conv_layer->getPadding()) / conv_layer->getStride() + 1;
            width = (width - conv_layer->getKernelSize() + 2 * conv_layer->getPadding()) / conv_layer->getStride() + 1;
            conv_layer->setOptimizer(Optimizer::create(optimizerType, default_params));
            batchNormTarget = BNTarget::ConvolutionLayer;
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
            fc_layer->setInputSize(input_size, initializeWeights);
            input_size = fc_layer->getOutputSize();
            fc_layer->setOptimizer(Optimizer::create(optimizerType, default_params));
            batchNormTarget = BNTarget::DenseLayer;
        }
        else if (auto batch_norm_layer = dynamic_cast<BatchNormalizationLayer *>(layers[i].get()))
        {
            if (BNTarget::None == batchNormTarget)
            {
                throw std::runtime_error("Incorrect place for BatchNormalization Layer, check model.");
            }
            batch_norm_layer->setTarget(batchNormTarget);
        }
        else if (dynamic_cast<FlattenLayer *>(layers[i].get()))
        {
            input_size = height * width * currentDepth;
        }
    }
    compiled = true;
}

Eigen::Tensor<double, 4> NeuralNetwork::forward(const Eigen::Tensor<double, 4> &input)
{
    if (LogLevel::LayerSummary == logLevel)
    {
        NNLogger::printTensorSummary(input, "INPUT", PropagationType::FORWARD);
    }

    Eigen::Tensor<double, 4> output = input;
    layerInputs.clear();

    for (size_t i = 0; i < layers.size(); ++i)
    {
        layerInputs.push_back(output);
        output = layers[i]->forward(output);

        if (LogLevel::LayerSummary == logLevel)
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
            else if (dynamic_cast<BatchNormalizationLayer *>(layers[i].get()))
            {
                layerType = "Batch Normalization Layer";
            }

            if (LogLevel::FullTensor == logLevel)
            {
                NNLogger::printFullTensor(output, layerType, PropagationType::FORWARD);
            }
            else if (LogLevel::LayerSummary == logLevel)
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
    if (LogLevel::LayerSummary == logLevel)
    {
        NNLogger::printTensorSummary(d_output, "OUTPUT", PropagationType::BACK);
    }

    if (true == this->adamIncrement)
    {
        Adam::incrementT();
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
        else if (dynamic_cast<BatchNormalizationLayer *>(layers[i].get()))
        {
            layerType = "Batch Normalization Layer";
        }

        d_input = layers[i]->backward(d_input, layerInputs[i], learning_rate);

        if (GradientClippingMode::ENABLED == clippingMode)
        {
            GradientClipping::clipGradients(d_input, clipValue);
        }

        if (LogLevel::FullTensor == logLevel)
        {
            NNLogger::printFullTensor(d_input, layerType, PropagationType::BACK);
        }
        else if (LogLevel::LayerSummary == logLevel)
        {
            NNLogger::printTensorSummary(d_input, layerType, PropagationType::BACK);
        }
    }
}

void NeuralNetwork::train(const ImageContainer &imageContainer,
                          int epochs,
                          int batch_size,
                          double learning_rate)
{
    if (!compiled)
    {
        throw std::runtime_error("Network must be compiled before training.");
    }
    if (!lossFunction)
    {
        throw std::runtime_error("Loss function must be set before training.");
    }

    this->batchSize = batch_size;

    NNLogger::initializeCSV(outputCSV);

    batchManager = std::make_unique<BatchManager>(
        imageContainer,
        batch_size,
        BatchType::Training,
        this->batchMode);

    std::cout << "Training started..." << std::endl;
    auto start = std::chrono::steady_clock::now();
    double current_learning_rate = learning_rate;

    double cumulative_loss = 0.0;
    int total_batches_completed = 0;

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        BatchNormalizationLayer::setMode(BNMode::Training);
        if (LearningDecayType::NONE != learningDecayMode && learningDecay)
        {
            current_learning_rate = learningDecay->computeLearningRate(learning_rate, epoch);
            std::cout << "Learning rate during epoch " << epoch + 1 << ": " << current_learning_rate << std::endl;
        }
        Eigen::Tensor<double, 4> batch_input;
        Eigen::Tensor<int, 2> batch_label;
        int totalBatches = batchManager->getTotalBatches();
        int batchCounter = 0;
        double total_epoch_loss = 0.0;
        int correct_predictions = 0;
        int num_epoch_samples = 0;

        while (batchManager->getNextBatch(batch_input, batch_label))
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

            if (ProgressLevel::None != progressLevel)
            {
                NNLogger::printProgress(epoch, epochs, batchCounter, totalBatches, start, batch_loss, progressLevel, cumulative_loss, total_batches_completed);
            }

            batchCounter++;
        }

        double average_loss = total_epoch_loss / num_epoch_samples;
        double accuracy = static_cast<double>(correct_predictions) / num_epoch_samples;

        std::cout << "Evaluating..." << std::endl;
        std::tuple<double, double> evaluation = evaluate(imageContainer);
        this->makeSinglePredictions(imageContainer);

        if (ELRALES_Mode::ENABLED == elralesMode)
        {
            ELRALES_Retval elralesEvaluation = elrales->updateState(average_loss, layers, current_learning_rate, elralesStateMachine);
            std::string elralesState = toString(elralesStateMachine);

            if (ELRALES_Retval::SUCCESSFUL_EPOCH == elralesEvaluation)
            {
                std::cout << "Epoch " << epoch + 1 << " completed." << std::endl;
                std::cout << "Training Accuracy: " << accuracy << std::endl;
                std::cout << "Training Loss: " << average_loss << std::endl;
                std::cout << "Validation Accuracy: " << std::get<0>(evaluation) << std::endl;
                std::cout << "Validation Loss: " << std::get<1>(evaluation) << std::endl;
                std::cout << "ELRALES: " << elralesState << std::endl;
                NNLogger::appendToCSV(outputCSV, epoch + 1, accuracy, average_loss, std::get<0>(evaluation), std::get<1>(evaluation), elralesState);
            }
            else if (ELRALES_Retval::WASTED_EPOCH == elralesEvaluation)
            {
                std::cout << "Wasted Epoch " << epoch + 1 << " completed." << std::endl;
                std::cout << "Wasted Training Accuracy: " << accuracy << std::endl;
                std::cout << "Wasted Training Loss: " << average_loss << std::endl;
                std::cout << "Wasted Validation Accuracy: " << std::get<0>(evaluation) << std::endl;
                std::cout << "Wasted Validation Loss: " << std::get<1>(evaluation) << std::endl;
                std::cout << "ELRALES: " << elralesState << std::endl;
                NNLogger::appendToCSV(outputCSV, epoch + 1, accuracy, average_loss, std::get<0>(evaluation), std::get<1>(evaluation), elralesState);
                ++epochs; // This ensures the number of successful epochs remains constant
            }
            else if (ELRALES_Retval::END_LEARNING == elralesEvaluation)
            {
                std::cout << "EarlyStopping Epoch " << epoch + 1 << " completed." << std::endl;
                std::cout << "EarlyStopping Training Accuracy: " << accuracy << std::endl;
                std::cout << "EarlyStopping Training Loss: " << average_loss << std::endl;
                std::cout << "EarlyStopping Validation Accuracy: " << std::get<0>(evaluation) << std::endl;
                std::cout << "EarlyStopping Validation Loss: " << std::get<1>(evaluation) << std::endl;
                std::cout << "ELRALES: " << elralesState << std::endl;
                NNLogger::appendToCSV(outputCSV, epoch + 1, accuracy, average_loss, std::get<0>(evaluation), std::get<1>(evaluation), elralesState);
                break;
            }
            elralesStateMachineTimeLine.push_back(static_cast<ELRALES_StateMachine>(elralesStateMachine));
        }
        else
        {
            std::cout << "Epoch " << epoch + 1 << " completed." << std::endl;
            std::cout << "Training Accuracy: " << accuracy << std::endl;
            std::cout << "Training Loss: " << average_loss << std::endl;
            std::cout << "Validation Accuracy: " << std::get<0>(evaluation) << std::endl;
            std::cout << "Validation Loss: " << std::get<1>(evaluation) << std::endl;
            std::cout << "ELRALES: OFF" << std::endl;
            NNLogger::appendToCSV(outputCSV, epoch + 1, accuracy, average_loss, std::get<0>(evaluation), std::get<1>(evaluation), "OFF");
        }
    }

    // After the training loop
    std::cout << std::endl;
    std::cout << "Training ended!" << std::endl;
    std::cout << std::endl;
    trained = true;
}

std::tuple<double, double> NeuralNetwork::evaluate(const ImageContainer &imageContainer)
{
    if (!compiled)
    {
        throw std::runtime_error("Network must be compiled before evaluation.");
    }
    if (!lossFunction)
    {
        throw std::runtime_error("Loss function must be set before evaluation.");
    }

    // Set the Batch Normalization mode to Inference
    BatchNormalizationLayer::setMode(BNMode::Inference);

    BatchManager batchManager = BatchManager(
        imageContainer,
        imageContainer.getTestImages().size(),
        BatchType::Testing,
        this->batchMode);
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

std::unordered_map<std::string, double>
NeuralNetwork::thoroughEvaluation(const ImageContainer &imageContainer)
{
    if (!compiled)
        throw std::runtime_error("Network must be compiled before evaluation.");
    if (!lossFunction)
        throw std::runtime_error("Loss function must be set before evaluation.");

    BatchNormalizationLayer::setMode(BNMode::Inference);

    batchManager = std::make_unique<BatchManager>(
        imageContainer,
        static_cast<int>(imageContainer.getTestImages().size()),
        BatchType::Testing,
        this->batchMode);

    Eigen::Tensor<double, 4> batch_input;
    Eigen::Tensor<int, 2> batch_label;

    double total_loss = 0.0;
    int64_t num_samples = 0;
    int64_t correct = 0;

    bool confusionInit = false;
    int C = 0;
    Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic> confusion;

    while (batchManager->getNextBatch(batch_input, batch_label))
    {
        Eigen::Tensor<double, 4> preds = forward(batch_input);

        const int N = static_cast<int>(preds.dimension(0));
        const int predC = static_cast<int>(preds.dimension(3));
        const int labelC = static_cast<int>(batch_label.dimension(1));

        if (N <= 0)
            continue;

        // ---------- CONFUSION MATRIX INIT ----------
        if (!confusionInit)
        {
            C = labelC;

            // If you're doing BCE with [N,1] labels, C should be 2 for metrics.
            // But in your pipeline BatchManager produces one-hot => labelC==2, so this is normally fine.
            if (C <= 0)
                throw std::runtime_error("thoroughEvaluation: invalid class count from labels.");

            confusion = Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic>::Zero(C, C);
            confusionInit = true;
        }

        // ---------- LOSS (branch by lossType) ----------
        double batch_loss = 0.0;

        if (lossType == LossType::CATEGORICAL_CROSS_ENTROPY)
        {
            // Your CCE expects:
            // preds: [N,1,1,C], targets: [N,C] one-hot
            batch_loss = lossFunction->compute(preds, batch_label);
        }
        else if (lossType == LossType::BINARY_CROSS_ENTROPY)
        {
            // Your BCE expects:
            // preds: [N,1,1,1], targets: [N,1] with values 0/1

            if (predC != 1)
            {
                throw std::runtime_error(
                    "thoroughEvaluation: BCE selected but model output predC != 1. "
                    "Binary head should output [N,1,1,1].");
            }

            // Case A: batch_label is already [N,1]
            if (labelC == 1)
            {
                batch_loss = lossFunction->compute(preds, batch_label);
            }
            // Case B: batch_label is one-hot [N,2] (your BatchManager does this)
            else if (labelC == 2)
            {
                Eigen::Tensor<int, 2> bceTargets(N, 1);

                // IMPORTANT: define what "positive" means.
                // With one-hot, simplest convention:
                // class 0 => y=0, class 1 => y=1
                // (this matches your predicted_label thresholding which returns 0 or 1)
                for (int i = 0; i < N; ++i)
                {
                    // argmax on [2] -> {0,1}
                    int best = 0;
                    int bestVal = batch_label(i, 0);
                    for (int j = 1; j < 2; ++j)
                    {
                        const int v = batch_label(i, j);
                        if (v > bestVal)
                        {
                            bestVal = v;
                            best = j;
                        }
                    }
                    bceTargets(i, 0) = best; // 0 or 1
                }

                batch_loss = lossFunction->compute(preds, bceTargets); // uses your original BCE implementation
            }
            else
            {
                throw std::runtime_error(
                    "thoroughEvaluation: BCE selected but labelC is neither 1 nor 2.");
            }
        }
        else
        {
            throw std::runtime_error("thoroughEvaluation: Unsupported lossType for this evaluation.");
        }

        total_loss += batch_loss * static_cast<double>(N);

        // ---------- METRICS UPDATE (confusion + accuracy) ----------
        for (int i = 0; i < N; ++i)
        {
            // TRUE LABEL:
            // - if one-hot: argmax
            // - if [N,1] binary: use 0/1 directly
            int true_label = 0;
            if (labelC == 1)
            {
                true_label = batch_label(i, 0);
            }
            else
            {
                int best = 0;
                int bestVal = batch_label(i, 0);
                for (int j = 1; j < C; ++j)
                {
                    const int v = batch_label(i, j);
                    if (v > bestVal)
                    {
                        bestVal = v;
                        best = j;
                    }
                }
                true_label = best;
            }

            // PRED LABEL:
            int predicted_label = 0;
            if (predC == 1)
            {
                predicted_label = (preds(i, 0, 0, 0) >= 0.5) ? 1 : 0;
            }
            else
            {
                int best = 0;
                double bestVal = preds(i, 0, 0, 0);
                for (int j = 1; j < predC; ++j)
                {
                    const double v = preds(i, 0, 0, j);
                    if (v > bestVal)
                    {
                        bestVal = v;
                        best = j;
                    }
                }
                predicted_label = best;
            }

            if (predicted_label == true_label)
                ++correct;

            if (true_label >= 0 && true_label < C &&
                predicted_label >= 0 && predicted_label < C)
            {
                confusion(true_label, predicted_label)++;
            }

            ++num_samples;
        }
    }

    if (!confusionInit || num_samples == 0)
        throw std::runtime_error("thoroughEvaluation: no samples were evaluated.");

    const double accuracy = static_cast<double>(correct) / static_cast<double>(num_samples);
    const double avg_loss = total_loss / static_cast<double>(num_samples);

    // ---------- PER-CLASS PRECISION/RECALL/F1 ----------
    std::vector<double> precision(C, 0.0), recall(C, 0.0), f1(C, 0.0), support(C, 0.0);

    long long totalTP = 0, totalFP = 0, totalFN = 0;

    for (int k = 0; k < C; ++k)
    {
        const long long TP = confusion(k, k);

        long long rowSum = 0;
        long long colSum = 0;
        for (int j = 0; j < C; ++j)
        {
            rowSum += confusion(k, j); // true=k predicted=j
            colSum += confusion(j, k); // true=j predicted=k
        }

        const long long FN = rowSum - TP;
        const long long FP = colSum - TP;

        support[k] = static_cast<double>(rowSum);

        const double p = (TP + FP) > 0 ? static_cast<double>(TP) / static_cast<double>(TP + FP) : 0.0;
        const double r = (TP + FN) > 0 ? static_cast<double>(TP) / static_cast<double>(TP + FN) : 0.0;
        const double f = (p + r) > 0 ? (2.0 * p * r) / (p + r) : 0.0;

        precision[k] = p;
        recall[k] = r;
        f1[k] = f;

        totalTP += TP;
        totalFP += FP;
        totalFN += FN;
    }

    // ---------- MACRO ----------
    double macro_precision = 0.0, macro_recall = 0.0, macro_f1 = 0.0;
    for (int k = 0; k < C; ++k)
    {
        macro_precision += precision[k];
        macro_recall += recall[k];
        macro_f1 += f1[k];
    }
    macro_precision = (C > 0) ? macro_precision / static_cast<double>(C) : 0.0;
    macro_recall = (C > 0) ? macro_recall / static_cast<double>(C) : 0.0;
    macro_f1 = (C > 0) ? macro_f1 / static_cast<double>(C) : 0.0;

    // Balanced accuracy = macro recall
    const double balanced_accuracy = macro_recall;

    // ---------- MICRO ----------
    const double micro_precision = (totalTP + totalFP) > 0
                                       ? static_cast<double>(totalTP) / static_cast<double>(totalTP + totalFP)
                                       : 0.0;

    const double micro_recall = (totalTP + totalFN) > 0
                                    ? static_cast<double>(totalTP) / static_cast<double>(totalTP + totalFN)
                                    : 0.0;

    const double micro_f1 = (micro_precision + micro_recall) > 0
                                ? (2.0 * micro_precision * micro_recall) / (micro_precision + micro_recall)
                                : 0.0;

    // ---------- PACK INTO HASHMAP ----------
    std::unordered_map<std::string, double> metrics;
    metrics.reserve(static_cast<size_t>(16 + 4 * C + C * C));

    metrics["samples"] = static_cast<double>(num_samples);

    metrics["loss/avg"] = avg_loss; // BCE or CCE depending on lossType
    metrics["accuracy/top1"] = accuracy;
    metrics["accuracy/balanced"] = balanced_accuracy;

    metrics["precision/macro"] = macro_precision;
    metrics["recall/macro"] = macro_recall;
    metrics["f1/macro"] = macro_f1;

    metrics["precision/micro"] = micro_precision;
    metrics["recall/micro"] = micro_recall;
    metrics["f1/micro"] = micro_f1;

    for (int k = 0; k < C; ++k)
    {
        std::string cname;
        try
        {
            cname = batchManager->getCategoryName(k);
        }
        catch (...)
        {
            cname = "class_" + std::to_string(k);
        }

        metrics["class/" + cname + "/precision"] = precision[k];
        metrics["class/" + cname + "/recall"] = recall[k];
        metrics["class/" + cname + "/f1"] = f1[k];
        metrics["class/" + cname + "/support"] = support[k];
    }

    for (int t = 0; t < C; ++t)
    {
        std::string tname;
        try
        {
            tname = batchManager->getCategoryName(t);
        }
        catch (...)
        {
            tname = "class_" + std::to_string(t);
        }

        for (int p = 0; p < C; ++p)
        {
            std::string pname;
            try
            {
                pname = batchManager->getCategoryName(p);
            }
            catch (...)
            {
                pname = "class_" + std::to_string(p);
            }

            metrics["confusion/true=" + tname + "/pred=" + pname] =
                static_cast<double>(confusion(t, p));
        }
    }

    // ---------- PRINTS ----------
    std::cout << "\n========== Thorough Evaluation ==========\n";
    std::cout << "Samples: " << num_samples << "\n";
    std::cout << "Loss (avg): " << avg_loss
              << ((lossType == LossType::BINARY_CROSS_ENTROPY) ? " (BCE)" : (lossType == LossType::CATEGORICAL_CROSS_ENTROPY) ? " (CCE)"
                                                                                                                              : "")
              << "\n";
    std::cout << "Top-1 Accuracy: " << accuracy << "\n";
    std::cout << "Balanced Accuracy: " << balanced_accuracy << "\n";
    std::cout << "Macro Precision/Recall/F1: "
              << macro_precision << " / " << macro_recall << " / " << macro_f1 << "\n";
    std::cout << "Micro Precision/Recall/F1: "
              << micro_precision << " / " << micro_recall << " / " << micro_f1 << "\n";

    std::cout << "\n---- Per-class metrics ----\n";
    std::cout << "Class\tSupport\tPrecision\tRecall\tF1\n";
    for (int k = 0; k < C; ++k)
    {
        std::string cname;
        try
        {
            cname = batchManager->getCategoryName(k);
        }
        catch (...)
        {
            cname = "class_" + std::to_string(k);
        }

        std::cout << cname << "\t"
                  << static_cast<long long>(support[k]) << "\t"
                  << precision[k] << "\t"
                  << recall[k] << "\t"
                  << f1[k] << "\n";
    }

    std::cout << "\n---- Confusion Matrix (rows=true, cols=pred) ----\n";
    std::cout << "\t";
    for (int p = 0; p < C; ++p)
    {
        std::string pname;
        try
        {
            pname = batchManager->getCategoryName(p);
        }
        catch (...)
        {
            pname = "class_" + std::to_string(p);
        }
        std::cout << pname << "\t";
    }
    std::cout << "\n";

    for (int t = 0; t < C; ++t)
    {
        std::string tname;
        try
        {
            tname = batchManager->getCategoryName(t);
        }
        catch (...)
        {
            tname = "class_" + std::to_string(t);
        }

        std::cout << tname << "\t";
        for (int p = 0; p < C; ++p)
            std::cout << confusion(t, p) << "\t";
        std::cout << "\n";
    }
    std::cout << "========================================\n\n";

    return metrics;
}

void NeuralNetwork::thoroughEvaluationToJson(
    const ImageContainer &imageContainer,
    const std::string &jsonPath,
    bool pretty)
{
    // 1) Compute metrics
    std::unordered_map<std::string, double> metrics =
        thoroughEvaluation(imageContainer);

    // 2) Stable ordering (useful for diffs & reproducibility)
    std::vector<std::string> keys;
    keys.reserve(metrics.size());
    for (const auto &kv : metrics)
        keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());

    // 3) Open output file
    std::ofstream out(jsonPath, std::ios::out | std::ios::trunc);
    if (!out.is_open())
        throw std::runtime_error(
            "thoroughEvaluationToJson: failed to open file: " + jsonPath);

    const std::string indent = pretty ? "  " : "";
    const std::string nl = pretty ? "\n" : "";
    const std::string sp = pretty ? " " : "";

    out << "{" << nl;
    out << std::setprecision(std::numeric_limits<double>::max_digits10);

    for (size_t i = 0; i < keys.size(); ++i)
    {
        const std::string &k = keys[i];
        const double v = metrics.at(k);

        out << indent
            << "\"" << jsonEscape(k) << "\":" << sp;

        // JSON has no NaN/Inf → write null
        if (std::isfinite(v))
            out << v;
        else
            out << "null";

        if (i + 1 < keys.size())
            out << ",";

        out << nl;
    }

    out << "}" << nl;
    out.close();

    if (!out)
        throw std::runtime_error(
            "thoroughEvaluationToJson: write failed for file: " + jsonPath);
}

void NeuralNetwork::makeSinglePredictions(const ImageContainer &imageContainer)
{
    if (!compiled)
    {
        throw std::runtime_error("Network must be compiled before making single predictions.");
    }
    // if (!trained)
    // {
    //     throw std::runtime_error("Network must be trained before making single predictions.");
    // }
    if (0 == batchSize)
    {
        throw std::runtime_error("Bad batch size, unknown error.");
    }

    // Set the Batch Normalization mode to Inference
    BatchNormalizationLayer::setMode(BNMode::Inference);

    // Create a batch manager for single prediction
    BatchManager batchManager(imageContainer, batchSize, BatchType::Testing, this->batchMode);
    batchManager.loadSinglePredictionBatch();

    // Process each batch of single prediction images
    while (true)
    {
        Eigen::Tensor<double, 4> batchImages;
        Eigen::Tensor<int, 2> batchLabels;

        // Get a batch of single prediction images and their names
        std::vector<std::string> imageNames = batchManager.getSinglePredictionBatch(batchImages, batchLabels);

        if (imageNames.empty())
        {
            break; // No more images to process
        }

        // Perform forward pass to get predictions
        Eigen::Tensor<double, 4> predictions = forward(batchImages);

        for (int i = 0; i < imageNames.size(); ++i)
        {
            const std::string &imageName = imageNames[i];

            if (imageName.empty())
            {
                continue; // Skip empty slots
            }

            std::string predictedCategory;
            double confidence = 0.0;

            if (predictions.dimension(3) == 1) // Binary classification
            {
                double score = predictions(i, 0, 0, 0);

                // Interpret the prediction
                predictedCategory = score >= 0.5 ? batchManager.getCategoryName(0) : batchManager.getCategoryName(1);
                confidence = score >= 0.5 ? score * 100.0 : (1 - score) * 100.0;
            }
            else // Multi-class classification
            {
                int predictedLabel = 0;
                double maxConfidence = predictions(i, 0, 0, 0);

                for (int j = 1; j < predictions.dimension(3); ++j)
                {
                    if (predictions(i, 0, 0, j) > maxConfidence)
                    {
                        maxConfidence = predictions(i, 0, 0, j);
                        predictedLabel = j;
                    }
                }

                predictedCategory = batchManager.getCategoryName(predictedLabel);
                confidence = maxConfidence * 100.0;
            }

            std::cout << "Prediction for \"" << imageName << "\" is \"" << predictedCategory
                      << "\" with confidence " << confidence << "%." << std::endl;
        }
    }
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
    this->hardReset();
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

    if (LearningDecayType::NONE != learningDecayMode && ELRALES_Mode::ENABLED == elralesMode)
    {
        throw std::runtime_error("Cannot use ELRALES when LearningDecay is enabled.");
    }

    if (ELRALES_Mode::ENABLED == mode)
    {
        this->elrales = std::make_unique<ELRALES>(learning_rate_coef, maxSuccessiveEpochFailures, maxEpochFailures, tolerance, layers);
        std::cout << "|ELRALES Enabled with LRC: " << learning_rate_coef
                  << ", MSEF: " << maxSuccessiveEpochFailures
                  << ", MEF: " << maxEpochFailures
                  << ", TOL: " << tolerance
                  << "|" << std::endl;
    }
    if (ELRALES_Mode::ENABLED != mode && ELRALES_Mode::DISABLED != mode)
    {
        throw std::runtime_error("Unknown ELRALES mode.");
    }
    elralesStateMachineTimeLine.push_back(static_cast<ELRALES_StateMachine>(elralesStateMachine));
    this->hardReset();
}

void NeuralNetwork::enableLearningDecay(LearningDecayType decayType,
                                        const std::unordered_map<std::string, double> &params)
{
    if (ELRALES_Mode::ENABLED == elralesMode)
    {
        throw std::runtime_error("Cannot use LearningDecay when ELRALES is enabled.");
    }

    learningDecayMode = decayType;
    learningDecay = std::make_unique<LearningDecay>(decayType, params);
    std::cout << "|Learning Decay Enabled with Type: " << toString(decayType) << "|" << std::endl;
    this->hardReset();
}

void NeuralNetwork::setBatchMode(BatchMode mode)
{
    this->batchMode = mode;
}

void NeuralNetwork::hardReset()
{
    this->compiled = false;
    this->trained = false;
    this->currentDepth = 3;
    this->batchSize = 0;
}

void NeuralNetwork::saveModel(std::string path)
{
    std::vector<std::string> categories;
    if (this->batchManager)
    {
        categories = this->batchManager->getCategories();
    }

    ModelSerializer::exportOnnx(path, layers, categories);
}

void NeuralNetwork::loadModel(const std::string &onnxPath,
                              OptimizerType optimizerType,
                              const std::unordered_map<std::string, double> &optimizer_params)
{
    std::vector<std::string> classNames;
    ModelSerializer::importOnnx(onnxPath, layers, classNames);

    flattenAdded = false;
    for (const auto &l : layers)
        if (dynamic_cast<FlattenLayer *>(l.get()))
            flattenAdded = true;

    // To implement
    this->batchManager->updateClassNames(classNames);

    compile(optimizerType, optimizer_params, false);

    trained = true;
}

std::string NeuralNetwork::jsonEscape(const std::string &s)
{
    std::string out;
    out.reserve(s.size() + 8);

    for (unsigned char c : s)
    {
        switch (c)
        {
        case '\"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20)
            {
                std::ostringstream oss;
                oss << "\\u00"
                    << std::hex << std::uppercase
                    << std::setw(2) << std::setfill('0')
                    << static_cast<int>(c);
                out += oss.str();
            }
            else
            {
                out += static_cast<char>(c);
            }
        }
    }
    return out;
}