#include "NeuralNetwork.hpp"
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <filesystem>

NeuralNetwork::NeuralNetwork() : flattenAdded(false), currentDepth(3), logLevel(LogLevel::None), progressLevel(ProgressLevel::None) {}

void NeuralNetwork::setImageSize(const int targetWidth, const int targetHeight)
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

void NeuralNetwork::addConvolutionLayer(int filters, int kernel_size, int stride, int padding, ConvKernelInitialization kernel_init, ConvBiasInitialization bias_init)
{
    layers.push_back(std::make_shared<ConvolutionLayer>(filters, kernel_size, stride, padding, kernel_init, bias_init));
    if (logLevel == LogLevel::All)
    {
        std::cout << "Added Convolution Layer with " << filters << " filters, kernel size " << kernel_size << ", stride " << stride << ", padding " << padding << std::endl;
    }
}

void NeuralNetwork::addMaxPoolingLayer(int pool_size, int stride)
{
    layers.push_back(std::make_shared<MaxPoolingLayer>(pool_size, stride));
    if (logLevel == LogLevel::All)
    {
        std::cout << "Added Max Pooling Layer with pool size " << pool_size << ", stride " << stride << std::endl;
    }
}

void NeuralNetwork::addAveragePoolingLayer(int pool_size, int stride)
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

void NeuralNetwork::addFullyConnectedLayer(int output_size, DenseWeightInitialization weight_init, DenseBiasInitialization bias_init)
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
        std::cout << "Added Activation Layer of type " << type << std::endl;
    }
}

void NeuralNetwork::setLossFunction(LossType type)
{
    lossFunction = LossFunction::create(type);
    if (logLevel == LogLevel::All)
    {
        std::cout << "Set Loss Function of type " << (int)type << std::endl;
    }
}

void NeuralNetwork::compile(Optimizer::Type optimizerType, const std::unordered_map<std::string, double> &optimizer_params)
{
    std::unordered_map<std::string, double> default_params;

    switch (optimizerType)
    {
    case Optimizer::Type::SGD:
        // No parameters needed for SGD, empty map
        break;
    case Optimizer::Type::SGDWithMomentum:
        default_params = {{"momentum", 0.9}};
        break;
    case Optimizer::Type::Adam:
        default_params = {{"beta1", 0.9}, {"beta2", 0.999}, {"epsilon", 1e-7}};
        break;
    case Optimizer::Type::RMSprop:
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
        std::cout << "-------------------------------------------------------------------" << std::endl;
        printTensorSummary(input, "INPUT", PropagationType::FORWARD);
        std::cout << "-------------------------------------------------------------------" << std::endl;
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

            std::cout << "-------------------------------------------------------------------" << std::endl;
            if (logLevel == LogLevel::All)
            {
                printFullTensor(output, layerType, PropagationType::FORWARD);
            }
            else
            {
                printTensorSummary(output, layerType, PropagationType::FORWARD);
            }
            std::cout << "-------------------------------------------------------------------" << std::endl;
        }
    }

    return output;
}

void NeuralNetwork::printFullTensor(const Eigen::Tensor<double, 4> &tensor, const std::string &layerType, PropagationType propagationType)
{
    switch (propagationType)
    {
    case PropagationType::FORWARD:
        std::cout << layerType << " Forward pass:\n";
        break;

    case PropagationType::BACK:
        std::cout << layerType << " Back pass:\n";
        break;

    default:
        break;
    }
    // Add logic to print tensor
    std::cout << tensor << std::endl;
}

void NeuralNetwork::printFullTensor(const Eigen::Tensor<double, 2> &tensor, const std::string &layerType, PropagationType propagationType)
{
    switch (propagationType)
    {
    case PropagationType::FORWARD:
        std::cout << layerType << " Forward pass:\n";
        break;

    case PropagationType::BACK:
        std::cout << layerType << " Back pass:\n";
        break;

    default:
        break;
    }
    // Add logic to print tensor
    std::cout << tensor << std::endl;
}

void NeuralNetwork::printTensorSummary(const Eigen::Tensor<double, 4> &tensor, const std::string &layerType, PropagationType propagationType)
{
    std::vector<double> tensorVec(tensor.size());
    std::copy(tensor.data(), tensor.data() + tensor.size(), tensorVec.begin());

    double mean = std::accumulate(tensorVec.begin(), tensorVec.end(), 0.0) / tensorVec.size();

    std::vector<double> diff(tensorVec.size());
    std::transform(tensorVec.begin(), tensorVec.end(), diff.begin(), [mean](double x)
                   { return x - mean; });

    std::vector<double> squaredDiff(diff.size());
    std::transform(diff.begin(), diff.end(), squaredDiff.begin(), [](double x)
                   { return x * x; });

    double variance = std::accumulate(squaredDiff.begin(), squaredDiff.end(), 0.0) / squaredDiff.size();
    double stddev = std::sqrt(variance);

    double minCoeff = *std::min_element(tensorVec.begin(), tensorVec.end());
    double maxCoeff = *std::max_element(tensorVec.begin(), tensorVec.end());

    double zeroPercentage = std::count(tensorVec.begin(), tensorVec.end(), 0.0) / static_cast<double>(tensorVec.size()) * 100.0;
    double negativeCount = std::count_if(tensorVec.begin(), tensorVec.end(), [](double x)
                                         { return x < 0.0; });
    double positiveCount = std::count_if(tensorVec.begin(), tensorVec.end(), [](double x)
                                         { return x > 0.0; });

    switch (propagationType)
    {
    case PropagationType::FORWARD:
        std::cout << layerType << " Forward pass summary:\n";
        break;

    case PropagationType::BACK:
        std::cout << layerType << " Back pass summary:\n";
        break;

    default:
        break;
    }

    std::cout << "Dimensions: " << tensor.dimension(0) << "x" << tensor.dimension(1) << "x" << tensor.dimension(2) << "x" << tensor.dimension(3) << "\n";
    std::cout << "Mean: " << mean << "\n";
    std::cout << "Standard Deviation: " << stddev << "\n";
    std::cout << "Min: " << minCoeff << "\n";
    std::cout << "Max: " << maxCoeff << "\n";
    std::cout << "Percentage of Zeros: " << zeroPercentage << "%\n";
    std::cout << "Number of Negative Values: " << negativeCount << "\n";
    std::cout << "Number of Positive Values: " << positiveCount << "\n\n";
}

void NeuralNetwork::printTensorSummary(const Eigen::Tensor<double, 2> &tensor, const std::string &layerType, PropagationType propagationType)
{
    std::vector<double> tensorVec(tensor.size());
    std::copy(tensor.data(), tensor.data() + tensor.size(), tensorVec.begin());

    double mean = std::accumulate(tensorVec.begin(), tensorVec.end(), 0.0) / tensorVec.size();

    std::vector<double> diff(tensorVec.size());
    std::transform(tensorVec.begin(), tensorVec.end(), diff.begin(), [mean](double x)
                   { return x - mean; });

    std::vector<double> squaredDiff(diff.size());
    std::transform(diff.begin(), diff.end(), squaredDiff.begin(), [](double x)
                   { return x * x; });

    double variance = std::accumulate(squaredDiff.begin(), squaredDiff.end(), 0.0) / squaredDiff.size();
    double stddev = std::sqrt(variance);

    double minCoeff = *std::min_element(tensorVec.begin(), tensorVec.end());
    double maxCoeff = *std::max_element(tensorVec.begin(), tensorVec.end());

    double zeroPercentage = std::count(tensorVec.begin(), tensorVec.end(), 0.0) / static_cast<double>(tensorVec.size()) * 100.0;
    double negativeCount = std::count_if(tensorVec.begin(), tensorVec.end(), [](double x)
                                         { return x < 0.0; });
    double positiveCount = std::count_if(tensorVec.begin(), tensorVec.end(), [](double x)
                                         { return x > 0.0; });

    switch (propagationType)
    {
    case PropagationType::FORWARD:
        std::cout << layerType << " Forward pass summary:\n";
        break;

    case PropagationType::BACK:
        std::cout << layerType << " Back pass summary:\n";
        break;

    default:
        break;
    }

    std::cout << "Dimensions: " << tensor.dimension(0) << "x" << tensor.dimension(1) << "\n";
    std::cout << "Mean: " << mean << "\n";
    std::cout << "Standard Deviation: " << stddev << "\n";
    std::cout << "Min: " << minCoeff << "\n";
    std::cout << "Max: " << maxCoeff << "\n";
    std::cout << "Percentage of Zeros: " << zeroPercentage << "%\n";
    std::cout << "Number of Negative Values: " << negativeCount << "\n";
    std::cout << "Number of Positive Values: " << positiveCount << "\n\n";
}

void NeuralNetwork::backward(const Eigen::Tensor<double, 4> &d_output, double learning_rate)
{
    if (logLevel == LogLevel::All || logLevel == LogLevel::LayerOutputs)
    {
        std::cout << "-------------------------------------------------------------------" << std::endl;
        printTensorSummary(d_output, "OUTPUT", PropagationType::BACK);
        std::cout << "-------------------------------------------------------------------" << std::endl;
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

        if (logLevel == LogLevel::All || logLevel == LogLevel::LayerOutputs)
        {
            std::cout << "-------------------------------------------------------------------" << std::endl;
            if (logLevel == LogLevel::All)
            {
                printFullTensor(d_input, layerType, PropagationType::BACK);
            }
            else
            {
                printTensorSummary(d_input, layerType, PropagationType::BACK);
            }
            std::cout << "-------------------------------------------------------------------" << std::endl;
        }
    }
}

void NeuralNetwork::printProgress(int epoch, int epochs, int batch, int totalBatches, std::chrono::steady_clock::time_point trainingStart, double currentBatchLoss)
{
    if (progressLevel == ProgressLevel::None)
    {
        return;
    }

    static double cumulative_loss = 0.0;
    static int total_batches_completed = 0;

    cumulative_loss += currentBatchLoss;
    total_batches_completed++;

    double average_loss = cumulative_loss / total_batches_completed;

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - trainingStart).count();
    double progress = static_cast<double>(batch + 1) / totalBatches;
    int barWidth = 50;
    int pos = static_cast<int>(barWidth * progress);

    double timePerBatch = elapsed / static_cast<double>(total_batches_completed);
    double remainingTime = (totalBatches * epochs - total_batches_completed) * timePerBatch;

    double overallProgress = static_cast<double>(epoch * totalBatches + batch + 1) / (totalBatches * epochs);
    int overallPos = static_cast<int>(barWidth * overallProgress);

    std::ostringstream oss;
    oss << "-------------------------------------------------------------------" << std::endl;

    std::ostringstream epochProgress;
    epochProgress << "Epoch " << epoch + 1 << "/" << epochs << " | Batch " << batch + 1 << "/" << totalBatches << " [";
    for (int i = 0; i < barWidth; ++i)
    {
        if (i < pos)
            epochProgress << "=";
        else if (i == pos)
            epochProgress << ">";
        else
            epochProgress << " ";
    }
    epochProgress << "] " << int(progress * 100.0) << "%\n";

    oss << epochProgress.str();

    std::ostringstream overallProgressLabel;
    overallProgressLabel << "Overall Progress: ";
    int labelLength = overallProgressLabel.str().length();

    // Dynamically adjust the overall progress label length to match epoch progress length
    std::string epochPrefix = "Epoch " + std::to_string(epoch + 1) + "/" + std::to_string(epochs) + " | Batch " + std::to_string(batch + 1) + "/" + std::to_string(totalBatches);
    while (overallProgressLabel.str().length() < epochPrefix.length())
    {
        overallProgressLabel << " ";
    }

    overallProgressLabel << " [";
    for (int i = 0; i < barWidth; ++i)
    {
        if (i < overallPos)
            overallProgressLabel << "=";
        else if (i == overallPos)
            overallProgressLabel << ">";
        else
            overallProgressLabel << " ";
    }
    overallProgressLabel << "] " << int(overallProgress * 100.0) << "%\n";

    oss << overallProgressLabel.str();

    if (progressLevel == ProgressLevel::ProgressTime || progressLevel == ProgressLevel::Time)
    {
        auto formatTime = [](double seconds)
        {
            int h = static_cast<int>(seconds) / 3600;
            int m = (static_cast<int>(seconds) % 3600) / 60;
            int s = static_cast<int>(seconds) % 60;
            std::ostringstream oss;
            oss << std::setw(2) << std::setfill('0') << h << "H:"
                << std::setw(2) << std::setfill('0') << m << "M:"
                << std::setw(2) << std::setfill('0') << s << "S";
            return oss.str();
        };

        oss << "Elapsed: " << formatTime(elapsed) << "\n";
        oss << "ETA: " << formatTime(remainingTime) << "\n";
    }
    oss << "Loss: " << average_loss << "\n";
    oss << "-------------------------------------------------------------------" << std::endl;
    std::cout << oss.str() << std::flush;
}

void NeuralNetwork::train(const ImageContainer &imageContainer, int epochs, int batch_size, double learning_rate)
{
    if (!lossFunction)
    {
        throw std::runtime_error("Loss function must be set before training.");
    }

    BatchManager batchManager(imageContainer, batch_size, BatchManager::BatchType::Training);
    std::cout << "Training started..." << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        Eigen::Tensor<double, 4> batch_input;
        Eigen::Tensor<int, 2> batch_label;
        int totalBatches = batchManager.getTotalBatches();
        auto start = std::chrono::steady_clock::now();
        int batchCounter = 0;
        double total_loss = 0.0;
        int correct_predictions = 0;
        int num_samples = 0;

        while (batchManager.getNextBatch(batch_input, batch_label))
        {
            // Forward pass
            Eigen::Tensor<double, 4> predictions = forward(batch_input);

            // Compute loss
            double batch_loss = lossFunction->compute(predictions, batch_label);
            total_loss += batch_loss;

            // Count correct predictions
            for (int i = 0; i < predictions.dimension(0); ++i)
            {
                int predicted_label = (predictions(i, 0, 0, 0) >= 0.5) ? 1 : 0;
                int true_label = batch_label(i, 0);
                if (predicted_label == true_label)
                {
                    correct_predictions++;
                }
                num_samples++;
            }

            // Backward pass
            Eigen::Tensor<double, 4> d_output = lossFunction->derivative(predictions, batch_label);
            backward(d_output, learning_rate);

            // Print progress
            printProgress(epoch, epochs, batchCounter, totalBatches, start, batch_loss);
            batchCounter++;
        }

        double average_loss = total_loss / num_samples;
        double accuracy = static_cast<double>(correct_predictions) / num_samples;

        std::cout << std::endl
                  << "Epoch " << epoch + 1 << " complete." << std::endl;
        std::cout << "Accuracy: " << accuracy << std::endl;

        // Perform evaluation after each epoch
        evaluate(imageContainer);
    }
}

void NeuralNetwork::evaluate(const ImageContainer &imageContainer)
{
    if (!lossFunction)
    {
        throw std::runtime_error("Loss function must be set before evaluation.");
    }

    BatchManager batchManager(imageContainer, imageContainer.getTestImages().size(), BatchManager::BatchType::Testing);
    Eigen::Tensor<double, 4> batch_input;
    Eigen::Tensor<int, 2> batch_label;

    double total_loss = 0.0;
    int correct_predictions = 0;
    int num_samples = 0;

    while (batchManager.getNextBatch(batch_input, batch_label))
    {
        Eigen::Tensor<double, 4> predictions = forward(batch_input);
        total_loss += lossFunction->compute(predictions, batch_label);

        // Convert Eigen::Tensor to standard arrays for processing
        std::vector<int> pred_labels(predictions.dimension(0));
        std::vector<int> true_labels(predictions.dimension(0));

        for (int i = 0; i < predictions.dimension(0); ++i)
        {
            if (predictions.dimension(3) == 1) // Binary classification
            {
                pred_labels[i] = predictions(i, 0, 0, 0) >= 0.5 ? 1 : 0;
                true_labels[i] = batch_label(i, 0);
            }
            else // Multi-class classification
            {
                int maxIndex = 0;
                double maxValue = predictions(i, 0, 0, 0);
                for (int j = 1; j < predictions.dimension(3); ++j)
                {
                    if (predictions(i, 0, 0, j) > maxValue)
                    {
                        maxValue = predictions(i, 0, 0, j);
                        maxIndex = j;
                    }
                }
                pred_labels[i] = maxIndex;

                int trueMaxIndex = 0;
                int trueMaxValue = batch_label(i, 0);
                for (int j = 1; j < batch_label.dimension(1); ++j)
                {
                    if (batch_label(i, j) > trueMaxValue)
                    {
                        trueMaxValue = batch_label(i, j);
                        trueMaxIndex = j;
                    }
                }
                true_labels[i] = trueMaxIndex;
            }

            if (pred_labels[i] == true_labels[i])
            {
                correct_predictions++;
            }
            num_samples++;
        }
    }

    double average_loss = total_loss / num_samples;
    double accuracy = static_cast<double>(correct_predictions) / num_samples;

    std::cout << "Evaluation results - Loss: " << average_loss << ", Accuracy: " << accuracy << std::endl;
}
