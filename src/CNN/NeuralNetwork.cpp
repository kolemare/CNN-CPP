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

void NeuralNetwork::compile(std::unique_ptr<Optimizer> optimizer)
{
    this->optimizer = std::move(optimizer);

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
        }
        else if (dynamic_cast<FlattenLayer *>(layers[i].get()))
        {
            input_size = height * width * currentDepth;
        }
    }
}

Eigen::MatrixXd NeuralNetwork::forward(const Eigen::MatrixXd &input)
{
    if (logLevel == LogLevel::All || logLevel == LogLevel::LayerOutputs)
    {
        std::cout << "-------------------------------------------------------------------" << std::endl;
        printMatrixSummary(input, "INPUT", PropagationType::FORWARD);
        std::cout << "-------------------------------------------------------------------" << std::endl;
    }

    Eigen::MatrixXd output = input;
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
                printFullMatrix(output, layerType, PropagationType::FORWARD);
            }
            else
            {
                printMatrixSummary(output, layerType, PropagationType::FORWARD);
            }
            std::cout << "-------------------------------------------------------------------" << std::endl;
        }
    }

    return output;
}

void NeuralNetwork::printFullMatrix(const Eigen::MatrixXd &matrix, const std::string &layerType, PropagationType propagationType)
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
    std::cout << matrix << std::endl;
}

void NeuralNetwork::printMatrixSummary(const Eigen::MatrixXd &matrix, const std::string &layerType, PropagationType propagationType)
{
    double mean = matrix.mean();
    double stddev = std::sqrt((matrix.array() - mean).square().sum() / (matrix.size() - 1));
    double minCoeff = matrix.minCoeff();
    double maxCoeff = matrix.maxCoeff();
    double zeroPercentage = (matrix.array() == 0).count() / static_cast<double>(matrix.size()) * 100.0;
    double negativeCount = (matrix.array() < 0).count();
    double positiveCount = (matrix.array() > 0).count();

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
    std::cout << "Dimensions: " << matrix.rows() << "x" << matrix.cols() << "\n";
    std::cout << "Mean: " << mean << "\n";
    std::cout << "Standard Deviation: " << stddev << "\n";
    std::cout << "Min: " << minCoeff << "\n";
    std::cout << "Max: " << maxCoeff << "\n";
    std::cout << "Percentage of Zeros: " << zeroPercentage << "%\n";
    std::cout << "Number of Negative Values: " << negativeCount << "\n";
    std::cout << "Number of Positive Values: " << positiveCount << "\n\n";
}

void NeuralNetwork::backward(const Eigen::MatrixXd &d_output, double learning_rate)
{
    if (logLevel == LogLevel::All || logLevel == LogLevel::LayerOutputs)
    {
        std::cout << "-------------------------------------------------------------------" << std::endl;
        printMatrixSummary(d_output, "OUTPUT", PropagationType::BACK);
        std::cout << "-------------------------------------------------------------------" << std::endl;
    }

    Eigen::MatrixXd d_input = d_output;

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
                printFullMatrix(d_input, layerType, PropagationType::BACK);
            }
            else
            {
                printMatrixSummary(d_input, layerType, PropagationType::BACK);
            }
            std::cout << "-------------------------------------------------------------------" << std::endl;
        }
    }
}

void NeuralNetwork::printProgress(int epoch, int epochs, int batch, int totalBatches, std::chrono::steady_clock::time_point start)
{
    if (progressLevel == ProgressLevel::None)
    {
        return;
    }

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
    double progress = static_cast<double>(batch + 1) / totalBatches;
    int barWidth = 50;
    int pos = static_cast<int>(barWidth * progress);

    double timePerBatch = elapsed / static_cast<double>(batch + 1);
    double remainingTime = (totalBatches - batch - 1) * timePerBatch;
    double totalTime = elapsed + remainingTime;

    double overallProgress = static_cast<double>(epoch * totalBatches + batch + 1) / (totalBatches * epochs);
    int overallPos = static_cast<int>(barWidth * overallProgress);
    double overallRemainingTime = (totalBatches * epochs - (epoch * totalBatches + batch + 1)) * timePerBatch;

    std::ostringstream oss;
    oss << "-------------------------------------------------------------------" << std::endl;

    std::ostringstream epochProgress;
    epochProgress << "Epoch " << epoch + 1 << " | Batch " << batch + 1 << "/" << totalBatches << " [";
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
    overallProgressLabel << "Overall Progress";
    int labelLength = overallProgressLabel.str().length();

    // Dynamically adjust the overall progress label length to match epoch progress length
    std::string epochPrefix = "Epoch " + std::to_string(epoch + 1) + " | Batch " + std::to_string(batch + 1) + "/" + std::to_string(totalBatches);
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

        double batchDuration = elapsed - (batch * timePerBatch);
        oss << "Elapsed: " << formatTime(elapsed) << "\n";
        oss << "ETA for current epoch: " << formatTime(remainingTime) << "\n";
        oss << "Duration of current batch: " << formatTime(batchDuration) << "\n";
        oss << "ETA for overall progress: " << formatTime(overallRemainingTime) << "\n";

        if (batch == 0) // Print only at the beginning of the first epoch
        {
            oss << "Total time for current epoch: " << formatTime(totalTime) << "\n";
            if (epoch == 0)
            {
                double overallTotalTime = elapsed + overallRemainingTime;
                oss << "Overall estimated total time: " << formatTime(overallTotalTime) << "\n";
            }
        }
    }

    oss << "-------------------------------------------------------------------" << std::endl;
    std::cout << oss.str() << std::flush;
}

void saveBatchImages(const Eigen::MatrixXd &batch_input, const Eigen::MatrixXd &batch_label, const std::string &folderPath)
{
    std::filesystem::create_directory(folderPath);

    for (int i = 0; i < batch_input.rows(); ++i)
    {
        // Convert Eigen::MatrixXd row to cv::Mat
        Eigen::VectorXd eigen_image = batch_input.row(i);
        Eigen::MatrixXf float_image = eigen_image.cast<float>(); // Convert to float

        // Assuming the images are 32x32 with 3 channels (adjust as needed)
        cv::Mat image(32, 32, CV_32FC3, float_image.data());

        // Convert the image to 8-bit for saving without normalization
        image.convertTo(image, CV_8UC3, 255.0);

        // Construct the filename
        std::string filename = folderPath + "/image_" + std::to_string(i) + "_label_" + std::to_string(static_cast<int>(batch_label(i, 0))) + ".png";

        // Save the image
        cv::imwrite(filename, image);
    }
}

void NeuralNetwork::train(const ImageContainer &imageContainer, int epochs, double learning_rate, int batch_size, const std::vector<std::string> &categories)
{
    if (!lossFunction)
    {
        throw std::runtime_error("Loss function must be set before training.");
    }

    BatchManager batchManager(imageContainer, batch_size, categories, BatchManager::BatchType::Training);
    std::cout << "Training started..." << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        Eigen::MatrixXd batch_input, batch_label;
        int totalBatches = batchManager.getTotalBatches();
        auto start = std::chrono::steady_clock::now();
        int batchCounter = 0;
        double total_loss = 0.0;
        int correct_predictions = 0;
        int num_samples = 0;

        while (batchManager.getNextBatch(batch_input, batch_label))
        {
            // Forward pass
            Eigen::MatrixXd predictions = forward(batch_input);

            // Compute loss
            double batch_loss = lossFunction->compute(predictions, batch_label);
            total_loss += batch_loss;

            // Count correct predictions
            for (int i = 0; i < predictions.rows(); ++i)
            {
                int predicted_label = (predictions(i, 0) >= 0.5) ? 1 : 0;
                int true_label = batch_label(i, 0);
                if (predicted_label == true_label)
                {
                    correct_predictions++;
                }
                num_samples++;
            }

            std::cout << "Batch " << batchCounter + 1 << "/" << totalBatches << " - Loss: " << batch_loss << std::endl;

            // Save one batch of images and labels
            if (epoch == 0 && batchCounter == 0)
            {
                saveBatchImages(batch_input, batch_label, "saved_batch");
                throw std::runtime_error("Batch saved for inspection.");
            }

            // Backward pass
            Eigen::MatrixXd d_output = lossFunction->derivative(predictions, batch_label);
            backward(d_output, learning_rate);

            // Print progress
            // printProgress(epoch, epochs, batchCounter, totalBatches, start);
            batchCounter++;
        }

        double average_loss = total_loss / num_samples;
        double accuracy = static_cast<double>(correct_predictions) / num_samples;

        std::cout << std::endl
                  << "Epoch " << epoch + 1 << " complete." << std::endl;
        std::cout << "Training Loss: " << average_loss << ", Accuracy: " << accuracy << std::endl;

        // Perform evaluation after each epoch
        evaluate(imageContainer, categories);
    }
}

void NeuralNetwork::evaluate(const ImageContainer &imageContainer, const std::vector<std::string> &categories)
{
    if (!lossFunction)
    {
        throw std::runtime_error("Loss function must be set before evaluation.");
    }

    BatchManager batchManager(imageContainer, imageContainer.getTestImages().size(), categories, BatchManager::BatchType::Testing);
    Eigen::MatrixXd batch_input, batch_label;

    double total_loss = 0.0;
    int correct_predictions = 0;
    int num_samples = 0;

    while (batchManager.getNextBatch(batch_input, batch_label))
    {
        Eigen::MatrixXd predictions = forward(batch_input);
        total_loss += lossFunction->compute(predictions, batch_label);

        // Assuming classification (binary or multi-class)
        for (int i = 0; i < predictions.rows(); ++i)
        {
            int predicted_label = -1;
            int true_label = -1;

            if (predictions.cols() == 1) // Binary classification
            {
                predicted_label = predictions(i, 0) >= 0.5 ? 1 : 0;
                true_label = batch_label(i, 0);
            }
            else // Multi-class classification
            {
                predicted_label = std::distance(predictions.row(i).data(),
                                                std::max_element(predictions.row(i).data(), predictions.row(i).data() + predictions.cols()));
                true_label = std::distance(batch_label.row(i).data(),
                                           std::max_element(batch_label.row(i).data(), batch_label.row(i).data() + batch_label.cols()));
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

    std::cout << "Evaluation results - Loss: " << average_loss << ", Accuracy: " << accuracy << std::endl;
}
