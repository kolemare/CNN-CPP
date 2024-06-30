#include "NeuralNetwork.hpp"
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>

NeuralNetwork::NeuralNetwork() : flattenAdded(false) {}

void NeuralNetwork::addConvolutionLayer(int filters, int kernel_size, int input_depth, int stride, int padding, const Eigen::VectorXd &biases)
{
    layers.push_back(std::make_shared<ConvolutionLayer>(filters, kernel_size, input_depth, stride, padding, biases));
    std::cout << "Added Convolution Layer with " << filters << " filters, kernel size " << kernel_size << ", input depth " << input_depth << ", stride " << stride << ", padding " << padding << std::endl;
}

void NeuralNetwork::addMaxPoolingLayer(int pool_size, int stride)
{
    layers.push_back(std::make_shared<MaxPoolingLayer>(pool_size, stride));
    std::cout << "Added Max Pooling Layer with pool size " << pool_size << ", stride " << stride << std::endl;
}

void NeuralNetwork::addAveragePoolingLayer(int pool_size, int stride)
{
    layers.push_back(std::make_shared<AveragePoolingLayer>(pool_size, stride));
    std::cout << "Added Average Pooling Layer with pool size " << pool_size << ", stride " << stride << std::endl;
}

void NeuralNetwork::addFlattenLayer()
{
    if (!flattenAdded)
    {
        layers.push_back(std::make_shared<FlattenLayer>());
        flattenAdded = true;
        std::cout << "Added Flatten Layer" << std::endl;
    }
    else
    {
        std::cerr << "Flatten layer already added." << std::endl;
    }
}

void NeuralNetwork::addFullyConnectedLayer(int input_size, int output_size, std::unique_ptr<Optimizer> optimizer)
{
    layers.push_back(std::make_shared<FullyConnectedLayer>(input_size, output_size, std::move(optimizer)));
    std::cout << "Added Fully Connected Layer with input size " << input_size << ", output size " << output_size << std::endl;
}

void NeuralNetwork::addActivationLayer(ActivationType type)
{
    layers.push_back(std::make_shared<ActivationLayer>(type));
    std::cout << "Added Activation Layer of type " << type << std::endl;
}

void NeuralNetwork::setLossFunction(LossType type)
{
    lossFunction = LossFunction::create(type);
    std::cout << "Set Loss Function of type " << (int)type << std::endl;
}

Eigen::MatrixXd NeuralNetwork::forward(const Eigen::MatrixXd &input)
{
    std::cout << "-------------------------------------------------------------------" << std::endl;
    printMatrixSummary(input, "INPUT", PropagationType::FORWARD);
    std::cout << "-------------------------------------------------------------------" << std::endl;
    Eigen::MatrixXd output = input;
    layerInputs.clear();

    for (size_t i = 0; i < layers.size(); ++i)
    {
        layerInputs.push_back(output);
        output = layers[i]->forward(output);

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
        printMatrixSummary(output, layerType, PropagationType::FORWARD);
        std::cout << "-------------------------------------------------------------------" << std::endl;
    }

    return output;
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
    Eigen::MatrixXd d_input = d_output;

    for (int i = layers.size() - 1; i >= 0; --i)
    {
        d_input = layers[i]->backward(d_input, layerInputs[i], learning_rate);
    }
}

void NeuralNetwork::train(const ImageContainer &imageContainer, int epochs, double learning_rate, int batch_size, const std::vector<std::string> &categories)
{
    if (!lossFunction)
    {
        throw std::runtime_error("Loss function must be set before training.");
    }

    BatchManager batchManager(imageContainer, batch_size, categories);

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        Eigen::MatrixXd batch_input, batch_label;
        while (batchManager.getNextBatch(batch_input, batch_label))
        {
            // Forward pass
            Eigen::MatrixXd predictions = forward(batch_input);

            // Compute loss and backward pass
            Eigen::MatrixXd d_output = lossFunction->derivative(predictions, batch_label);
            backward(d_output, learning_rate);
        }

        std::cout << "Epoch " << epoch + 1 << " complete." << std::endl;
    }
}

void NeuralNetwork::evaluate(const ImageContainer &imageContainer, const std::vector<std::string> &categories)
{
    if (!lossFunction)
    {
        throw std::runtime_error("Loss function must be set before evaluation.");
    }

    BatchManager batchManager(imageContainer, imageContainer.getTestImages().size(), categories);
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
