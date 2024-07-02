#include "FullyConnectedLayer.hpp"
#include <stdexcept>
#include <iostream>

FullyConnectedLayer::FullyConnectedLayer(int output_size, DenseWeightInitialization weight_init, DenseBiasInitialization bias_init, unsigned int seed)
    : output_size(output_size)
{
    if (output_size <= 0)
    {
        throw std::invalid_argument("Output size must be a positive integer.");
    }

    initializeWeights(weight_init, seed);
    initializeBiases(bias_init);

    std::cout << "Initialized FullyConnectedLayer with Neurons: " << output_size << std::endl;
}

void FullyConnectedLayer::setInputSize(int input_size)
{
    if (input_size <= 0)
    {
        throw std::invalid_argument("Input size must be a positive integer.");
    }

    this->input_size = input_size;
    initializeWeights(DenseWeightInitialization::XAVIER, 42); // Default weight initialization
    initializeBiases(DenseBiasInitialization::ZERO);          // Default bias initialization
}

void FullyConnectedLayer::initializeWeights(DenseWeightInitialization weight_init, unsigned int seed)
{
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution;

    switch (weight_init)
    {
    case DenseWeightInitialization::XAVIER:
        distribution = std::normal_distribution<double>(0.0, std::sqrt(1.0 / input_size));
        break;
    case DenseWeightInitialization::HE:
        distribution = std::normal_distribution<double>(0.0, std::sqrt(2.0 / input_size));
        break;
    case DenseWeightInitialization::RANDOM_NORMAL:
        distribution = std::normal_distribution<double>(0.0, 1.0);
        break;
    }

    weights = Eigen::MatrixXd(output_size, input_size).unaryExpr([&](double)
                                                                 { return distribution(generator); });
}

void FullyConnectedLayer::initializeBiases(DenseBiasInitialization bias_init)
{
    if (bias_init == DenseBiasInitialization::NONE)
    {
        biases = Eigen::VectorXd::Zero(output_size);
        return;
    }

    biases.resize(output_size);

    std::default_random_engine generator(42); // Seed for bias initialization
    std::normal_distribution<> bias_dis(0, 1.0);

    switch (bias_init)
    {
    case DenseBiasInitialization::ZERO:
        biases = Eigen::VectorXd::Zero(output_size);
        break;
    case DenseBiasInitialization::RANDOM_NORMAL:
        for (int i = 0; i < output_size; ++i)
        {
            biases(i) = bias_dis(generator);
        }
        break;
    case DenseBiasInitialization::NONE:
        break;
    }
}

Eigen::MatrixXd FullyConnectedLayer::forward(const Eigen::MatrixXd &input_batch)
{
    if (input_batch.cols() != input_size)
    {
        throw std::invalid_argument("Input size does not match the expected size.");
    }

    Eigen::MatrixXd output = (input_batch * weights.transpose()).rowwise() + biases.transpose();
    return output;
}

Eigen::MatrixXd FullyConnectedLayer::backward(const Eigen::MatrixXd &d_output_batch, const Eigen::MatrixXd &input_batch, double learning_rate)
{
    if (d_output_batch.rows() != input_batch.rows())
    {
        throw std::invalid_argument("Output batch size does not match input batch size.");
    }

    Eigen::MatrixXd d_weights = d_output_batch.transpose() * input_batch;
    Eigen::VectorXd d_biases = d_output_batch.colwise().sum();
    Eigen::MatrixXd d_input = d_output_batch * weights;

    // Update weights and biases
    weights -= learning_rate * d_weights / input_batch.rows();
    biases -= learning_rate * d_biases / input_batch.rows();

    return d_input;
}

void FullyConnectedLayer::setWeights(const Eigen::MatrixXd &new_weights)
{
    if (new_weights.rows() != output_size || new_weights.cols() != input_size)
    {
        throw std::invalid_argument("The size of new_weights must match the layer dimensions.");
    }
    weights = new_weights;
}

Eigen::MatrixXd FullyConnectedLayer::getWeights() const
{
    return weights;
}

void FullyConnectedLayer::setBiases(const Eigen::VectorXd &new_biases)
{
    if (new_biases.size() != output_size)
    {
        throw std::invalid_argument("The size of new_biases must match the output size.");
    }
    biases = new_biases;
}

Eigen::VectorXd FullyConnectedLayer::getBiases() const
{
    return biases;
}

int FullyConnectedLayer::getOutputSize() const
{
    return output_size;
}
