#include "FullyConnectedLayer.hpp"
#include <stdexcept>
#include <random>
#include <iostream>

FullyConnectedLayer::FullyConnectedLayer(int input_size, int output_size, std::unique_ptr<Optimizer> optimizer, unsigned int seed)
    : input_size(input_size), output_size(output_size), optimizer(std::move(optimizer))
{
    if (input_size <= 0 || output_size <= 0)
    {
        throw std::invalid_argument("Input size and output size must be positive integers.");
    }

    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0.0, std::sqrt(2.0 / input_size)); // Xavier initialization for weights

    weights = Eigen::MatrixXd(output_size, input_size);
    for (int i = 0; i < output_size; ++i)
    {
        for (int j = 0; j < input_size; ++j)
        {
            weights(i, j) = distribution(generator);
        }
    }

    // Initialize biases to zero
    biases = Eigen::VectorXd::Zero(output_size);

    std::cout << "Initialized FullyConnectedLayer with random weights and zero biases." << std::endl;
}

Eigen::MatrixXd FullyConnectedLayer::forward(const Eigen::MatrixXd &input_batch)
{
    if (input_batch.cols() != input_size)
    {
        throw std::invalid_argument("Input size does not match the expected size.");
    }

    // Debugging output for the input
    std::cout << "Forward pass input: \n"
              << input_batch << std::endl;

    Eigen::MatrixXd output = (input_batch * weights.transpose()).rowwise() + biases.transpose();
    std::cout << "Fully Connected Layer forward pass output dimensions: " << output.rows() << "x" << output.cols() << std::endl;

    // Add debugging output
    std::cout << "Forward pass weights: \n"
              << weights << std::endl;
    std::cout << "Forward pass biases: \n"
              << biases << std::endl;
    std::cout << "Forward pass output: \n"
              << output << std::endl;

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

    optimizer->update(weights, biases, d_weights, d_biases, learning_rate);

    // Add debugging output
    std::cout << "Backward pass d_weights: \n"
              << d_weights << std::endl;
    std::cout << "Backward pass d_biases: \n"
              << d_biases << std::endl;
    std::cout << "Backward pass d_input: \n"
              << d_input << std::endl;

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
