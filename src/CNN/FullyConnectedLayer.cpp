#include "FullyConnectedLayer.hpp"
#include <stdexcept>
#include <random>

FullyConnectedLayer::FullyConnectedLayer(int input_size, int output_size, std::unique_ptr<Optimizer> optimizer, unsigned int seed)
    : input_size(input_size), output_size(output_size), optimizer(std::move(optimizer))
{
    if (input_size <= 0 || output_size <= 0)
    {
        throw std::invalid_argument("Input size and output size must be positive integers.");
    }

    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    weights = Eigen::MatrixXd(output_size, input_size);
    for (int i = 0; i < output_size; ++i)
    {
        for (int j = 0; j < input_size; ++j)
        {
            weights(i, j) = distribution(generator);
        }
    }

    biases = Eigen::VectorXd::Zero(output_size);
}

Eigen::MatrixXd FullyConnectedLayer::forward(const Eigen::MatrixXd &input_batch)
{
    if (input_batch.cols() != input_size)
    {
        throw std::invalid_argument("Input size does not match the expected size.");
    }

    return (input_batch * weights.transpose()).rowwise() + biases.transpose();
}

Eigen::MatrixXd FullyConnectedLayer::backward(const Eigen::MatrixXd &d_output_batch, const Eigen::MatrixXd &input_batch, double learning_rate)
{
    Eigen::MatrixXd d_weights = d_output_batch.transpose() * input_batch;
    Eigen::VectorXd d_biases = d_output_batch.colwise().sum();
    Eigen::MatrixXd d_input = d_output_batch * weights;

    optimizer->update(weights, biases, d_weights, d_biases, learning_rate);

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
