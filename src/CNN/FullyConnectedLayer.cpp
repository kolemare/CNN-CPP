#include "FullyConnectedLayer.hpp"
#include <stdexcept>
#include <iostream>

FullyConnectedLayer::FullyConnectedLayer(int output_size, DenseWeightInitialization weight_init, DenseBiasInitialization bias_init, unsigned int seed)
    : output_size(output_size), weight_init(weight_init), bias_init(bias_init), seed(seed)
{
    if (output_size <= 0)
    {
        throw std::invalid_argument("Output size must be a positive integer.");
    }
}

bool FullyConnectedLayer::needsOptimizer() const
{
    return true;
}

void FullyConnectedLayer::setOptimizer(std::shared_ptr<Optimizer> optimizer)
{
    this->optimizer = optimizer;
}

void FullyConnectedLayer::setInputSize(int input_size)
{
    if (input_size <= 0)
    {
        throw std::invalid_argument("Input size must be a positive integer.");
    }

    this->input_size = input_size;
    initializeWeights();
    initializeBiases();
}

void FullyConnectedLayer::initializeWeights()
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

    weights = Eigen::Tensor<double, 2>(output_size, input_size);
    for (int i = 0; i < output_size; ++i)
    {
        for (int j = 0; j < input_size; ++j)
        {
            weights(i, j) = distribution(generator);
        }
    }
}

void FullyConnectedLayer::initializeBiases()
{
    if (bias_init == DenseBiasInitialization::NONE)
    {
        biases = Eigen::Tensor<double, 1>(output_size);
        biases.setZero();
        return;
    }

    biases = Eigen::Tensor<double, 1>(output_size);

    std::default_random_engine generator(seed); // Seed for bias initialization
    std::normal_distribution<> bias_dis(0, 1.0);

    switch (bias_init)
    {
    case DenseBiasInitialization::ZERO:
        biases.setZero();
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

Eigen::Tensor<double, 4> FullyConnectedLayer::forward(const Eigen::Tensor<double, 4> &input_batch)
{
    int batch_size = input_batch.dimension(0);
    int input_flattened_size = input_batch.dimension(1) * input_batch.dimension(2) * input_batch.dimension(3);

    if (input_flattened_size != input_size)
    {
        throw std::invalid_argument("Input size does not match the expected size.");
    }

    Eigen::Tensor<double, 2> input_2d = input_batch.reshape(Eigen::array<int, 2>{batch_size, input_flattened_size});
    Eigen::Tensor<double, 2> output_2d = (input_2d.contract(weights, Eigen::array<Eigen::IndexPair<int>, 1>{{Eigen::IndexPair<int>(1, 1)}})).reshape(Eigen::array<int, 2>{batch_size, output_size});
    output_2d = output_2d + biases.broadcast(Eigen::array<int, 2>{batch_size, 1});

    return output_2d.reshape(Eigen::array<int, 4>{batch_size, 1, 1, output_size});
}

Eigen::Tensor<double, 4> FullyConnectedLayer::backward(const Eigen::Tensor<double, 4> &d_output_batch, const Eigen::Tensor<double, 4> &input_batch, double learning_rate)
{
    int batch_size = input_batch.dimension(0);
    int input_flattened_size = input_batch.dimension(1) * input_batch.dimension(2) * input_batch.dimension(3);

    if (d_output_batch.dimension(0) != batch_size || d_output_batch.dimension(3) != output_size)
    {
        throw std::invalid_argument("Output batch size does not match input batch size.");
    }

    Eigen::Tensor<double, 2> input_2d = input_batch.reshape(Eigen::array<int, 2>{batch_size, input_flattened_size});
    Eigen::Tensor<double, 2> d_output_2d = d_output_batch.reshape(Eigen::array<int, 2>{batch_size, output_size});

    Eigen::Tensor<double, 2> d_weights = d_output_2d.contract(input_2d, Eigen::array<Eigen::IndexPair<int>, 1>{{Eigen::IndexPair<int>(0, 0)}});
    Eigen::Tensor<double, 1> d_biases = d_output_2d.sum(Eigen::array<int, 1>{0});
    Eigen::Tensor<double, 2> d_input_2d = d_output_2d.contract(weights, Eigen::array<Eigen::IndexPair<int>, 1>{{Eigen::IndexPair<int>(1, 0)}});

    optimizer->update(weights, biases, d_weights, d_biases, learning_rate);

    return d_input_2d.reshape(Eigen::array<int, 4>{batch_size, static_cast<int>(input_batch.dimension(1)), static_cast<int>(input_batch.dimension(2)), static_cast<int>(input_batch.dimension(3))});
}

void FullyConnectedLayer::setWeights(const Eigen::Tensor<double, 2> &new_weights)
{
    if (new_weights.dimension(0) != output_size || new_weights.dimension(1) != input_size)
    {
        throw std::invalid_argument("The size of new_weights must match the layer dimensions.");
    }
    weights = new_weights;
}

Eigen::Tensor<double, 2> FullyConnectedLayer::getWeights() const
{
    return weights;
}

void FullyConnectedLayer::setBiases(const Eigen::Tensor<double, 1> &new_biases)
{
    if (new_biases.dimension(0) != output_size)
    {
        throw std::invalid_argument("The size of new_biases must match the output size.");
    }
    biases = new_biases;
}

Eigen::Tensor<double, 1> FullyConnectedLayer::getBiases() const
{
    return biases;
}

int FullyConnectedLayer::getOutputSize() const
{
    return output_size;
}
