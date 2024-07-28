#include "FullyConnectedLayer.hpp"
#include <stdexcept>
#include <iostream>

// Constructor to initialize the fully connected layer
FullyConnectedLayer::FullyConnectedLayer(int output_size, DenseWeightInitialization weight_init, DenseBiasInitialization bias_init, unsigned int seed)
    : output_size(output_size), weight_init(weight_init), bias_init(bias_init), seed(seed)
{
    if (output_size <= 0)
    {
        throw std::invalid_argument("Output size must be a positive integer.");
    }
}

// Function to check if an optimizer is needed
bool FullyConnectedLayer::needsOptimizer() const
{
    return true;
}

// Function to set the optimizer
void FullyConnectedLayer::setOptimizer(std::shared_ptr<Optimizer> optimizer)
{
    this->optimizer = optimizer;
}

// Function to set the input size and initialize weights and biases
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

// Function to initialize the weights of the layer
void FullyConnectedLayer::initializeWeights()
{
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution;

    // Select the distribution based on the weight initialization method
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

// Function to initialize the biases of the layer
void FullyConnectedLayer::initializeBiases()
{
    biases = Eigen::Tensor<double, 1>(output_size);

    // If no bias is required, set biases to zero
    if (bias_init == DenseBiasInitialization::NONE)
    {
        biases.setZero();
        return;
    }

    std::default_random_engine generator(seed);
    std::normal_distribution<> bias_dis(0, 1.0);

    // Select the bias initialization method
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

// Forward pass for the fully connected layer
Eigen::Tensor<double, 4> FullyConnectedLayer::forward(const Eigen::Tensor<double, 4> &input_batch)
{
    int batch_size = input_batch.dimension(0);                                                                 // Number of images in the batch
    int input_flattened_size = input_batch.dimension(1) * input_batch.dimension(2) * input_batch.dimension(3); // Flattened input size

    if (input_flattened_size != input_size)
    {
        throw std::invalid_argument("Input size does not match the expected size.");
    }

    // Reshape the input to 2D tensor for matrix multiplication
    Eigen::Tensor<double, 2> input_2d(batch_size, input_flattened_size);
    for (int b = 0; b < batch_size; ++b)
    {
        for (int i = 0; i < input_batch.dimension(1); ++i)
        {
            for (int j = 0; j < input_batch.dimension(2); ++j)
            {
                for (int k = 0; k < input_batch.dimension(3); ++k)
                {
                    input_2d(b, i * input_batch.dimension(2) * input_batch.dimension(3) + j * input_batch.dimension(3) + k) = input_batch(b, i, j, k);
                }
            }
        }
    }

    // Compute the output by matrix multiplication and adding biases
    Eigen::Tensor<double, 2> output_2d(batch_size, output_size);
    output_2d.setZero();
    for (int b = 0; b < batch_size; ++b)
    {
        for (int o = 0; o < output_size; ++o)
        {
            for (int i = 0; i < input_size; ++i)
            {
                output_2d(b, o) += input_2d(b, i) * weights(o, i);
            }
            output_2d(b, o) += biases(o);
        }
    }

    // Reshape the output back to 4D tensor
    return output_2d.reshape(Eigen::array<int, 4>{batch_size, 1, 1, output_size});
}

// Backward pass for the fully connected layer
Eigen::Tensor<double, 4> FullyConnectedLayer::backward(const Eigen::Tensor<double, 4> &d_output_batch, const Eigen::Tensor<double, 4> &input_batch, double learning_rate)
{
    int batch_size = input_batch.dimension(0);                                                                 // Number of images in the batch
    int input_flattened_size = input_batch.dimension(1) * input_batch.dimension(2) * input_batch.dimension(3); // Flattened input size

    if (d_output_batch.dimension(0) != batch_size || d_output_batch.dimension(3) != output_size)
    {
        throw std::invalid_argument("Output batch size does not match input batch size.");
    }

    // Reshape the input and output gradients to 2D tensors
    Eigen::Tensor<double, 2> input_2d(batch_size, input_flattened_size);
    for (int b = 0; b < batch_size; ++b)
    {
        for (int i = 0; i < input_batch.dimension(1); ++i)
        {
            for (int j = 0; j < input_batch.dimension(2); ++j)
            {
                for (int k = 0; k < input_batch.dimension(3); ++k)
                {
                    input_2d(b, i * input_batch.dimension(2) * input_batch.dimension(3) + j * input_batch.dimension(3) + k) = input_batch(b, i, j, k);
                }
            }
        }
    }

    Eigen::Tensor<double, 2> d_output_2d(batch_size, output_size);
    for (int b = 0; b < batch_size; ++b)
    {
        for (int o = 0; o < output_size; ++o)
        {
            d_output_2d(b, o) = d_output_batch(b, 0, 0, o);
        }
    }

    // Calculate the gradients with respect to weights and biases
    Eigen::Tensor<double, 2> d_weights(output_size, input_size);
    d_weights.setZero();
    for (int o = 0; o < output_size; ++o)
    {
        for (int i = 0; i < input_size; ++i)
        {
            for (int b = 0; b < batch_size; ++b)
            {
                d_weights(o, i) += d_output_2d(b, o) * input_2d(b, i);
            }
        }
    }

    Eigen::Tensor<double, 1> d_biases(output_size);
    d_biases.setZero();
    for (int o = 0; o < output_size; ++o)
    {
        for (int b = 0; b < batch_size; ++b)
        {
            d_biases(o) += d_output_2d(b, o);
        }
    }

    // Calculate the gradients with respect to the input
    Eigen::Tensor<double, 2> d_input_2d(batch_size, input_size);
    d_input_2d.setZero();
    for (int b = 0; b < batch_size; ++b)
    {
        for (int i = 0; i < input_size; ++i)
        {
            for (int o = 0; o < output_size; ++o)
            {
                d_input_2d(b, i) += d_output_2d(b, o) * weights(o, i);
            }
        }
    }

    // Update weights and biases using the optimizer
    optimizer->update(weights, biases, d_weights, d_biases, learning_rate);

    // Reshape the input gradient back to 4D tensor
    return d_input_2d.reshape(Eigen::array<int, 4>{batch_size, static_cast<int>(input_batch.dimension(1)), static_cast<int>(input_batch.dimension(2)), static_cast<int>(input_batch.dimension(3))});
}

// Function to set the weights of the layer
void FullyConnectedLayer::setWeights(const Eigen::Tensor<double, 2> &new_weights)
{
    if (new_weights.dimension(0) != output_size || new_weights.dimension(1) != input_size)
    {
        throw std::invalid_argument("The size of new_weights must match the layer dimensions.");
    }
    weights = new_weights;
}

// Function to get the weights of the layer
Eigen::Tensor<double, 2> FullyConnectedLayer::getWeights() const
{
    return Eigen::Tensor<double, 2>(weights); // Return a copy
}

// Function to set the biases of the layer
void FullyConnectedLayer::setBiases(const Eigen::Tensor<double, 1> &new_biases)
{
    if (new_biases.dimension(0) != output_size)
    {
        throw std::invalid_argument("The size of new_biases must match the output size.");
    }
    biases = new_biases;
}

// Function to get the biases of the layer
Eigen::Tensor<double, 1> FullyConnectedLayer::getBiases() const
{
    return Eigen::Tensor<double, 1>(biases); // Return a copy
}

// Function to get the output size of the layer
int FullyConnectedLayer::getOutputSize() const
{
    return output_size;
}
