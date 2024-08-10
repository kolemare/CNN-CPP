#include "FlattenLayer.hpp"
#include <iostream>
#include <stdexcept>

bool FlattenLayer::needsOptimizer() const
{
    return false; // Flatten layer does not need an optimizer
}

void FlattenLayer::setOptimizer(std::shared_ptr<Optimizer> optimizer)
{
    return; // No operation for flatten layer as it does not use an optimizer
}

std::shared_ptr<Optimizer> FlattenLayer::getOptimizer()
{
    return nullptr; // Return nullptr since no optimizer is associated with the flatten layer
}

Eigen::Tensor<double, 4> FlattenLayer::forward(const Eigen::Tensor<double, 4> &input_batch)
{
    // Store the batch size and original dimensions
    batch_size = static_cast<int>(input_batch.dimension(0));
    original_dimensions = {batch_size, static_cast<int>(input_batch.dimension(1)), static_cast<int>(input_batch.dimension(2)), static_cast<int>(input_batch.dimension(3))};

    // Calculate the flattened size by multiplying the dimensions (depth * height * width)
    int flattened_size = original_dimensions[1] * original_dimensions[2] * original_dimensions[3];

    // Reshape the input to a 2D tensor of shape (batch_size, flattened_size)
    Eigen::Tensor<double, 2> flattened_input = input_batch.reshape(Eigen::array<int, 2>{batch_size, flattened_size});

    // Reshape the 2D tensor back to a 4D tensor of shape (batch_size, 1, 1, flattened_size)
    return flattened_input.reshape(Eigen::array<int, 4>{batch_size, 1, 1, flattened_size});
}

Eigen::Tensor<double, 4> FlattenLayer::backward(const Eigen::Tensor<double, 4> &d_output_batch,
                                                const Eigen::Tensor<double, 4> &input_batch,
                                                double learning_rate)
{
    // Calculate the flattened size by multiplying the dimensions (depth * height * width)
    int flattened_size = original_dimensions[1] * original_dimensions[2] * original_dimensions[3];

    // Reshape the gradient output to a 2D tensor of shape (batch_size, flattened_size)
    Eigen::Tensor<double, 2> d_output_2d = d_output_batch.reshape(Eigen::array<int, 2>{batch_size, flattened_size});

    // Reshape the 2D tensor back to the original 4D tensor shape
    return d_output_2d.reshape(Eigen::array<int, 4>{original_dimensions[0], original_dimensions[1], original_dimensions[2], original_dimensions[3]});
}
