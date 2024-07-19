#include "FlattenLayer.hpp"
#include <iostream>
#include <stdexcept>

bool FlattenLayer::needsOptimizer() const
{
    return false;
}

void FlattenLayer::setOptimizer(std::shared_ptr<Optimizer> optimizer)
{
    return;
}

Eigen::Tensor<double, 4> FlattenLayer::forward(const Eigen::Tensor<double, 4> &input_batch)
{
    batch_size = static_cast<int>(input_batch.dimension(0));
    original_dimensions = {batch_size, static_cast<int>(input_batch.dimension(1)), static_cast<int>(input_batch.dimension(2)), static_cast<int>(input_batch.dimension(3))};

    int flattened_size = original_dimensions[1] * original_dimensions[2] * original_dimensions[3];
    Eigen::Tensor<double, 2> flattened_input = input_batch.reshape(Eigen::array<int, 2>{batch_size, flattened_size});

    // Reshape to 4D tensor
    return flattened_input.reshape(Eigen::array<int, 4>{batch_size, 1, 1, flattened_size});
}

Eigen::Tensor<double, 4> FlattenLayer::backward(const Eigen::Tensor<double, 4> &d_output_batch, const Eigen::Tensor<double, 4> &input_batch, double learning_rate)
{
    int flattened_size = original_dimensions[1] * original_dimensions[2] * original_dimensions[3];
    Eigen::Tensor<double, 2> d_output_2d = d_output_batch.reshape(Eigen::array<int, 2>{batch_size, flattened_size});

    // Reshape back to original dimensions
    return d_output_2d.reshape(Eigen::array<int, 4>{original_dimensions[0], original_dimensions[1], original_dimensions[2], original_dimensions[3]});
}
