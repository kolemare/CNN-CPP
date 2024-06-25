#include "FlattenLayer.hpp"
#include <iostream>
#include <stdexcept>

Eigen::MatrixXd FlattenLayer::forward(const Eigen::MatrixXd &input)
{
    batch_size = input.rows();
    original_size = input.cols();

    return Eigen::Map<const Eigen::MatrixXd>(input.data(), batch_size, original_size);
}

Eigen::MatrixXd FlattenLayer::backward(const Eigen::MatrixXd &d_output, const Eigen::MatrixXd &input, double learning_rate)
{
    return Eigen::Map<const Eigen::MatrixXd>(d_output.data(), batch_size, original_size);
}
