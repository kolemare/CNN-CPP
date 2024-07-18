#include "FlattenLayer.hpp"
#include <iostream>
#include <stdexcept>

bool FlattenLayer::needsOptimizer() const
{
    return false;
}

void FlattenLayer::setOptimizer(std::unique_ptr<Optimizer> optimizer)
{
    return;
}

Eigen::MatrixXd FlattenLayer::forward(const Eigen::MatrixXd &input)
{
    batch_size = input.rows();
    original_size = input.cols();

    Eigen::MatrixXd output = Eigen::Map<const Eigen::MatrixXd>(input.data(), batch_size, original_size);

    return output;
}

Eigen::MatrixXd FlattenLayer::backward(const Eigen::MatrixXd &d_output, const Eigen::MatrixXd &input, double learning_rate)
{
    return Eigen::Map<const Eigen::MatrixXd>(d_output.data(), batch_size, original_size);
}
