#include "TensorOperations.hpp"

void TensorOperations::applyUpdates(Eigen::Tensor<double, 2> &weights, const Eigen::Tensor<double, 2> &updates, double scale)
{
    weights -= scale * updates;
}

void TensorOperations::applyUpdates(Eigen::Tensor<double, 4> &weights, const Eigen::Tensor<double, 4> &updates, double scale)
{
    weights -= scale * updates;
}

void TensorOperations::applyUpdates(Eigen::Tensor<double, 1> &biases, const Eigen::Tensor<double, 1> &updates, double scale)
{
    biases -= scale * updates;
}
