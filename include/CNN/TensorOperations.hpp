#ifndef TENSOR_OPERATIONS_HPP
#define TENSOR_OPERATIONS_HPP

#include <unsupported/Eigen/CXX11/Tensor>

class TensorOperations
{
public:
    static void applyUpdates(Eigen::Tensor<double, 2> &weights, const Eigen::Tensor<double, 2> &updates, double scale);
    static void applyUpdates(Eigen::Tensor<double, 4> &weights, const Eigen::Tensor<double, 4> &updates, double scale);
    static void applyUpdates(Eigen::Tensor<double, 1> &biases, const Eigen::Tensor<double, 1> &updates, double scale);
};

#endif // TENSOR_OPERATIONS_HPP
