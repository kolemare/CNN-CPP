#ifndef LAYER_HPP
#define LAYER_HPP

#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include "Common.hpp"
#include "Optimizer.hpp"

class Layer
{
public:
    virtual Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4> &input_batch) = 0;
    virtual Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4> &d_output_batch, const Eigen::Tensor<double, 4> &input_batch, double learning_rate) = 0;
    virtual bool needsOptimizer() const = 0;
    virtual void setOptimizer(std::shared_ptr<Optimizer> optimizer) = 0;
    virtual ~Layer() = default;
};

#endif // LAYER_HPP
