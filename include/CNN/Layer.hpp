#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Dense>
#include <memory>
#include "Optimizer.hpp"

class Layer
{
public:
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd &input_batch) = 0;
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd &d_output_batch, const Eigen::MatrixXd &input_batch, double learning_rate) = 0;
    virtual bool needsOptimizer() const = 0;
    virtual void setOptimizer(std::unique_ptr<Optimizer> optimizer) = 0;
    virtual ~Layer() = default;
};

#endif // LAYER_HPP
