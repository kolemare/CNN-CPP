#ifndef FLATTENLAYER_HPP
#define FLATTENLAYER_HPP

#include "Layer.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

class FlattenLayer : public Layer
{
public:
    Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4> &input_batch) override;
    Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4> &d_output_batch, const Eigen::Tensor<double, 4> &input_batch, double learning_rate) override;
    bool needsOptimizer() const override;
    void setOptimizer(std::shared_ptr<Optimizer> optimizer) override;
    std::shared_ptr<Optimizer> getOptimizer() override;

private:
    int batch_size;
    std::vector<int> original_dimensions;
};

#endif // FLATTENLAYER_HPP
