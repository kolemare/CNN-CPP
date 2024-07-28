#ifndef MAX_POOLING_LAYER_HPP
#define MAX_POOLING_LAYER_HPP

#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include "Layer.hpp"

class MaxPoolingLayer : public Layer
{
public:
    MaxPoolingLayer(int pool_size, int stride);

    Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4> &input_batch) override;
    Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4> &d_output_batch, const Eigen::Tensor<double, 4> &input_batch, double learning_rate) override;
    bool needsOptimizer() const override;
    void setOptimizer(std::shared_ptr<Optimizer> optimizer) override;
    std::shared_ptr<Optimizer> getOptimizer() override;

    int getPoolSize() const;
    int getStride() const;

private:
    int pool_size;
    int stride;

    std::vector<Eigen::Tensor<int, 4>> max_indices;

    Eigen::Tensor<double, 3> maxPool(const Eigen::Tensor<double, 3> &input, Eigen::Tensor<int, 3> &indices);
    Eigen::Tensor<double, 3> maxPoolBackward(const Eigen::Tensor<double, 3> &d_output, const Eigen::Tensor<int, 3> &indices);
};

#endif // MAX_POOLING_LAYER_HPP
