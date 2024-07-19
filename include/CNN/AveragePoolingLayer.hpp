#ifndef AVERAGE_POOLING_LAYER_HPP
#define AVERAGE_POOLING_LAYER_HPP

#include "Layer.hpp"
#include <unsupported/Eigen/CXX11/Tensor>

class AveragePoolingLayer : public Layer
{
public:
    AveragePoolingLayer(int pool_size, int stride);

    Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4> &input_batch) override;
    Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4> &d_output_batch, const Eigen::Tensor<double, 4> &input_batch, double learning_rate) override;
    bool needsOptimizer() const override;
    void setOptimizer(std::shared_ptr<Optimizer> optimizer) override;

    int getPoolSize();
    int getStride();

private:
    int pool_size;
    int stride;
};

#endif // AVERAGEPOOLINGLAYER_HPP
