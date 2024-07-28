#include "AveragePoolingLayer.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

AveragePoolingLayer::AveragePoolingLayer(int pool_size, int stride)
    : pool_size(pool_size), stride(stride) {}

bool AveragePoolingLayer::needsOptimizer() const
{
    return false;
}

void AveragePoolingLayer::setOptimizer(std::shared_ptr<Optimizer> optimizer)
{
    return;
}

std::shared_ptr<Optimizer> AveragePoolingLayer::getOptimizer()
{
    return nullptr;
}

int AveragePoolingLayer::getPoolSize()
{
    return pool_size;
}

int AveragePoolingLayer::getStride()
{
    return stride;
}

Eigen::Tensor<double, 4> AveragePoolingLayer::forward(const Eigen::Tensor<double, 4> &input_batch)
{
    int batch_size = input_batch.dimension(0);
    int depth = input_batch.dimension(1);
    int input_height = input_batch.dimension(2);
    int input_width = input_batch.dimension(3);
    int output_height = (input_height - pool_size) / stride + 1;
    int output_width = (input_width - pool_size) / stride + 1;

    Eigen::Tensor<double, 4> output_batch(batch_size, depth, output_height, output_width);

    for (int b = 0; b < batch_size; ++b)
    {
        for (int d = 0; d < depth; ++d)
        {
            for (int i = 0; i < output_height; ++i)
            {
                for (int j = 0; j < output_width; ++j)
                {
                    int row_start = i * stride;
                    int col_start = j * stride;
                    double sum = 0.0;

                    for (int m = 0; m < pool_size; ++m)
                    {
                        for (int n = 0; n < pool_size; ++n)
                        {
                            sum += input_batch(b, d, row_start + m, col_start + n);
                        }
                    }

                    output_batch(b, d, i, j) = sum / (pool_size * pool_size);
                }
            }
        }
    }

    return output_batch;
}

Eigen::Tensor<double, 4> AveragePoolingLayer::backward(const Eigen::Tensor<double, 4> &d_output_batch, const Eigen::Tensor<double, 4> &input_batch, double learning_rate)
{
    int batch_size = input_batch.dimension(0);
    int depth = input_batch.dimension(1);
    int input_height = input_batch.dimension(2);
    int input_width = input_batch.dimension(3);
    int output_height = d_output_batch.dimension(2);
    int output_width = d_output_batch.dimension(3);

    Eigen::Tensor<double, 4> d_input_batch(batch_size, depth, input_height, input_width);
    d_input_batch.setZero();

    for (int b = 0; b < batch_size; ++b)
    {
        for (int d = 0; d < depth; ++d)
        {
            for (int i = 0; i < output_height; ++i)
            {
                for (int j = 0; j < output_width; ++j)
                {
                    int row_start = i * stride;
                    int col_start = j * stride;
                    double gradient = d_output_batch(b, d, i, j) / (pool_size * pool_size);

                    for (int m = 0; m < pool_size; ++m)
                    {
                        for (int n = 0; n < pool_size; ++n)
                        {
                            d_input_batch(b, d, row_start + m, col_start + n) += gradient;
                        }
                    }
                }
            }
        }
    }

    return d_input_batch;
}
