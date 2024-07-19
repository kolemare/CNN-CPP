#include "MaxPoolingLayer.hpp"
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cmath>

MaxPoolingLayer::MaxPoolingLayer(int pool_size, int stride)
    : pool_size(pool_size), stride(stride) {}

bool MaxPoolingLayer::needsOptimizer() const
{
    return false;
}

void MaxPoolingLayer::setOptimizer(std::unishared_ptrque_ptr<Optimizer> optimizer)
{
    return;
}

int MaxPoolingLayer::getPoolSize() const
{
    return pool_size;
}

int MaxPoolingLayer::getStride() const
{
    return stride;
}

Eigen::Tensor<double, 4> MaxPoolingLayer::forward(const Eigen::Tensor<double, 4> &input_batch)
{
    int batch_size = input_batch.dimension(0);
    int input_depth = input_batch.dimension(1);
    int input_height = input_batch.dimension(2);
    int input_width = input_batch.dimension(3);

    int output_height = (input_height - pool_size) / stride + 1;
    int output_width = (input_width - pool_size) / stride + 1;
    if (output_height <= 0 || output_width <= 0)
    {
        throw std::invalid_argument("Invalid output size calculated, possibly due to incompatible pool size or stride.");
    }

    Eigen::Tensor<double, 4> output_batch(batch_size, input_depth, output_height, output_width);
    output_batch.setZero();

    max_indices.clear();
    max_indices.resize(batch_size, Eigen::Tensor<int, 4>(input_depth, output_height, output_width, 2));

    for (int b = 0; b < batch_size; ++b)
    {
        for (int d = 0; d < input_depth; ++d)
        {
            Eigen::Tensor<double, 3> input = input_batch.chip(b, 0).chip(d, 0);
            Eigen::Tensor<int, 3> index(output_height, output_width, 2);
            Eigen::Tensor<double, 3> pooled_output = maxPool(input, index);
            output_batch.chip(b, 0).chip(d, 0) = pooled_output;
            max_indices[b].chip(d, 0) = index;
        }
    }

    return output_batch;
}

Eigen::Tensor<double, 3> MaxPoolingLayer::maxPool(const Eigen::Tensor<double, 3> &input, Eigen::Tensor<int, 3> &indices)
{
    int input_height = input.dimension(0);
    int input_width = input.dimension(1);
    int output_height = (input_height - pool_size) / stride + 1;
    int output_width = (input_width - pool_size) / stride + 1;

    if (output_height <= 0 || output_width <= 0)
    {
        throw std::invalid_argument("Invalid output size calculated in maxPool, possibly due to incompatible pool size or stride.");
    }

    Eigen::Tensor<double, 3> output(output_height, output_width, 1);
    output.setZero();

    for (int i = 0; i < output_height; ++i)
    {
        for (int j = 0; j < output_width; ++j)
        {
            int row_start = i * stride;
            int col_start = j * stride;
            double max_val = -std::numeric_limits<double>::infinity();
            int max_row = -1, max_col = -1;

            for (int r = 0; r < pool_size; ++r)
            {
                for (int c = 0; c < pool_size; ++c)
                {
                    int row = row_start + r;
                    int col = col_start + c;
                    if (input(row, col, 0) > max_val)
                    {
                        max_val = input(row, col, 0);
                        max_row = row;
                        max_col = col;
                    }
                }
            }

            output(i, j, 0) = max_val;
            indices(i, j, 0) = max_row;
            indices(i, j, 1) = max_col;
        }
    }

    return output;
}

Eigen::Tensor<double, 4> MaxPoolingLayer::backward(const Eigen::Tensor<double, 4> &d_output_batch, const Eigen::Tensor<double, 4> &input_batch, double learning_rate)
{
    int batch_size = d_output_batch.dimension(0);
    int input_depth = d_output_batch.dimension(1);
    int input_height = input_batch.dimension(2);
    int input_width = input_batch.dimension(3);
    int output_height = d_output_batch.dimension(2);
    int output_width = d_output_batch.dimension(3);

    Eigen::Tensor<double, 4> d_input_batch(batch_size, input_depth, input_height, input_width);
    d_input_batch.setZero();

    for (int b = 0; b < batch_size; ++b)
    {
        for (int d = 0; d < input_depth; ++d)
        {
            Eigen::Tensor<int, 3> index = max_indices[b].chip(d, 0);
            Eigen::Tensor<double, 3> d_output = d_output_batch.chip(b, 0).chip(d, 0);

            Eigen::Tensor<double, 3> d_input = maxPoolBackward(d_output, index);
            d_input_batch.chip(b, 0).chip(d, 0) = d_input;
        }
    }

    return d_input_batch;
}

Eigen::Tensor<double, 3> MaxPoolingLayer::maxPoolBackward(const Eigen::Tensor<double, 3> &d_output, const Eigen::Tensor<int, 3> &indices)
{
    int input_height = indices.dimension(0) * stride + pool_size - stride;
    int input_width = indices.dimension(1) * stride + pool_size - stride;
    int output_height = d_output.dimension(0);
    int output_width = d_output.dimension(1);
    Eigen::Tensor<double, 3> d_input(input_height, input_width, 1);
    d_input.setZero();

    for (int i = 0; i < output_height; ++i)
    {
        for (int j = 0; j < output_width; ++j)
        {
            int row = indices(i, j, 0);
            int col = indices(i, j, 1);
            d_input(row, col, 0) += d_output(i, j, 0);
        }
    }

    return d_input;
}
