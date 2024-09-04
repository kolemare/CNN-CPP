/*
MIT License
Copyright (c) 2024 Marko Kostić

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

This project is the CNN-CPP Framework. Usage of this code is free, and 
uploading and using the code is also free, with a humble request to mention 
the origin of the implementation, the author Marko Kostić, and the repository 
link: https://github.com/kolemare/CNN-CPP.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
*/

#include "AveragePoolingLayer.hpp"

AveragePoolingLayer::AveragePoolingLayer(int pool_size,
                                         int stride)
{
    this->pool_size = pool_size;
    this->stride = stride;
}

bool AveragePoolingLayer::needsOptimizer() const
{
    return false; // Average pooling layers do not require an optimizer
}

void AveragePoolingLayer::setOptimizer(std::shared_ptr<Optimizer> optimizer)
{
    return; // No-op, as average pooling layers do not use optimizers
}

std::shared_ptr<Optimizer> AveragePoolingLayer::getOptimizer()
{
    return nullptr; // Average pooling layers do not use optimizers
}

int AveragePoolingLayer::getPoolSize()
{
    return pool_size; // Get pool size used in the average pooling operation
}

int AveragePoolingLayer::getStride()
{
    return stride; // Get stride used in the average pooling operation
}

Eigen::Tensor<double, 4> AveragePoolingLayer::forward(const Eigen::Tensor<double, 4> &input_batch)
{
    // Get dimensions of the input batch
    int batch_size = input_batch.dimension(0);
    int depth = input_batch.dimension(1);
    int input_height = input_batch.dimension(2);
    int input_width = input_batch.dimension(3);

    // Calculate output dimensions
    int output_height = (input_height - pool_size) / stride + 1;
    int output_width = (input_width - pool_size) / stride + 1;

    // Initialize the output tensor
    Eigen::Tensor<double, 4> output_batch(batch_size, depth, output_height, output_width);

    // Iterate over each image in the batch
    for (int b = 0; b < batch_size; ++b)
    {
        // Iterate over each depth channel
        for (int d = 0; d < depth; ++d)
        {
            // Iterate over each position in the output tensor
            for (int i = 0; i < output_height; ++i)
            {
                for (int j = 0; j < output_width; ++j)
                {
                    int row_start = i * stride; // Starting row for the pooling window
                    int col_start = j * stride; // Starting column for the pooling window
                    double sum = 0.0;           // Initialize the sum for the pooling window

                    // Sum over the pooling window
                    for (int m = 0; m < pool_size; ++m)
                    {
                        for (int n = 0; n < pool_size; ++n)
                        {
                            sum += input_batch(b, d, row_start + m, col_start + n);
                        }
                    }

                    // Calculate the average and assign it to the output
                    output_batch(b, d, i, j) = sum / (pool_size * pool_size);
                }
            }
        }
    }

    // Return the output batch
    return output_batch;
}

Eigen::Tensor<double, 4> AveragePoolingLayer::backward(const Eigen::Tensor<double, 4> &d_output_batch,
                                                       const Eigen::Tensor<double, 4> &input_batch,
                                                       double learning_rate)
{
    // Get dimensions of the input and output batches
    int batch_size = input_batch.dimension(0);
    int depth = input_batch.dimension(1);
    int input_height = input_batch.dimension(2);
    int input_width = input_batch.dimension(3);
    int output_height = d_output_batch.dimension(2);
    int output_width = d_output_batch.dimension(3);

    // Initialize the gradient tensor for the input
    Eigen::Tensor<double, 4> d_input_batch(batch_size, depth, input_height, input_width);
    d_input_batch.setZero(); // Initialize to zero

    // Iterate over each image in the batch
    for (int b = 0; b < batch_size; ++b)
    {
        // Iterate over each depth channel
        for (int d = 0; d < depth; ++d)
        {
            // Iterate over each position in the output tensor
            for (int i = 0; i < output_height; ++i)
            {
                for (int j = 0; j < output_width; ++j)
                {
                    int row_start = i * stride;                                             // Starting row for the pooling window
                    int col_start = j * stride;                                             // Starting column for the pooling window
                    double gradient = d_output_batch(b, d, i, j) / (pool_size * pool_size); // Calculate the gradient for the pooling window

                    // Distribute the gradient to the input positions in the pooling window
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

    // Return the gradient with respect to the input batch
    return d_input_batch;
}
