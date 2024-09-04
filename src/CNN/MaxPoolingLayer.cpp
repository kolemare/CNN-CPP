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

#include "MaxPoolingLayer.hpp"

// Constructor to initialize the pooling layer with pool size and stride
MaxPoolingLayer::MaxPoolingLayer(int pool_size,
                                 int stride)
{
    this->pool_size = pool_size;
    this->stride = stride;
}

// Max pooling layer doesn't need an optimizer, so this function returns false
bool MaxPoolingLayer::needsOptimizer() const
{
    return false;
}

// Placeholder function for setting the optimizer, does nothing
void MaxPoolingLayer::setOptimizer(std::shared_ptr<Optimizer> optimizer)
{
    return;
}

std::shared_ptr<Optimizer> MaxPoolingLayer::getOptimizer()
{
    return nullptr;
}

// Getter for pool size
int MaxPoolingLayer::getPoolSize() const
{
    return pool_size;
}

// Getter for stride
int MaxPoolingLayer::getStride() const
{
    return stride;
}

// Forward pass for max pooling layer
Eigen::Tensor<double, 4> MaxPoolingLayer::forward(const Eigen::Tensor<double, 4> &input_batch)
{
    int batch_size = input_batch.dimension(0);   // Number of images in the batch
    int input_depth = input_batch.dimension(1);  // Depth of the input, e.g., 3 for RGB
    int input_height = input_batch.dimension(2); // Height of each image
    int input_width = input_batch.dimension(3);  // Width of each image

    // Calculate the output height and width after pooling
    int output_height = (input_height - pool_size) / stride + 1;
    int output_width = (input_width - pool_size) / stride + 1;
    if (output_height <= 0 || output_width <= 0)
    {
        throw std::invalid_argument("Invalid output size calculated, possibly due to incompatible pool size or stride.");
    }

    // Output tensor: (batch_size, input_depth, output_height, output_width)
    Eigen::Tensor<double, 4> output_batch(batch_size, input_depth, output_height, output_width);
    output_batch.setZero();

    // Store indices of max values for backpropagation
    max_indices.clear();
    max_indices.resize(batch_size, Eigen::Tensor<int, 4>(input_depth, output_height, output_width, 2));

    // Iterate over each image in the batch
    for (int b = 0; b < batch_size; ++b)
    {
        for (int d = 0; d < input_depth; ++d)
        {
            // Extract a single depth channel of the current image
            Eigen::Tensor<double, 3> input(input_height, input_width, 1);
            for (int i = 0; i < input_height; ++i)
            {
                for (int j = 0; j < input_width; ++j)
                {
                    input(i, j, 0) = input_batch(b, d, i, j);
                }
            }

            // Tensor to store the indices of max values
            Eigen::Tensor<int, 3> index(output_height, output_width, 2);
            // Perform max pooling on the current depth channel
            Eigen::Tensor<double, 3> pooled_output = maxPool(input, index);

            // Store the pooled values and indices
            for (int i = 0; i < output_height; ++i)
            {
                for (int j = 0; j < output_width; ++j)
                {
                    output_batch(b, d, i, j) = pooled_output(i, j, 0);
                    max_indices[b](d, i, j, 0) = index(i, j, 0);
                    max_indices[b](d, i, j, 1) = index(i, j, 1);
                }
            }
        }
    }

    return output_batch;
}

// Max pooling operation
Eigen::Tensor<double, 3> MaxPoolingLayer::maxPool(const Eigen::Tensor<double, 3> &input,
                                                  Eigen::Tensor<int, 3> &indices)
{
    int input_height = input.dimension(0);                       // Height of the input tensor
    int input_width = input.dimension(1);                        // Width of the input tensor
    int output_height = (input_height - pool_size) / stride + 1; // Height of the output tensor after pooling
    int output_width = (input_width - pool_size) / stride + 1;   // Width of the output tensor after pooling

    // Output tensor: (output_height, output_width, 1)
    Eigen::Tensor<double, 3> output(output_height, output_width, 1);
    output.setZero();

    // Perform max pooling
    for (int i = 0; i < output_height; ++i)
    {
        for (int j = 0; j < output_width; ++j)
        {
            int row_start = i * stride; // Starting row of the current pooling window
            int col_start = j * stride; // Starting column of the current pooling window
            double max_val = -std::numeric_limits<double>::infinity();
            int max_row = -1, max_col = -1;

            // Iterate over the pooling window
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

            output(i, j, 0) = max_val;  // Store the max value in the output tensor
            indices(i, j, 0) = max_row; // Store the row index of the max value
            indices(i, j, 1) = max_col; // Store the column index of the max value
        }
    }

    return output;
}

// Backward pass for max pooling layer
Eigen::Tensor<double, 4> MaxPoolingLayer::backward(const Eigen::Tensor<double, 4> &d_output_batch,
                                                   const Eigen::Tensor<double, 4> &input_batch,
                                                   double learning_rate)
{
    int batch_size = d_output_batch.dimension(0);    // Number of images in the batch
    int input_depth = d_output_batch.dimension(1);   // Depth of the input, e.g., 3 for RGB
    int input_height = input_batch.dimension(2);     // Height of each image
    int input_width = input_batch.dimension(3);      // Width of each image
    int output_height = d_output_batch.dimension(2); // Height of the output tensor
    int output_width = d_output_batch.dimension(3);  // Width of the output tensor

    // Gradient tensor for the input batch
    Eigen::Tensor<double, 4> d_input_batch(batch_size, input_depth, input_height, input_width);
    d_input_batch.setZero();

    // Iterate over each image in the batch
    for (int b = 0; b < batch_size; ++b)
    {
        for (int d = 0; d < input_depth; ++d)
        {
            // Tensor to store the indices of max values
            Eigen::Tensor<int, 3> index(output_height, output_width, 2);
            for (int i = 0; i < output_height; ++i)
            {
                for (int j = 0; j < output_width; ++j)
                {
                    index(i, j, 0) = max_indices[b](d, i, j, 0);
                    index(i, j, 1) = max_indices[b](d, i, j, 1);
                }
            }

            // Tensor to store the gradient of the output
            Eigen::Tensor<double, 3> d_output(output_height, output_width, 1);
            for (int i = 0; i < output_height; ++i)
            {
                for (int j = 0; j < output_width; ++j)
                {
                    d_output(i, j, 0) = d_output_batch(b, d, i, j);
                }
            }

            // Calculate the gradient of the input
            Eigen::Tensor<double, 3> d_input = maxPoolBackward(d_output, index);
            for (int i = 0; i < input_height; ++i)
            {
                for (int j = 0; j < input_width; ++j)
                {
                    d_input_batch(b, d, i, j) = d_input(i, j, 0);
                }
            }
        }
    }

    return d_input_batch;
}

// Backward pooling operation
Eigen::Tensor<double, 3> MaxPoolingLayer::maxPoolBackward(const Eigen::Tensor<double, 3> &d_output,
                                                          const Eigen::Tensor<int, 3> &indices)
{
    int input_height = indices.dimension(0) * stride + pool_size - stride; // Height of the input tensor
    int input_width = indices.dimension(1) * stride + pool_size - stride;  // Width of the input tensor
    int output_height = d_output.dimension(0);                             // Height of the output tensor
    int output_width = d_output.dimension(1);                              // Width of the output tensor
    Eigen::Tensor<double, 3> d_input(input_height, input_width, 1);
    d_input.setZero();

    // Propagate the gradients back to the input
    for (int i = 0; i < output_height; ++i)
    {
        for (int j = 0; j < output_width; ++j)
        {
            int row = indices(i, j, 0);                // Row index of the max value in the input
            int col = indices(i, j, 1);                // Column index of the max value in the input
            d_input(row, col, 0) += d_output(i, j, 0); // Accumulate the gradient for the input
        }
    }

    return d_input;
}
