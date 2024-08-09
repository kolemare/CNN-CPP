#include "AveragePoolingLayer.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

/**
 * @brief Constructor to initialize an AveragePoolingLayer with a specific pool size and stride.
 *
 * @param pool_size Size of the pooling window.
 * @param stride Stride of the pooling operation.
 */
AveragePoolingLayer::AveragePoolingLayer(int pool_size,
                                         int stride)
{
    this->pool_size = pool_size;
    this->stride = stride;
}

/**
 * @brief Indicates whether the layer needs an optimizer.
 *
 * @return False, as average pooling layers do not require an optimizer.
 */
bool AveragePoolingLayer::needsOptimizer() const
{
    return false;
}

/**
 * @brief Sets the optimizer for the layer.
 *
 * Average pooling layers do not use optimizers, so this function does nothing.
 *
 * @param optimizer A shared pointer to the optimizer.
 */
void AveragePoolingLayer::setOptimizer(std::shared_ptr<Optimizer> optimizer)
{
    return;
}

/**
 * @brief Gets the optimizer associated with the layer.
 *
 * @return A nullptr, as average pooling layers do not use optimizers.
 */
std::shared_ptr<Optimizer> AveragePoolingLayer::getOptimizer()
{
    return nullptr;
}

/**
 * @brief Gets the pool size used in the average pooling operation.
 *
 * @return The size of the pooling window.
 */
int AveragePoolingLayer::getPoolSize()
{
    return pool_size;
}

/**
 * @brief Gets the stride used in the average pooling operation.
 *
 * @return The stride of the pooling operation.
 */
int AveragePoolingLayer::getStride()
{
    return stride;
}

/**
 * @brief Performs the forward pass using average pooling on the input batch.
 *
 * @param input_batch A 4D tensor representing the input batch.
 * @return A 4D tensor with the pooled output.
 */
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

/**
 * @brief Performs the backward pass, computing gradients with respect to the input batch.
 *
 * @param d_output_batch A 4D tensor of gradients with respect to the output.
 * @param input_batch A 4D tensor representing the original input batch.
 * @param learning_rate The learning rate used for weight updates (not used here).
 * @return A 4D tensor of gradients with respect to the input.
 */
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
