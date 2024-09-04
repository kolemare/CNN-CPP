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

#ifndef MAX_POOLING_LAYER_HPP
#define MAX_POOLING_LAYER_HPP

#include "Layer.hpp"

/**
 * @class MaxPoolingLayer
 * @brief A layer that performs max pooling operations on the input data.
 *
 * The MaxPoolingLayer reduces the spatial dimensions of the input by taking
 * the maximum value over a specified window. It does not require an optimizer
 * as there are no trainable parameters.
 */
class MaxPoolingLayer : public Layer
{
public:
    /**
     * @brief Constructs a MaxPoolingLayer with the given pool size and stride.
     *
     * @param pool_size Size of the pooling window.
     * @param stride Stride of the pooling operation.
     */
    MaxPoolingLayer(int pool_size,
                    int stride);

    /**
     * @brief Performs the forward pass of the max pooling layer.
     *
     * This function reduces the dimensions of the input by taking the maximum
     * value over a specified window.
     *
     * @param input_batch A 4D tensor representing the input batch.
     * @return A 4D tensor containing the pooled output.
     */
    Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4> &input_batch) override;

    /**
     * @brief Performs the backward pass, computing gradients with respect to the input batch.
     *
     * This function computes the gradient of the loss with respect to the input,
     * propagating the gradients back through the max pooling operation.
     *
     * @param d_output_batch A 4D tensor of gradients with respect to the output.
     * @param input_batch A 4D tensor representing the original input batch.
     * @param learning_rate The learning rate used for weight updates (not used here).
     * @return A 4D tensor of gradients with respect to the input.
     */
    Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4> &d_output_batch,
                                      const Eigen::Tensor<double, 4> &input_batch,
                                      double learning_rate) override;

    /**
     * @brief Indicates whether the layer needs an optimizer.
     *
     * @return false, as max pooling layers do not require an optimizer.
     */
    bool needsOptimizer() const override;

    /**
     * @brief Sets the optimizer for the layer (not used in max pooling).
     *
     * This function is a placeholder and does nothing for max pooling layers
     * as they do not use an optimizer.
     *
     * @param optimizer Shared pointer to an optimizer.
     */
    void setOptimizer(std::shared_ptr<Optimizer> optimizer) override;

    /**
     * @brief Gets the optimizer associated with the layer.
     *
     * @return nullptr, as max pooling layers do not use optimizers.
     */
    std::shared_ptr<Optimizer> getOptimizer() override;

    /**
     * @brief Gets the pool size used in the max pooling operation.
     *
     * @return The size of the pooling window.
     */
    int getPoolSize() const;

    /**
     * @brief Gets the stride used in the max pooling operation.
     *
     * @return The stride of the pooling operation.
     */
    int getStride() const;

private:
    int pool_size; ///< The size of the pooling window.
    int stride;    ///< The stride of the pooling operation.

    /**
     * @brief Stores indices of maximum values during the forward pass for use in backpropagation.
     */
    std::vector<Eigen::Tensor<int, 4>> max_indices;

    /**
     * @brief Performs the max pooling operation on a single input channel.
     *
     * This function calculates the maximum values and their indices over the
     * pooling window for a single channel.
     *
     * @param input A 3D tensor representing a single input channel.
     * @param indices A 3D tensor to store the indices of maximum values.
     * @return A 3D tensor containing the pooled output.
     */
    Eigen::Tensor<double, 3> maxPool(const Eigen::Tensor<double, 3> &input,
                                     Eigen::Tensor<int, 3> &indices);

    /**
     * @brief Performs the backward pass of the max pooling operation.
     *
     * This function computes the gradient of the loss with respect to the input
     * by propagating gradients back through the max pooling operation.
     *
     * @param d_output A 3D tensor of gradients with respect to the output.
     * @param indices A 3D tensor containing the indices of maximum values from the forward pass.
     * @return A 3D tensor of gradients with respect to the input.
     */
    Eigen::Tensor<double, 3> maxPoolBackward(const Eigen::Tensor<double, 3> &d_output,
                                             const Eigen::Tensor<int, 3> &indices);
};

#endif // MAX_POOLING_LAYER_HPP
