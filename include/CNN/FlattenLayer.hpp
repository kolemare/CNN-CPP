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

#ifndef FLATTENLAYER_HPP
#define FLATTENLAYER_HPP

#include "Layer.hpp"

/**
 * @class FlattenLayer
 * @brief A neural network layer that flattens its input from 4D to 2D, preserving the batch size.
 *
 * The FlattenLayer class reshapes its input tensor from four dimensions (batch size, depth, height, width)
 * to a two-dimensional tensor (batch size, flattened size) during the forward pass. It reshapes the tensor
 * back to its original dimensions during the backward pass.
 */
class FlattenLayer : public Layer
{
public:
    /**
     * @brief Perform the forward pass to flatten the input tensor.
     *
     * This method takes a 4D tensor input and reshapes it into a 2D tensor
     * while keeping track of the original dimensions for use during the backward pass.
     *
     * @param input_batch The input tensor with dimensions (batch_size, depth, height, width).
     * @return Eigen::Tensor<double, 4> The flattened tensor with dimensions (batch_size, 1, 1, flattened_size).
     */
    Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4> &input_batch) override;

    /**
     * @brief Perform the backward pass to reshape the gradient tensor.
     *
     * This method reshapes the gradient tensor from the 2D flattened shape back to
     * its original 4D dimensions using the stored original dimensions.
     *
     * @param d_output_batch The gradient tensor with dimensions (batch_size, 1, 1, flattened_size).
     * @param input_batch The original input tensor (not used in this layer).
     * @param learning_rate The learning rate (not used in this layer).
     * @return Eigen::Tensor<double, 4> The reshaped gradient tensor with original dimensions (batch_size, depth, height, width).
     */
    Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4> &d_output_batch,
                                      const Eigen::Tensor<double, 4> &input_batch,
                                      double learning_rate) override;

    /**
     * @brief Check if the layer requires an optimizer.
     *
     * The flatten layer does not have any trainable parameters and thus does not require an optimizer.
     *
     * @return false Always returns false as no optimizer is needed.
     */
    bool needsOptimizer() const override;

    /**
     * @brief Set an optimizer for the layer.
     *
     * This function is a placeholder and does nothing for the flatten layer.
     *
     * @param optimizer Shared pointer to an optimizer (not used).
     */
    void setOptimizer(std::shared_ptr<Optimizer> optimizer) override;

    /**
     * @brief Get the optimizer used by the layer.
     *
     * The flatten layer does not use an optimizer, so this function returns nullptr.
     *
     * @return std::shared_ptr<Optimizer> Always returns nullptr.
     */
    std::shared_ptr<Optimizer> getOptimizer() override;

private:
    int batch_size;                       ///< The batch size of the input tensor.
    std::vector<int> original_dimensions; ///< The original dimensions of the input tensor for reshaping in the backward pass.
};

#endif // FLATTENLAYER_HPP
