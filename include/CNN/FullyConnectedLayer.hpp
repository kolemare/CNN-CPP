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

#ifndef FULLYCONNECTEDLAYER_HPP
#define FULLYCONNECTEDLAYER_HPP

#include "Layer.hpp"

/**
 * @brief Class representing a fully connected (dense) layer in a neural network.
 *
 * This class provides the implementation of a fully connected layer,
 * including methods for forward and backward propagation, as well as
 * methods to manage the layer's weights, biases, and optimizers.
 */
class FullyConnectedLayer : public Layer
{
public:
    /**
     * @brief Construct a new FullyConnectedLayer object.
     *
     * Initializes the fully connected layer with the specified number of output neurons,
     * weight initialization method, bias initialization method, and random seed.
     *
     * @param output_size The number of output neurons in the layer.
     * @param weight_init The method to initialize the weights (default is Xavier).
     * @param bias_init The method to initialize the biases (default is Zero).
     * @param seed The random seed for initializing weights and biases (default is 42).
     */
    FullyConnectedLayer(int output_size,
                        DenseWeightInitialization weight_init = DenseWeightInitialization::XAVIER,
                        DenseBiasInitialization bias_init = DenseBiasInitialization::ZERO,
                        unsigned int seed = 42);

    /**
     * @brief Perform the forward pass for the fully connected layer.
     *
     * Computes the output of the layer by performing matrix multiplication between the input
     * and weights, then adds the biases.
     *
     * @param input_batch The input tensor with dimensions (batch_size, depth, height, width).
     * @return Eigen::Tensor<double, 4> The output tensor with dimensions (batch_size, 1, 1, output_size).
     */
    Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4> &input_batch) override;

    /**
     * @brief Perform the backward pass for the fully connected layer.
     *
     * Computes the gradients with respect to the weights, biases, and input, and updates
     * the weights and biases using the optimizer.
     *
     * @param d_output_batch The gradient tensor with dimensions (batch_size, 1, 1, output_size).
     * @param input_batch The original input tensor with dimensions (batch_size, depth, height, width).
     * @param learning_rate The learning rate for updating the weights and biases.
     * @return Eigen::Tensor<double, 4> The gradient tensor with dimensions (batch_size, depth, height, width).
     */
    Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4> &d_output_batch,
                                      const Eigen::Tensor<double, 4> &input_batch,
                                      double learning_rate) override;

    /**
     * @brief Set the weights of the layer.
     *
     * Assigns new values to the weights of the layer, ensuring the dimensions match
     * the expected shape.
     *
     * @param new_weights The new weights tensor with dimensions (output_size, 1, 1, input_size).
     */
    void setWeights(const Eigen::Tensor<double, 4> &new_weights);

    /**
     * @brief Get the weights of the layer.
     *
     * @return Eigen::Tensor<double, 4> A copy of the weights tensor.
     */
    Eigen::Tensor<double, 4> getWeights() const;

    /**
     * @brief Set the biases of the layer.
     *
     * Assigns new values to the biases of the layer, ensuring the dimensions match
     * the expected shape.
     *
     * @param new_biases The new biases tensor with dimensions (output_size).
     */
    void setBiases(const Eigen::Tensor<double, 1> &new_biases);

    /**
     * @brief Get the biases of the layer.
     *
     * @return Eigen::Tensor<double, 1> A copy of the biases tensor.
     */
    Eigen::Tensor<double, 1> getBiases() const;

    /**
     * @brief Set the input size and initialize weights and biases.
     *
     * This function sets the input size for the layer and initializes the weights
     * and biases based on the specified initialization methods.
     *
     * @param input_size The size of the input to the layer.
     */
    void setInputSize(int input_size);

    /**
     * @brief Get the output size of the layer.
     *
     * @return int The number of output neurons in the layer.
     */
    int getOutputSize() const;

    /**
     * @brief Check if the layer requires an optimizer.
     *
     * Fully connected layers typically require an optimizer for training.
     *
     * @return true if the layer needs an optimizer, otherwise false.
     */
    bool needsOptimizer() const override;

    /**
     * @brief Set the optimizer for the layer.
     *
     * @param optimizer A shared pointer to the optimizer object.
     */
    void setOptimizer(std::shared_ptr<Optimizer> optimizer) override;

    /**
     * @brief Get the optimizer used by the layer.
     *
     * @return std::shared_ptr<Optimizer> The optimizer used by the layer.
     */
    std::shared_ptr<Optimizer> getOptimizer() override;

private:
    int input_size;                   ///< Input size of the layer
    int output_size;                  ///< Output size of the layer
    Eigen::Tensor<double, 4> weights; ///< Weights as a 4D tensor (output_size, 1, 1, input_size)
    Eigen::Tensor<double, 1> biases;  ///< Biases as a 1D tensor (output_size)

    DenseWeightInitialization weight_init; ///< Weight initialization method
    DenseBiasInitialization bias_init;     ///< Bias initialization method
    unsigned int seed;                     ///< Seed for random number generation

    std::shared_ptr<Optimizer> optimizer; ///< Optimizer

    /**
     * @brief Initialize the weights of the layer.
     *
     * This function initializes the weights using the specified initialization method.
     */
    void initializeWeights();

    /**
     * @brief Initialize the biases of the layer.
     *
     * This function initializes the biases using the specified initialization method.
     */
    void initializeBiases();
};

#endif // FULLYCONNECTEDLAYER_HPP
