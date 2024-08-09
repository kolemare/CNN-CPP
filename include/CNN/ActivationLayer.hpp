#ifndef ACTIVATIONLAYER_HPP
#define ACTIVATIONLAYER_HPP

#include <unsupported/Eigen/CXX11/Tensor>
#include "Layer.hpp"
#include "Optimizer.hpp"
#include <iostream>

/**
 * @class ActivationLayer
 * @brief Implements various activation functions for a neural network layer.
 *
 * This class provides several common activation functions including ReLU, Leaky ReLU,
 * Sigmoid, Tanh, Softmax, and ELU. It supports both forward and backward propagation,
 * allowing the calculation of gradients for backpropagation.
 */
class ActivationLayer : public Layer
{
public:
    /**
     * @brief Constructs an ActivationLayer with a specified activation type.
     *
     * @param type The activation function type to use (e.g., RELU, SIGMOID).
     */
    ActivationLayer(ActivationType type);

    /**
     * @brief Performs the forward pass using the activation function on the input batch.
     *
     * @param input_batch A 4D tensor representing the input batch.
     * @return A 4D tensor with the activation function applied.
     */
    Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4> &input_batch) override;

    /**
     * @brief Performs the backward pass of the activation layer.
     *
     * @param d_output_batch A 4D tensor of the gradient from the next layer.
     * @param input_batch A 4D tensor representing the input batch.
     * @param learning_rate The learning rate, unused here.
     * @return A 4D tensor representing the gradient with respect to the input.
     */
    Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4> &d_output_batch,
                                      const Eigen::Tensor<double, 4> &input_batch,
                                      double learning_rate) override;

    /**
     * @brief Checks if the layer requires an optimizer.
     *
     * @return False, as activation layers do not require an optimizer.
     */
    bool needsOptimizer() const override;

    /**
     * @brief Sets the optimizer for the layer.
     *
     * Activation layers do not use optimizers, so this function does nothing.
     *
     * @param optimizer A shared pointer to the optimizer.
     */
    void setOptimizer(std::shared_ptr<Optimizer> optimizer) override;

    /**
     * @brief Gets the optimizer associated with the layer.
     *
     * @return A nullptr, as activation layers do not use optimizers.
     */
    std::shared_ptr<Optimizer> getOptimizer() override;

    /**
     * @brief Sets the alpha parameter for activation functions like Leaky ReLU and ELU.
     *
     * @param alphaValue The alpha value to set.
     */
    void setAlpha(double alphaValue);

    /**
     * @brief Gets the current alpha value.
     *
     * @return The current alpha value.
     */
    double getAlpha() const;

private:
    ActivationType type; ///< The type of activation function used.
    double alpha;        ///< Alpha parameter for Leaky ReLU and ELU.

    // Utility methods for tensor manipulation
    Eigen::Tensor<double, 4> wrap2DTensor(const Eigen::Tensor<double, 2> &input);
    Eigen::Tensor<double, 2> unwrap4DTensor(const Eigen::Tensor<double, 4> &input);

    // Activation functions
    Eigen::Tensor<double, 4> relu(const Eigen::Tensor<double, 4> &input_batch);
    Eigen::Tensor<double, 4> leakyRelu(const Eigen::Tensor<double, 4> &input_batch);
    Eigen::Tensor<double, 4> sigmoid(const Eigen::Tensor<double, 4> &input_batch);
    Eigen::Tensor<double, 4> tanh(const Eigen::Tensor<double, 4> &input_batch);
    Eigen::Tensor<double, 4> softmax(const Eigen::Tensor<double, 4> &input_batch);
    Eigen::Tensor<double, 4> elu(const Eigen::Tensor<double, 4> &input_batch);

    Eigen::Tensor<double, 2> relu(const Eigen::Tensor<double, 2> &input_batch);
    Eigen::Tensor<double, 2> leakyRelu(const Eigen::Tensor<double, 2> &input_batch);
    Eigen::Tensor<double, 2> sigmoid(const Eigen::Tensor<double, 2> &input_batch);
    Eigen::Tensor<double, 2> tanh(const Eigen::Tensor<double, 2> &input_batch);
    Eigen::Tensor<double, 2> softmax(const Eigen::Tensor<double, 2> &input_batch);
    Eigen::Tensor<double, 2> elu(const Eigen::Tensor<double, 2> &input_batch);

    // Derivatives of activation functions
    Eigen::Tensor<double, 4> relu_derivative(const Eigen::Tensor<double, 4> &input_batch);
    Eigen::Tensor<double, 4> leakyRelu_derivative(const Eigen::Tensor<double, 4> &input_batch);
    Eigen::Tensor<double, 4> sigmoid_derivative(const Eigen::Tensor<double, 4> &input_batch);
    Eigen::Tensor<double, 4> tanh_derivative(const Eigen::Tensor<double, 4> &input_batch);
    Eigen::Tensor<double, 4> softmax_derivative(const Eigen::Tensor<double, 4> &input_batch);
    Eigen::Tensor<double, 4> elu_derivative(const Eigen::Tensor<double, 4> &input_batch);

    Eigen::Tensor<double, 2> relu_derivative(const Eigen::Tensor<double, 2> &input_batch);
    Eigen::Tensor<double, 2> leakyRelu_derivative(const Eigen::Tensor<double, 2> &input_batch);
    Eigen::Tensor<double, 2> sigmoid_derivative(const Eigen::Tensor<double, 2> &input_batch);
    Eigen::Tensor<double, 2> tanh_derivative(const Eigen::Tensor<double, 2> &input_batch);
    Eigen::Tensor<double, 2> softmax_derivative(const Eigen::Tensor<double, 2> &input_batch);
    Eigen::Tensor<double, 2> elu_derivative(const Eigen::Tensor<double, 2> &input_batch);
};

#endif // ACTIVATIONLAYER_HPP
