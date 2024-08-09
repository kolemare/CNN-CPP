#include "ActivationLayer.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

/**
 * @brief Constructs an ActivationLayer with a specified activation type.
 *
 * @param type The type of activation function to use.
 */
ActivationLayer::ActivationLayer(ActivationType type)
{
    this->type = type;
    // Set sensible defaults for alpha based on activation type
    if (type == ActivationType::LEAKY_RELU)
    {
        alpha = 0.01; // Default alpha for Leaky ReLU
    }
    else if (type == ActivationType::ELU)
    {
        alpha = 1.0; // Default alpha for ELU
    }
    else
    {
        alpha = 0.0; // Alpha is not used for other activation types
    }
}

/**
 * @brief Checks if the activation layer needs an optimizer.
 *
 * @return false, as activation layers do not require optimizers.
 */
bool ActivationLayer::needsOptimizer() const
{
    return false;
}

/**
 * @brief Sets an optimizer for the activation layer.
 *
 * This is a no-op as activation layers do not use optimizers.
 *
 * @param optimizer A shared pointer to an Optimizer.
 */
void ActivationLayer::setOptimizer(std::shared_ptr<Optimizer> optimizer)
{
    return;
}

/**
 * @brief Gets the optimizer for the activation layer.
 *
 * @return nullptr, since activation layers do not use optimizers.
 */
std::shared_ptr<Optimizer> ActivationLayer::getOptimizer()
{
    return nullptr;
}

/**
 * @brief Sets the alpha value for activations that use it, such as Leaky ReLU and ELU.
 *
 * @param alphaValue The alpha value to set.
 */
void ActivationLayer::setAlpha(double alphaValue)
{
    alpha = alphaValue;
}

/**
 * @brief Gets the current alpha value.
 *
 * @return The current alpha value.
 */
double ActivationLayer::getAlpha() const
{
    return alpha;
}

/**
 * @brief Performs the forward pass of the activation layer.
 *
 * @param input_batch A 4D tensor representing the input batch.
 * @return A 4D tensor representing the activated output.
 */
Eigen::Tensor<double, 4> ActivationLayer::forward(const Eigen::Tensor<double, 4> &input_batch)
{
    // Check if input is effectively 2D
    if (input_batch.dimension(1) == 1 && input_batch.dimension(2) == 1)
    {
        Eigen::Tensor<double, 2> input_2d = unwrap4DTensor(input_batch);
        Eigen::Tensor<double, 2> output_2d;

        // Apply the appropriate activation function
        switch (type)
        {
        case ActivationType::RELU:
            output_2d = relu(input_2d);
            break;
        case ActivationType::LEAKY_RELU:
            output_2d = leakyRelu(input_2d);
            break;
        case ActivationType::SIGMOID:
            output_2d = sigmoid(input_2d);
            break;
        case ActivationType::TANH:
            output_2d = tanh(input_2d);
            break;
        case ActivationType::SOFTMAX:
            output_2d = softmax(input_2d);
            break;
        case ActivationType::ELU:
            output_2d = elu(input_2d);
            break;
        default:
            throw std::invalid_argument("Unsupported activation type");
        }

        // Wrap the result back into a 4D tensor
        return wrap2DTensor(output_2d);
    }
    else
    {
        // Apply the appropriate activation function
        switch (type)
        {
        case ActivationType::RELU:
            return relu(input_batch);
        case ActivationType::LEAKY_RELU:
            return leakyRelu(input_batch);
        case ActivationType::SIGMOID:
            return sigmoid(input_batch);
        case ActivationType::TANH:
            return tanh(input_batch);
        case ActivationType::SOFTMAX:
            return softmax(input_batch);
        case ActivationType::ELU:
            return elu(input_batch);
        default:
            throw std::invalid_argument("Unsupported activation type");
        }
    }
}

/**
 * @brief Performs the backward pass of the activation layer.
 *
 * @param d_output_batch A 4D tensor of the gradient from the next layer.
 * @param input_batch A 4D tensor representing the input batch.
 * @param learning_rate The learning rate, unused here.
 * @return A 4D tensor representing the gradient with respect to the input.
 */
Eigen::Tensor<double, 4> ActivationLayer::backward(const Eigen::Tensor<double, 4> &d_output_batch,
                                                   const Eigen::Tensor<double, 4> &input_batch,
                                                   double learning_rate)
{
    // Check if input is effectively 2D
    if (input_batch.dimension(1) == 1 && input_batch.dimension(2) == 1)
    {
        Eigen::Tensor<double, 2> input_2d = unwrap4DTensor(input_batch);
        Eigen::Tensor<double, 2> d_output_2d = unwrap4DTensor(d_output_batch);
        Eigen::Tensor<double, 2> d_input_2d;

        // Calculate the derivative of the activation function
        switch (type)
        {
        case ActivationType::RELU:
            d_input_2d = d_output_2d * relu_derivative(input_2d);
            break;
        case ActivationType::LEAKY_RELU:
            d_input_2d = d_output_2d * leakyRelu_derivative(input_2d);
            break;
        case ActivationType::SIGMOID:
            d_input_2d = d_output_2d * sigmoid_derivative(input_2d);
            break;
        case ActivationType::TANH:
            d_input_2d = d_output_2d * tanh_derivative(input_2d);
            break;
        case ActivationType::SOFTMAX:
            d_input_2d = softmax_derivative(input_2d); // Simplified derivative usage
            break;
        case ActivationType::ELU:
            d_input_2d = d_output_2d * elu_derivative(input_2d);
            break;
        default:
            throw std::invalid_argument("Unsupported activation type");
        }

        // Wrap the result back into a 4D tensor
        return wrap2DTensor(d_input_2d);
    }
    else
    {
        // Calculate the derivative of the activation function
        switch (type)
        {
        case ActivationType::RELU:
            return d_output_batch * relu_derivative(input_batch);
        case ActivationType::LEAKY_RELU:
            return d_output_batch * leakyRelu_derivative(input_batch);
        case ActivationType::SIGMOID:
            return d_output_batch * sigmoid_derivative(input_batch);
        case ActivationType::TANH:
            return d_output_batch * tanh_derivative(input_batch);
        case ActivationType::SOFTMAX:
            return softmax_derivative(input_batch); // Simplified derivative usage
        case ActivationType::ELU:
            return d_output_batch * elu_derivative(input_batch);
        default:
            throw std::invalid_argument("Unsupported activation type");
        }
    }
}

/**
 * @brief Converts a 2D tensor to a 4D tensor.
 *
 * @param input A 2D tensor to convert.
 * @return A 4D tensor with the same data as input.
 */
Eigen::Tensor<double, 4> ActivationLayer::wrap2DTensor(const Eigen::Tensor<double, 2> &input)
{
    // Create a 4D tensor from a 2D tensor by adding singleton dimensions
    Eigen::Tensor<double, 4> output(input.dimension(0), 1, 1, input.dimension(1));
    for (int i = 0; i < input.dimension(0); ++i)
    {
        for (int j = 0; j < input.dimension(1); ++j)
        {
            output(i, 0, 0, j) = input(i, j);
        }
    }
    return output;
}

/**
 * @brief Converts a 4D tensor to a 2D tensor.
 *
 * @param input A 4D tensor to convert.
 * @return A 2D tensor with the same data as input.
 */
Eigen::Tensor<double, 2> ActivationLayer::unwrap4DTensor(const Eigen::Tensor<double, 4> &input)
{
    // Flatten the 4D tensor into a 2D tensor by collapsing singleton dimensions
    Eigen::Tensor<double, 2> output(input.dimension(0), input.dimension(3));
    for (int i = 0; i < input.dimension(0); ++i)
    {
        for (int j = 0; j < input.dimension(3); ++j)
        {
            output(i, j) = input(i, 0, 0, j);
        }
    }
    return output;
}

/**
 * @brief Applies the ReLU activation function element-wise on a 4D tensor.
 *
 * @param input_batch A 4D tensor.
 * @return A 4D tensor with ReLU applied.
 */
Eigen::Tensor<double, 4> ActivationLayer::relu(const Eigen::Tensor<double, 4> &input_batch)
{
    // Apply ReLU activation element-wise
    return input_batch.unaryExpr([](double x)
                                 { return std::max(0.0, x); });
}

/**
 * @brief Applies the ReLU activation function element-wise on a 2D tensor.
 *
 * @param input_batch A 2D tensor.
 * @return A 2D tensor with ReLU applied.
 */
Eigen::Tensor<double, 2> ActivationLayer::relu(const Eigen::Tensor<double, 2> &input_batch)
{
    // Apply ReLU activation element-wise
    return input_batch.unaryExpr([](double x)
                                 { return std::max(0.0, x); });
}

/**
 * @brief Applies the Leaky ReLU activation function element-wise on a 4D tensor.
 *
 * @param input_batch A 4D tensor.
 * @return A 4D tensor with Leaky ReLU applied.
 */
Eigen::Tensor<double, 4> ActivationLayer::leakyRelu(const Eigen::Tensor<double, 4> &input_batch)
{
    // Apply Leaky ReLU activation element-wise
    return input_batch.unaryExpr([this](double x)
                                 { return x > 0 ? x : alpha * x; });
}

/**
 * @brief Applies the Leaky ReLU activation function element-wise on a 2D tensor.
 *
 * @param input_batch A 2D tensor.
 * @return A 2D tensor with Leaky ReLU applied.
 */
Eigen::Tensor<double, 2> ActivationLayer::leakyRelu(const Eigen::Tensor<double, 2> &input_batch)
{
    // Apply Leaky ReLU activation element-wise
    return input_batch.unaryExpr([this](double x)
                                 { return x > 0 ? x : alpha * x; });
}

/**
 * @brief Applies the Sigmoid activation function element-wise on a 4D tensor.
 *
 * @param input_batch A 4D tensor.
 * @return A 4D tensor with Sigmoid applied.
 */
Eigen::Tensor<double, 4> ActivationLayer::sigmoid(const Eigen::Tensor<double, 4> &input_batch)
{
    // Apply Sigmoid activation element-wise
    return input_batch.unaryExpr([](double x)
                                 { return 1.0 / (1.0 + std::exp(-x)); });
}

/**
 * @brief Applies the Sigmoid activation function element-wise on a 2D tensor.
 *
 * @param input_batch A 2D tensor.
 * @return A 2D tensor with Sigmoid applied.
 */
Eigen::Tensor<double, 2> ActivationLayer::sigmoid(const Eigen::Tensor<double, 2> &input_batch)
{
    // Apply Sigmoid activation element-wise
    return input_batch.unaryExpr([](double x)
                                 { return 1.0 / (1.0 + std::exp(-x)); });
}

/**
 * @brief Applies the Tanh activation function element-wise on a 4D tensor.
 *
 * @param input_batch A 4D tensor.
 * @return A 4D tensor with Tanh applied.
 */
Eigen::Tensor<double, 4> ActivationLayer::tanh(const Eigen::Tensor<double, 4> &input_batch)
{
    // Apply Tanh activation element-wise
    return input_batch.unaryExpr([](double x)
                                 { return std::tanh(x); });
}

/**
 * @brief Applies the Tanh activation function element-wise on a 2D tensor.
 *
 * @param input_batch A 2D tensor.
 * @return A 2D tensor with Tanh applied.
 */
Eigen::Tensor<double, 2> ActivationLayer::tanh(const Eigen::Tensor<double, 2> &input_batch)
{
    // Apply Tanh activation element-wise
    return input_batch.unaryExpr([](double x)
                                 { return std::tanh(x); });
}

/**
 * @brief Applies the Softmax activation function along the last dimension of a 4D tensor.
 *
 * @param input_batch A 4D tensor.
 * @return A 4D tensor with Softmax applied.
 */
Eigen::Tensor<double, 4> ActivationLayer::softmax(const Eigen::Tensor<double, 4> &input_batch)
{
    // Softmax applied along the last dimension
    Eigen::Tensor<double, 4> output_batch = input_batch;
    for (int b = 0; b < input_batch.dimension(0); ++b)
    {
        // Unwrap the 4D tensor to 2D
        Eigen::Tensor<double, 2> input_2d = unwrap4DTensor(input_batch.chip(b, 0));

        // Calculate exponentials
        Eigen::Tensor<double, 2> exp_values = input_2d.unaryExpr([](double x)
                                                                 { return std::exp(x); });

        // Calculate sum of exponentials
        Eigen::Tensor<double, 2> sum_exp_values = exp_values.sum(Eigen::array<int, 1>{1}).reshape(Eigen::array<int, 2>{exp_values.dimension(0), 1});

        // Normalize the exponentials
        output_batch.chip(b, 0) = (exp_values / sum_exp_values.broadcast(Eigen::array<int, 2>{1, static_cast<int>(exp_values.dimension(1))})).reshape(Eigen::array<int, 4>{1, 1, 1, static_cast<int>(exp_values.dimension(1))});
    }
    return output_batch;
}

/**
 * @brief Applies the Softmax activation function along the second dimension of a 2D tensor.
 *
 * @param input_batch A 2D tensor.
 * @return A 2D tensor with Softmax applied.
 */
Eigen::Tensor<double, 2> ActivationLayer::softmax(const Eigen::Tensor<double, 2> &input_batch)
{
    // Calculate exponentials
    Eigen::Tensor<double, 2> exp_values = input_batch.unaryExpr([](double x)
                                                                { return std::exp(x); });

    // Calculate sum of exponentials
    Eigen::Tensor<double, 2> sum_exp_values = exp_values.sum(Eigen::array<int, 1>{1}).reshape(Eigen::array<int, 2>{exp_values.dimension(0), 1});

    // Normalize the exponentials
    return exp_values / sum_exp_values.broadcast(Eigen::array<int, 2>{1, static_cast<int>(exp_values.dimension(1))});
}

/**
 * @brief Applies the ELU activation function element-wise on a 4D tensor.
 *
 * @param input_batch A 4D tensor.
 * @return A 4D tensor with ELU applied.
 */
Eigen::Tensor<double, 4> ActivationLayer::elu(const Eigen::Tensor<double, 4> &input_batch)
{
    // Apply ELU activation element-wise
    return input_batch.unaryExpr([this](double x)
                                 { return x >= 0 ? x : alpha * (std::exp(x) - 1); });
}

/**
 * @brief Applies the ELU activation function element-wise on a 2D tensor.
 *
 * @param input_batch A 2D tensor.
 * @return A 2D tensor with ELU applied.
 */
Eigen::Tensor<double, 2> ActivationLayer::elu(const Eigen::Tensor<double, 2> &input_batch)
{
    // Apply ELU activation element-wise
    return input_batch.unaryExpr([this](double x)
                                 { return x >= 0 ? x : alpha * (std::exp(x) - 1); });
}

/**
 * @brief Computes the derivative of the ReLU activation function element-wise on a 4D tensor.
 *
 * @param input_batch A 4D tensor.
 * @return A 4D tensor representing the derivative.
 */
Eigen::Tensor<double, 4> ActivationLayer::relu_derivative(const Eigen::Tensor<double, 4> &input_batch)
{
    // Calculate ReLU derivative element-wise
    return input_batch.unaryExpr([](double x)
                                 { return x > 0 ? 1.0 : 0.0; });
}

/**
 * @brief Computes the derivative of the ReLU activation function element-wise on a 2D tensor.
 *
 * @param input_batch A 2D tensor.
 * @return A 2D tensor representing the derivative.
 */
Eigen::Tensor<double, 2> ActivationLayer::relu_derivative(const Eigen::Tensor<double, 2> &input_batch)
{
    // Calculate ReLU derivative element-wise
    return input_batch.unaryExpr([](double x)
                                 { return x > 0 ? 1.0 : 0.0; });
}

/**
 * @brief Computes the derivative of the Leaky ReLU activation function element-wise on a 4D tensor.
 *
 * @param input_batch A 4D tensor.
 * @return A 4D tensor representing the derivative.
 */
Eigen::Tensor<double, 4> ActivationLayer::leakyRelu_derivative(const Eigen::Tensor<double, 4> &input_batch)
{
    // Calculate Leaky ReLU derivative element-wise
    return input_batch.unaryExpr([this](double x)
                                 { return x > 0 ? 1.0 : alpha; });
}

/**
 * @brief Computes the derivative of the Leaky ReLU activation function element-wise on a 2D tensor.
 *
 * @param input_batch A 2D tensor.
 * @return A 2D tensor representing the derivative.
 */
Eigen::Tensor<double, 2> ActivationLayer::leakyRelu_derivative(const Eigen::Tensor<double, 2> &input_batch)
{
    // Calculate Leaky ReLU derivative element-wise
    return input_batch.unaryExpr([this](double x)
                                 { return x > 0 ? 1.0 : alpha; });
}

/**
 * @brief Computes the derivative of the Sigmoid activation function element-wise on a 4D tensor.
 *
 * @param input_batch A 4D tensor.
 * @return A 4D tensor representing the derivative.
 */
Eigen::Tensor<double, 4> ActivationLayer::sigmoid_derivative(const Eigen::Tensor<double, 4> &input_batch)
{
    // Calculate the sigmoid function
    Eigen::Tensor<double, 4> sig = sigmoid(input_batch);

    // Calculate Sigmoid derivative
    return sig * (1.0 - sig);
}

/**
 * @brief Computes the derivative of the Sigmoid activation function element-wise on a 2D tensor.
 *
 * @param input_batch A 2D tensor.
 * @return A 2D tensor representing the derivative.
 */
Eigen::Tensor<double, 2> ActivationLayer::sigmoid_derivative(const Eigen::Tensor<double, 2> &input_batch)
{
    // Calculate the sigmoid function
    Eigen::Tensor<double, 2> sig = sigmoid(input_batch);

    // Calculate Sigmoid derivative
    return sig * (1.0 - sig);
}

/**
 * @brief Computes the derivative of the Tanh activation function element-wise on a 4D tensor.
 *
 * @param input_batch A 4D tensor.
 * @return A 4D tensor representing the derivative.
 */
Eigen::Tensor<double, 4> ActivationLayer::tanh_derivative(const Eigen::Tensor<double, 4> &input_batch)
{
    // Calculate the tanh function
    Eigen::Tensor<double, 4> t = tanh(input_batch);

    // Calculate Tanh derivative
    return 1.0 - t * t;
}

/**
 * @brief Computes the derivative of the Tanh activation function element-wise on a 2D tensor.
 *
 * @param input_batch A 2D tensor.
 * @return A 2D tensor representing the derivative.
 */
Eigen::Tensor<double, 2> ActivationLayer::tanh_derivative(const Eigen::Tensor<double, 2> &input_batch)
{
    // Calculate the tanh function
    Eigen::Tensor<double, 2> t = tanh(input_batch);

    // Calculate Tanh derivative
    return 1.0 - t * t;
}

/**
 * @brief Computes the derivative of the Softmax activation function for use with categorical cross-entropy.
 *
 * @param input_batch A 4D tensor.
 * @return A 4D tensor representing the derivative.
 */
Eigen::Tensor<double, 4> ActivationLayer::softmax_derivative(const Eigen::Tensor<double, 4> &input_batch)
{
    // Compute the softmax output
    Eigen::Tensor<double, 4> softmax_output = softmax(input_batch);

    // Calculate derivative for each element using categorical cross-entropy simplification
    return softmax_output * (1.0 - softmax_output);
}

/**
 * @brief Computes the derivative of the Softmax activation function for use with categorical cross-entropy.
 *
 * @param input_batch A 2D tensor.
 * @return A 2D tensor representing the derivative.
 */
Eigen::Tensor<double, 2> ActivationLayer::softmax_derivative(const Eigen::Tensor<double, 2> &input_batch)
{
    // Compute the softmax output
    Eigen::Tensor<double, 2> softmax_output = softmax(input_batch);

    // Calculate derivative for each element using categorical cross-entropy simplification
    return softmax_output * (1.0 - softmax_output);
}

/**
 * @brief Computes the derivative of the ELU activation function element-wise on a 4D tensor.
 *
 * @param input_batch A 4D tensor.
 * @return A 4D tensor representing the derivative.
 */
Eigen::Tensor<double, 4> ActivationLayer::elu_derivative(const Eigen::Tensor<double, 4> &input_batch)
{
    // Calculate ELU derivative element-wise
    return input_batch.unaryExpr([this](double x)
                                 { return x >= 0 ? 1.0 : alpha * std::exp(x); });
}

/**
 * @brief Computes the derivative of the ELU activation function element-wise on a 2D tensor.
 *
 * @param input_batch A 2D tensor.
 * @return A 2D tensor representing the derivative.
 */
Eigen::Tensor<double, 2> ActivationLayer::elu_derivative(const Eigen::Tensor<double, 2> &input_batch)
{
    // Calculate ELU derivative element-wise
    return input_batch.unaryExpr([this](double x)
                                 { return x >= 0 ? 1.0 : alpha * std::exp(x); });
}
