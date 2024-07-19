#include "ActivationLayer.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

ActivationLayer::ActivationLayer(ActivationType type) : type(type)
{
    // Set sensible defaults for alpha
    if (type == LEAKY_RELU)
    {
        alpha = 0.01;
    }
    else if (type == ELU)
    {
        alpha = 1.0;
    }
    else
    {
        alpha = 0.0; // Not used for other types
    }
}

bool ActivationLayer::needsOptimizer() const
{
    return false;
}

void ActivationLayer::setOptimizer(std::shared_ptr<Optimizer> optimizer)
{
    return;
}

void ActivationLayer::setAlpha(double alphaValue)
{
    alpha = alphaValue;
}

double ActivationLayer::getAlpha() const
{
    return alpha;
}

Eigen::Tensor<double, 4> ActivationLayer::forward(const Eigen::Tensor<double, 4> &input_batch)
{
    if (input_batch.dimension(1) == 1 && input_batch.dimension(2) == 1)
    {
        Eigen::Tensor<double, 2> input_2d = unwrap4DTensor(input_batch);
        Eigen::Tensor<double, 2> output_2d;

        switch (type)
        {
        case RELU:
            output_2d = relu(input_2d);
            break;
        case LEAKY_RELU:
            output_2d = leakyRelu(input_2d);
            break;
        case SIGMOID:
            output_2d = sigmoid(input_2d);
            break;
        case TANH:
            output_2d = tanh(input_2d);
            break;
        case SOFTMAX:
            output_2d = softmax(input_2d);
            break;
        case ELU:
            output_2d = elu(input_2d);
            break;
        default:
            throw std::invalid_argument("Unsupported activation type");
        }

        return wrap2DTensor(output_2d);
    }
    else
    {
        switch (type)
        {
        case RELU:
            return relu(input_batch);
        case LEAKY_RELU:
            return leakyRelu(input_batch);
        case SIGMOID:
            return sigmoid(input_batch);
        case TANH:
            return tanh(input_batch);
        case SOFTMAX:
            return softmax(input_batch);
        case ELU:
            return elu(input_batch);
        default:
            throw std::invalid_argument("Unsupported activation type");
        }
    }
}

Eigen::Tensor<double, 4> ActivationLayer::backward(const Eigen::Tensor<double, 4> &d_output_batch, const Eigen::Tensor<double, 4> &input_batch, double learning_rate)
{
    if (input_batch.dimension(1) == 1 && input_batch.dimension(2) == 1)
    {
        Eigen::Tensor<double, 2> input_2d = unwrap4DTensor(input_batch);
        Eigen::Tensor<double, 2> d_output_2d = unwrap4DTensor(d_output_batch);
        Eigen::Tensor<double, 2> d_input_2d;

        switch (type)
        {
        case RELU:
            d_input_2d = d_output_2d * relu_derivative(input_2d);
            break;
        case LEAKY_RELU:
            d_input_2d = d_output_2d * leakyRelu_derivative(input_2d);
            break;
        case SIGMOID:
            d_input_2d = d_output_2d * sigmoid_derivative(input_2d);
            break;
        case TANH:
            d_input_2d = d_output_2d * tanh_derivative(input_2d);
            break;
        case SOFTMAX:
            d_input_2d = d_output_2d * softmax_derivative(input_2d);
            break;
        case ELU:
            d_input_2d = d_output_2d * elu_derivative(input_2d);
            break;
        default:
            throw std::invalid_argument("Unsupported activation type");
        }

        return wrap2DTensor(d_input_2d);
    }
    else
    {
        switch (type)
        {
        case RELU:
            return d_output_batch * relu_derivative(input_batch);
        case LEAKY_RELU:
            return d_output_batch * leakyRelu_derivative(input_batch);
        case SIGMOID:
            return d_output_batch * sigmoid_derivative(input_batch);
        case TANH:
            return d_output_batch * tanh_derivative(input_batch);
        case SOFTMAX:
            return d_output_batch * softmax_derivative(input_batch);
        case ELU:
            return d_output_batch * elu_derivative(input_batch);
        default:
            throw std::invalid_argument("Unsupported activation type");
        }
    }
}

Eigen::Tensor<double, 4> ActivationLayer::wrap2DTensor(const Eigen::Tensor<double, 2> &input)
{
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

Eigen::Tensor<double, 2> ActivationLayer::unwrap4DTensor(const Eigen::Tensor<double, 4> &input)
{
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

Eigen::Tensor<double, 4> ActivationLayer::relu(const Eigen::Tensor<double, 4> &input_batch)
{
    return input_batch.unaryExpr([](double x)
                                 { return std::max(0.0, x); });
}

Eigen::Tensor<double, 2> ActivationLayer::relu(const Eigen::Tensor<double, 2> &input_batch)
{
    return input_batch.unaryExpr([](double x)
                                 { return std::max(0.0, x); });
}

Eigen::Tensor<double, 4> ActivationLayer::leakyRelu(const Eigen::Tensor<double, 4> &input_batch)
{
    return input_batch.unaryExpr([this](double x)
                                 { return x > 0 ? x : alpha * x; });
}

Eigen::Tensor<double, 2> ActivationLayer::leakyRelu(const Eigen::Tensor<double, 2> &input_batch)
{
    return input_batch.unaryExpr([this](double x)
                                 { return x > 0 ? x : alpha * x; });
}

Eigen::Tensor<double, 4> ActivationLayer::sigmoid(const Eigen::Tensor<double, 4> &input_batch)
{
    return input_batch.unaryExpr([](double x)
                                 { return 1.0 / (1.0 + std::exp(-x)); });
}

Eigen::Tensor<double, 2> ActivationLayer::sigmoid(const Eigen::Tensor<double, 2> &input_batch)
{
    return input_batch.unaryExpr([](double x)
                                 { return 1.0 / (1.0 + std::exp(-x)); });
}

Eigen::Tensor<double, 4> ActivationLayer::tanh(const Eigen::Tensor<double, 4> &input_batch)
{
    return input_batch.unaryExpr([](double x)
                                 { return std::tanh(x); });
}

Eigen::Tensor<double, 2> ActivationLayer::tanh(const Eigen::Tensor<double, 2> &input_batch)
{
    return input_batch.unaryExpr([](double x)
                                 { return std::tanh(x); });
}

Eigen::Tensor<double, 4> ActivationLayer::softmax(const Eigen::Tensor<double, 4> &input_batch)
{
    return input_batch;
}

Eigen::Tensor<double, 2> ActivationLayer::softmax(const Eigen::Tensor<double, 2> &input_batch)
{
    return input_batch;
}

Eigen::Tensor<double, 4> ActivationLayer::elu(const Eigen::Tensor<double, 4> &input_batch)
{
    return input_batch.unaryExpr([this](double x)
                                 { return x >= 0 ? x : alpha * (std::exp(x) - 1); });
}

Eigen::Tensor<double, 2> ActivationLayer::elu(const Eigen::Tensor<double, 2> &input_batch)
{
    return input_batch.unaryExpr([this](double x)
                                 { return x >= 0 ? x : alpha * (std::exp(x) - 1); });
}

Eigen::Tensor<double, 4> ActivationLayer::relu_derivative(const Eigen::Tensor<double, 4> &input_batch)
{
    return input_batch.unaryExpr([](double x)
                                 { return x > 0 ? 1.0 : 0.0; });
}

Eigen::Tensor<double, 2> ActivationLayer::relu_derivative(const Eigen::Tensor<double, 2> &input_batch)
{
    return input_batch.unaryExpr([](double x)
                                 { return x > 0 ? 1.0 : 0.0; });
}

Eigen::Tensor<double, 4> ActivationLayer::leakyRelu_derivative(const Eigen::Tensor<double, 4> &input_batch)
{
    return input_batch.unaryExpr([this](double x)
                                 { return x > 0 ? 1.0 : alpha; });
}

Eigen::Tensor<double, 2> ActivationLayer::leakyRelu_derivative(const Eigen::Tensor<double, 2> &input_batch)
{
    return input_batch.unaryExpr([this](double x)
                                 { return x > 0 ? 1.0 : alpha; });
}

Eigen::Tensor<double, 4> ActivationLayer::sigmoid_derivative(const Eigen::Tensor<double, 4> &input_batch)
{
    Eigen::Tensor<double, 4> sig = sigmoid(input_batch);
    return sig * (1.0 - sig);
}

Eigen::Tensor<double, 2> ActivationLayer::sigmoid_derivative(const Eigen::Tensor<double, 2> &input_batch)
{
    Eigen::Tensor<double, 2> sig = sigmoid(input_batch);
    return sig * (1.0 - sig);
}

Eigen::Tensor<double, 4> ActivationLayer::tanh_derivative(const Eigen::Tensor<double, 4> &input_batch)
{
    Eigen::Tensor<double, 4> t = tanh(input_batch);
    return 1.0 - t * t;
}

Eigen::Tensor<double, 2> ActivationLayer::tanh_derivative(const Eigen::Tensor<double, 2> &input_batch)
{
    Eigen::Tensor<double, 2> t = tanh(input_batch);
    return 1.0 - t * t;
}

Eigen::Tensor<double, 4> ActivationLayer::softmax_derivative(const Eigen::Tensor<double, 4> &input_batch)
{
    // The derivative of softmax is complex, usually not directly used in backpropagation.
    // Instead, the loss derivative with respect to input is computed directly in the loss function.
    return Eigen::Tensor<double, 4>(input_batch.dimensions());
}

Eigen::Tensor<double, 2> ActivationLayer::softmax_derivative(const Eigen::Tensor<double, 2> &input_batch)
{
    // The derivative of softmax is complex, usually not directly used in backpropagation.
    // Instead, the loss derivative with respect to input is computed directly in the loss function.
    return Eigen::Tensor<double, 2>(input_batch.dimensions());
}

Eigen::Tensor<double, 4> ActivationLayer::elu_derivative(const Eigen::Tensor<double, 4> &input_batch)
{
    return input_batch.unaryExpr([this](double x)
                                 { return x >= 0 ? 1.0 : alpha * std::exp(x); });
}

Eigen::Tensor<double, 2> ActivationLayer::elu_derivative(const Eigen::Tensor<double, 2> &input_batch)
{
    return input_batch.unaryExpr([this](double x)
                                 { return x >= 0 ? 1.0 : alpha * std::exp(x); });
}
