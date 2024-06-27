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

void ActivationLayer::setAlpha(double alphaValue)
{
    alpha = alphaValue;
}

double ActivationLayer::getAlpha() const
{
    return alpha;
}

Eigen::MatrixXd ActivationLayer::forward(const Eigen::MatrixXd &input_batch)
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

Eigen::MatrixXd ActivationLayer::backward(const Eigen::MatrixXd &d_output_batch, const Eigen::MatrixXd &input_batch, double learning_rate)
{
    switch (type)
    {
    case RELU:
        return d_output_batch.cwiseProduct(relu_derivative(input_batch));
    case LEAKY_RELU:
        return d_output_batch.cwiseProduct(leakyRelu_derivative(input_batch));
    case SIGMOID:
        return d_output_batch.cwiseProduct(sigmoid_derivative(input_batch));
    case TANH:
        return d_output_batch.cwiseProduct(tanh_derivative(input_batch));
    case SOFTMAX:
        return d_output_batch.cwiseProduct(softmax_derivative(input_batch));
    case ELU:
        return d_output_batch.cwiseProduct(elu_derivative(input_batch));
    default:
        throw std::invalid_argument("Unsupported activation type");
    }
}

Eigen::MatrixXd ActivationLayer::relu(const Eigen::MatrixXd &input_batch)
{
    return input_batch.unaryExpr([](double x)
                                 { return std::max(0.0, x); });
}

Eigen::MatrixXd ActivationLayer::leakyRelu(const Eigen::MatrixXd &input_batch)
{
    return input_batch.unaryExpr([this](double x)
                                 { return x > 0 ? x : alpha * x; });
}

Eigen::MatrixXd ActivationLayer::sigmoid(const Eigen::MatrixXd &input_batch)
{
    std::cout << "Sigmoid input: " << input_batch << std::endl;
    Eigen::MatrixXd sig = input_batch.unaryExpr([](double x)
                                                { return 1.0 / (1.0 + std::exp(-x)); });
    std::cout << "Sigmoid output: " << sig << std::endl;
    return sig;
}

Eigen::MatrixXd ActivationLayer::tanh(const Eigen::MatrixXd &input_batch)
{
    return input_batch.unaryExpr([](double x)
                                 { return std::tanh(x); });
}

Eigen::MatrixXd ActivationLayer::softmax(const Eigen::MatrixXd &input_batch)
{
    Eigen::MatrixXd expInput = input_batch.unaryExpr([](double x)
                                                     { return std::exp(x); });
    Eigen::VectorXd sumExpInput = expInput.rowwise().sum();
    return expInput.array().colwise() / sumExpInput.array();
}

Eigen::MatrixXd ActivationLayer::elu(const Eigen::MatrixXd &input_batch)
{
    return input_batch.unaryExpr([this](double x)
                                 { return x >= 0 ? x : alpha * (std::exp(x) - 1); });
}

Eigen::MatrixXd ActivationLayer::relu_derivative(const Eigen::MatrixXd &input_batch)
{
    return input_batch.unaryExpr([](double x)
                                 { return x > 0 ? 1.0 : 0.0; });
}

Eigen::MatrixXd ActivationLayer::leakyRelu_derivative(const Eigen::MatrixXd &input_batch)
{
    return input_batch.unaryExpr([this](double x)
                                 { return x > 0 ? 1.0 : alpha; });
}

Eigen::MatrixXd ActivationLayer::sigmoid_derivative(const Eigen::MatrixXd &input_batch)
{
    Eigen::MatrixXd sig = sigmoid(input_batch);
    std::cout << "Sigmoid (for derivative): " << sig << std::endl;
    Eigen::MatrixXd derivative = sig.cwiseProduct(Eigen::MatrixXd::Ones(sig.rows(), sig.cols()) - sig);
    std::cout << "Sigmoid derivative: " << derivative << std::endl;
    return derivative;
}

Eigen::MatrixXd ActivationLayer::tanh_derivative(const Eigen::MatrixXd &input_batch)
{
    Eigen::MatrixXd t = tanh(input_batch);
    return Eigen::MatrixXd::Ones(t.rows(), t.cols()) - t.cwiseProduct(t);
}

Eigen::MatrixXd ActivationLayer::softmax_derivative(const Eigen::MatrixXd &input_batch)
{
    // The derivative of softmax is complex, usually not directly used in backpropagation.
    // Instead, the loss derivative with respect to input is computed directly in the loss function.
    return Eigen::MatrixXd::Ones(input_batch.rows(), input_batch.cols());
}

Eigen::MatrixXd ActivationLayer::elu_derivative(const Eigen::MatrixXd &input_batch)
{
    return input_batch.unaryExpr([this](double x)
                                 { return x >= 0 ? 1.0 : alpha * std::exp(x); });
}
