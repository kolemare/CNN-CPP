#include "ActivationLayer.hpp"
#include <cmath>

ActivationLayer::ActivationLayer(ActivationType type) : type(type) {}

Eigen::MatrixXd ActivationLayer::forward(const Eigen::MatrixXd &input)
{
    switch (type)
    {
    case RELU:
        return relu(input);
    case LEAKY_RELU:
        return leakyRelu(input);
    case SIGMOID:
        return sigmoid(input);
    case TANH:
        return tanh(input);
    case SOFTMAX:
        return softmax(input);
    case ELU:
        return elu(input);
    default:
        return input; // Default case should not occur
    }
}

Eigen::MatrixXd ActivationLayer::relu(const Eigen::MatrixXd &input)
{
    return input.unaryExpr([](double x)
                           { return std::max(0.0, x); });
}

Eigen::MatrixXd ActivationLayer::leakyRelu(const Eigen::MatrixXd &input, double alpha)
{
    return input.unaryExpr([alpha](double x)
                           { return x > 0 ? x : alpha * x; });
}

Eigen::MatrixXd ActivationLayer::sigmoid(const Eigen::MatrixXd &input)
{
    return input.unaryExpr([](double x)
                           { return 1.0 / (1.0 + std::exp(-x)); });
}

Eigen::MatrixXd ActivationLayer::tanh(const Eigen::MatrixXd &input)
{
    return input.unaryExpr([](double x)
                           { return std::tanh(x); });
}

Eigen::MatrixXd ActivationLayer::softmax(const Eigen::MatrixXd &input)
{
    Eigen::MatrixXd expInput = input.unaryExpr([](double x)
                                               { return std::exp(x); });
    Eigen::VectorXd sumExpInput = expInput.rowwise().sum();
    return expInput.array().colwise() / sumExpInput.array();
}

Eigen::MatrixXd ActivationLayer::elu(const Eigen::MatrixXd &input, double alpha)
{
    return input.unaryExpr([alpha](double x)
                           { return x >= 0 ? x : alpha * (std::exp(x) - 1); });
}
