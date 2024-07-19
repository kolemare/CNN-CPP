#ifndef ACTIVATIONLAYER_HPP
#define ACTIVATIONLAYER_HPP

#include <unsupported/Eigen/CXX11/Tensor>
#include "Layer.hpp"
#include "Optimizer.hpp"
#include <iostream>

enum ActivationType
{
    RELU,
    LEAKY_RELU,
    SIGMOID,
    TANH,
    SOFTMAX,
    ELU
};

class ActivationLayer : public Layer
{
public:
    ActivationLayer(ActivationType type);

    Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4> &input_batch) override;
    Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4> &d_output_batch, const Eigen::Tensor<double, 4> &input_batch, double learning_rate) override;
    bool needsOptimizer() const override;
    void setOptimizer(std::shared_ptr<Optimizer> optimizer) override;

    void setAlpha(double alphaValue);
    double getAlpha() const;

private:
    ActivationType type;
    double alpha;

    Eigen::Tensor<double, 4> wrap2DTensor(const Eigen::Tensor<double, 2> &input);
    Eigen::Tensor<double, 2> unwrap4DTensor(const Eigen::Tensor<double, 4> &input);

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
