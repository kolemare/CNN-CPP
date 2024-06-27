#ifndef ACTIVATIONLAYER_HPP
#define ACTIVATIONLAYER_HPP

#include <Eigen/Dense>
#include "Layer.hpp"
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

    Eigen::MatrixXd forward(const Eigen::MatrixXd &input_batch) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd &d_output_batch, const Eigen::MatrixXd &input_batch, double learning_rate) override;

    void setAlpha(double alphaValue);
    double getAlpha() const;

private:
    ActivationType type;
    double alpha;

    Eigen::MatrixXd relu(const Eigen::MatrixXd &input_batch);
    Eigen::MatrixXd leakyRelu(const Eigen::MatrixXd &input_batch);
    Eigen::MatrixXd sigmoid(const Eigen::MatrixXd &input_batch);
    Eigen::MatrixXd tanh(const Eigen::MatrixXd &input_batch);
    Eigen::MatrixXd softmax(const Eigen::MatrixXd &input_batch);
    Eigen::MatrixXd elu(const Eigen::MatrixXd &input_batch);

    Eigen::MatrixXd relu_derivative(const Eigen::MatrixXd &input_batch);
    Eigen::MatrixXd leakyRelu_derivative(const Eigen::MatrixXd &input_batch);
    Eigen::MatrixXd sigmoid_derivative(const Eigen::MatrixXd &input_batch);
    Eigen::MatrixXd tanh_derivative(const Eigen::MatrixXd &input_batch);
    Eigen::MatrixXd softmax_derivative(const Eigen::MatrixXd &input_batch);
    Eigen::MatrixXd elu_derivative(const Eigen::MatrixXd &input_batch);
};

#endif // ACTIVATIONLAYER_HPP
