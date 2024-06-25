#ifndef ACTIVATION_LAYER_HPP
#define ACTIVATION_LAYER_HPP

#include <Eigen/Dense>

class ActivationLayer
{
public:
    enum ActivationType
    {
        RELU,
        LEAKY_RELU,
        SIGMOID,
        TANH,
        SOFTMAX,
        ELU
    };

    ActivationLayer(ActivationType type);
    void setAlpha(double alphaValue);
    double getAlpha() const;
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input);

private:
    ActivationType type;
    double alpha;
    Eigen::MatrixXd relu(const Eigen::MatrixXd &input);
    Eigen::MatrixXd leakyRelu(const Eigen::MatrixXd &input);
    Eigen::MatrixXd sigmoid(const Eigen::MatrixXd &input);
    Eigen::MatrixXd tanh(const Eigen::MatrixXd &input);
    Eigen::MatrixXd softmax(const Eigen::MatrixXd &input);
    Eigen::MatrixXd elu(const Eigen::MatrixXd &input);
};

#endif // ACTIVATION_LAYER_HPP
