#include "Optimizer.hpp"
#include <stdexcept>
#include <iostream>
#include <cmath>

// Factory method to create optimizers
std::unique_ptr<Optimizer> Optimizer::create(Type type, const std::unordered_map<std::string, double> &params)
{
    switch (type)
    {
    case Type::SGD:
    {
        return std::make_unique<SGD>();
    }
    case Type::SGDWithMomentum:
    {
        double momentum = params.at("momentum");
        return std::make_unique<SGDWithMomentum>(momentum);
    }
    case Type::Adam:
    {
        double beta1 = params.at("beta1");
        double beta2 = params.at("beta2");
        double epsilon = params.at("epsilon");
        return std::make_unique<Adam>(beta1, beta2, epsilon);
    }
    case Type::RMSprop:
    {
        double beta = params.at("beta");
        double epsilon = params.at("epsilon");
        return std::make_unique<RMSprop>(beta, epsilon);
    }
    default:
        throw std::invalid_argument("Unknown optimizer type");
    }
}

// SGD implementation
void SGD::update(Eigen::MatrixXd &weights, Eigen::VectorXd &biases, const Eigen::MatrixXd &d_weights, const Eigen::VectorXd &d_biases, double learning_rate)
{
    weights -= learning_rate * d_weights;
    biases -= learning_rate * d_biases;
}

// SGD with Momentum implementation
SGDWithMomentum::SGDWithMomentum(double momentum)
    : momentum(momentum), v_weights(Eigen::MatrixXd::Zero(1, 1)), v_biases(Eigen::VectorXd::Zero(1)) {}

void SGDWithMomentum::update(Eigen::MatrixXd &weights, Eigen::VectorXd &biases, const Eigen::MatrixXd &d_weights, const Eigen::VectorXd &d_biases, double learning_rate)
{
    if (v_weights.rows() != weights.rows() || v_weights.cols() != weights.cols())
    {
        v_weights = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    }
    if (v_biases.size() != biases.size())
    {
        v_biases = Eigen::VectorXd::Zero(biases.size());
    }

    v_weights = momentum * v_weights + learning_rate * d_weights;
    v_biases = momentum * v_biases + learning_rate * d_biases;

    weights -= v_weights;
    biases -= v_biases;
}

// Adam implementation
Adam::Adam(double beta1, double beta2, double epsilon)
    : beta1(beta1), beta2(beta2), epsilon(epsilon), m_weights(Eigen::MatrixXd::Zero(1, 1)), v_weights(Eigen::MatrixXd::Zero(1, 1)), m_biases(Eigen::VectorXd::Zero(1)), v_biases(Eigen::VectorXd::Zero(1)), t(0) {}

void Adam::update(Eigen::MatrixXd &weights, Eigen::VectorXd &biases, const Eigen::MatrixXd &d_weights, const Eigen::VectorXd &d_biases, double learning_rate)
{
    if (m_weights.rows() != weights.rows() || m_weights.cols() != weights.cols())
    {
        m_weights = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
        v_weights = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    }
    if (m_biases.size() != biases.size())
    {
        m_biases = Eigen::VectorXd::Zero(biases.size());
        v_biases = Eigen::VectorXd::Zero(biases.size());
    }

    t++;
    m_weights = beta1 * m_weights + (1.0 - beta1) * d_weights;
    v_weights = beta2 * v_weights + (1.0 - beta2) * d_weights.cwiseProduct(d_weights);

    Eigen::MatrixXd m_hat_weights = m_weights / (1.0 - std::pow(beta1, t));
    Eigen::MatrixXd v_hat_weights = v_weights / (1.0 - std::pow(beta2, t));

    m_biases = beta1 * m_biases + (1.0 - beta1) * d_biases;
    v_biases = beta2 * v_biases + (1.0 - beta2) * d_biases.cwiseProduct(d_biases);

    Eigen::VectorXd m_hat_biases = m_biases / (1.0 - std::pow(beta1, t));
    Eigen::VectorXd v_hat_biases = v_biases / (1.0 - std::pow(beta2, t));

    weights -= learning_rate * m_hat_weights.cwiseQuotient((v_hat_weights.array().sqrt() + epsilon).matrix());
    biases -= learning_rate * m_hat_biases.cwiseQuotient((v_hat_biases.array().sqrt() + epsilon).matrix());
}

// RMSprop implementation
RMSprop::RMSprop(double beta, double epsilon)
    : beta(beta), epsilon(epsilon), s_weights(Eigen::MatrixXd::Zero(1, 1)), s_biases(Eigen::VectorXd::Zero(1)) {}

void RMSprop::update(Eigen::MatrixXd &weights, Eigen::VectorXd &biases, const Eigen::MatrixXd &d_weights, const Eigen::VectorXd &d_biases, double learning_rate)
{
    if (s_weights.rows() != weights.rows() || s_weights.cols() != weights.cols())
    {
        s_weights = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    }
    if (s_biases.size() != biases.size())
    {
        s_biases = Eigen::VectorXd::Zero(biases.size());
    }

    s_weights = beta * s_weights + (1.0 - beta) * d_weights.cwiseProduct(d_weights);
    s_biases = beta * s_biases + (1.0 - beta) * d_biases.cwiseProduct(d_biases);

    weights -= learning_rate * d_weights.cwiseQuotient((s_weights.array().sqrt() + epsilon).matrix());
    biases -= learning_rate * d_biases.cwiseQuotient((s_biases.array().sqrt() + epsilon).matrix());
}
