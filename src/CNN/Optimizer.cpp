#include "Optimizer.hpp"
#include "TensorOperations.hpp"
#include <stdexcept>
#include <iostream>
#include <cmath>

// Factory method to create optimizers
std::shared_ptr<Optimizer> Optimizer::create(Type type, const std::unordered_map<std::string, double> &params)
{
    switch (type)
    {
    case Type::SGD:
        return std::make_shared<SGD>();
    case Type::SGDWithMomentum:
    {
        double momentum = params.at("momentum");
        return std::make_shared<SGDWithMomentum>(momentum);
    }
    case Type::Adam:
    {
        double beta1 = params.at("beta1");
        double beta2 = params.at("beta2");
        double epsilon = params.at("epsilon");
        return std::make_shared<Adam>(beta1, beta2, epsilon);
    }
    case Type::RMSprop:
    {
        double beta = params.at("beta");
        double epsilon = params.at("epsilon");
        return std::make_shared<RMSprop>(beta, epsilon);
    }
    default:
        throw std::invalid_argument("Unknown optimizer type");
    }
}

// SGD implementation
void SGD::update(Eigen::Tensor<double, 2> &weights, Eigen::Tensor<double, 1> &biases, const Eigen::Tensor<double, 2> &d_weights, const Eigen::Tensor<double, 1> &d_biases, double learning_rate)
{
    TensorOperations::applyUpdates(weights, d_weights, learning_rate);
    TensorOperations::applyUpdates(biases, d_biases, learning_rate);
}

void SGD::update(Eigen::Tensor<double, 4> &weights, Eigen::Tensor<double, 1> &biases, const Eigen::Tensor<double, 4> &d_weights, const Eigen::Tensor<double, 1> &d_biases, double learning_rate)
{
    TensorOperations::applyUpdates(weights, d_weights, learning_rate);
    TensorOperations::applyUpdates(biases, d_biases, learning_rate);
}

// SGD with Momentum implementation
SGDWithMomentum::SGDWithMomentum(double momentum)
    : momentum(momentum), v_weights_2d(Eigen::Tensor<double, 2>(1, 1)), v_biases_2d(Eigen::Tensor<double, 1>(1)), v_weights_4d(Eigen::Tensor<double, 4>(1, 1, 1, 1)), v_biases_4d(Eigen::Tensor<double, 1>(1)) {}

void SGDWithMomentum::update(Eigen::Tensor<double, 2> &weights, Eigen::Tensor<double, 1> &biases, const Eigen::Tensor<double, 2> &d_weights, const Eigen::Tensor<double, 1> &d_biases, double learning_rate)
{
    if (v_weights_2d.dimension(0) != weights.dimension(0) || v_weights_2d.dimension(1) != weights.dimension(1))
    {
        v_weights_2d = Eigen::Tensor<double, 2>(weights.dimension(0), weights.dimension(1));
        v_weights_2d.setZero();
    }
    if (v_biases_2d.dimension(0) != biases.dimension(0))
    {
        v_biases_2d = Eigen::Tensor<double, 1>(biases.dimension(0));
        v_biases_2d.setZero();
    }

    v_weights_2d = momentum * v_weights_2d + learning_rate * d_weights;
    v_biases_2d = momentum * v_biases_2d + learning_rate * d_biases;

    TensorOperations::applyUpdates(weights, v_weights_2d, 1.0);
    TensorOperations::applyUpdates(biases, v_biases_2d, 1.0);
}

void SGDWithMomentum::update(Eigen::Tensor<double, 4> &weights, Eigen::Tensor<double, 1> &biases, const Eigen::Tensor<double, 4> &d_weights, const Eigen::Tensor<double, 1> &d_biases, double learning_rate)
{
    if (v_weights_4d.dimension(0) != weights.dimension(0) || v_weights_4d.dimension(1) != weights.dimension(1) || v_weights_4d.dimension(2) != weights.dimension(2) || v_weights_4d.dimension(3) != weights.dimension(3))
    {
        v_weights_4d = Eigen::Tensor<double, 4>(weights.dimension(0), weights.dimension(1), weights.dimension(2), weights.dimension(3));
        v_weights_4d.setZero();
    }
    if (v_biases_4d.dimension(0) != biases.dimension(0))
    {
        v_biases_4d = Eigen::Tensor<double, 1>(biases.dimension(0));
        v_biases_4d.setZero();
    }

    v_weights_4d = momentum * v_weights_4d + learning_rate * d_weights;
    v_biases_4d = momentum * v_biases_4d + learning_rate * d_biases;

    TensorOperations::applyUpdates(weights, v_weights_4d, 1.0);
    TensorOperations::applyUpdates(biases, v_biases_4d, 1.0);
}

// Adam implementation
Adam::Adam(double beta1, double beta2, double epsilon)
    : beta1(beta1), beta2(beta2), epsilon(epsilon), t(0),
      m_weights_2d(Eigen::Tensor<double, 2>(1, 1)), v_weights_2d(Eigen::Tensor<double, 2>(1, 1)), m_biases_2d(Eigen::Tensor<double, 1>(1)), v_biases_2d(Eigen::Tensor<double, 1>(1)),
      m_weights_4d(Eigen::Tensor<double, 4>(1, 1, 1, 1)), v_weights_4d(Eigen::Tensor<double, 4>(1, 1, 1, 1)), m_biases_4d(Eigen::Tensor<double, 1>(1)), v_biases_4d(Eigen::Tensor<double, 1>(1)) {}

void Adam::update(Eigen::Tensor<double, 2> &weights, Eigen::Tensor<double, 1> &biases, const Eigen::Tensor<double, 2> &d_weights, const Eigen::Tensor<double, 1> &d_biases, double learning_rate)
{
    if (m_weights_2d.dimension(0) != weights.dimension(0) || m_weights_2d.dimension(1) != weights.dimension(1))
    {
        m_weights_2d = Eigen::Tensor<double, 2>(weights.dimension(0), weights.dimension(1));
        v_weights_2d = Eigen::Tensor<double, 2>(weights.dimension(0), weights.dimension(1));
        m_weights_2d.setZero();
        v_weights_2d.setZero();
    }
    if (m_biases_2d.dimension(0) != biases.dimension(0))
    {
        m_biases_2d = Eigen::Tensor<double, 1>(biases.dimension(0));
        v_biases_2d = Eigen::Tensor<double, 1>(biases.dimension(0));
        m_biases_2d.setZero();
        v_biases_2d.setZero();
    }

    t++;
    m_weights_2d = beta1 * m_weights_2d + (1.0 - beta1) * d_weights;
    v_weights_2d = beta2 * v_weights_2d + (1.0 - beta2) * d_weights.square();

    auto m_hat_weights = m_weights_2d / (1.0 - std::pow(beta1, t));
    auto v_hat_weights = v_weights_2d / (1.0 - std::pow(beta2, t));

    m_biases_2d = beta1 * m_biases_2d + (1.0 - beta1) * d_biases;
    v_biases_2d = beta2 * v_biases_2d + (1.0 - beta2) * d_biases.square();

    auto m_hat_biases = m_biases_2d / (1.0 - std::pow(beta1, t));
    auto v_hat_biases = v_biases_2d / (1.0 - std::pow(beta2, t));

    TensorOperations::applyUpdates(weights, m_hat_weights / (v_hat_weights.sqrt() + epsilon), learning_rate);
    TensorOperations::applyUpdates(biases, m_hat_biases / (v_hat_biases.sqrt() + epsilon), learning_rate);
}

void Adam::update(Eigen::Tensor<double, 4> &weights, Eigen::Tensor<double, 1> &biases, const Eigen::Tensor<double, 4> &d_weights, const Eigen::Tensor<double, 1> &d_biases, double learning_rate)
{
    if (m_weights_4d.dimension(0) != weights.dimension(0) || m_weights_4d.dimension(1) != weights.dimension(1) || m_weights_4d.dimension(2) != weights.dimension(2) || m_weights_4d.dimension(3) != weights.dimension(3))
    {
        m_weights_4d = Eigen::Tensor<double, 4>(weights.dimension(0), weights.dimension(1), weights.dimension(2), weights.dimension(3));
        v_weights_4d = Eigen::Tensor<double, 4>(weights.dimension(0), weights.dimension(1), weights.dimension(2), weights.dimension(3));
        m_weights_4d.setZero();
        v_weights_4d.setZero();
    }
    if (m_biases_4d.dimension(0) != biases.dimension(0))
    {
        m_biases_4d = Eigen::Tensor<double, 1>(biases.dimension(0));
        v_biases_4d = Eigen::Tensor<double, 1>(biases.dimension(0));
        m_biases_4d.setZero();
        v_biases_4d.setZero();
    }

    t++;
    m_weights_4d = beta1 * m_weights_4d + (1.0 - beta1) * d_weights;
    v_weights_4d = beta2 * v_weights_4d + (1.0 - beta2) * d_weights.square();

    auto m_hat_weights = m_weights_4d / (1.0 - std::pow(beta1, t));
    auto v_hat_weights = v_weights_4d / (1.0 - std::pow(beta2, t));

    m_biases_4d = beta1 * m_biases_4d + (1.0 - beta1) * d_biases;
    v_biases_4d = beta2 * v_biases_4d + (1.0 - beta2) * d_biases.square();

    auto m_hat_biases = m_biases_4d / (1.0 - std::pow(beta1, t));
    auto v_hat_biases = v_biases_4d / (1.0 - std::pow(beta2, t));

    TensorOperations::applyUpdates(weights, m_hat_weights / (v_hat_weights.sqrt() + epsilon), learning_rate);
    TensorOperations::applyUpdates(biases, m_hat_biases / (v_hat_biases.sqrt() + epsilon), learning_rate);
}

// RMSprop implementation
RMSprop::RMSprop(double beta, double epsilon)
    : beta(beta), epsilon(epsilon), s_weights_2d(Eigen::Tensor<double, 2>(1, 1)), s_biases_2d(Eigen::Tensor<double, 1>(1)),
      s_weights_4d(Eigen::Tensor<double, 4>(1, 1, 1, 1)), s_biases_4d(Eigen::Tensor<double, 1>(1)) {}

void RMSprop::update(Eigen::Tensor<double, 2> &weights, Eigen::Tensor<double, 1> &biases, const Eigen::Tensor<double, 2> &d_weights, const Eigen::Tensor<double, 1> &d_biases, double learning_rate)
{
    if (s_weights_2d.dimension(0) != weights.dimension(0) || s_weights_2d.dimension(1) != weights.dimension(1))
    {
        s_weights_2d = Eigen::Tensor<double, 2>(weights.dimension(0), weights.dimension(1));
        s_weights_2d.setZero();
    }
    if (s_biases_2d.dimension(0) != biases.dimension(0))
    {
        s_biases_2d = Eigen::Tensor<double, 1>(biases.dimension(0));
        s_biases_2d.setZero();
    }

    s_weights_2d = beta * s_weights_2d + (1.0 - beta) * d_weights.square();
    s_biases_2d = beta * s_biases_2d + (1.0 - beta) * d_biases.square();

    TensorOperations::applyUpdates(weights, d_weights / (s_weights_2d.sqrt() + epsilon), learning_rate);
    TensorOperations::applyUpdates(biases, d_biases / (s_biases_2d.sqrt() + epsilon), learning_rate);
}

void RMSprop::update(Eigen::Tensor<double, 4> &weights, Eigen::Tensor<double, 1> &biases, const Eigen::Tensor<double, 4> &d_weights, const Eigen::Tensor<double, 1> &d_biases, double learning_rate)
{
    if (s_weights_4d.dimension(0) != weights.dimension(0) || s_weights_4d.dimension(1) != weights.dimension(1) || s_weights_4d.dimension(2) != weights.dimension(2) || s_weights_4d.dimension(3) != weights.dimension(3))
    {
        s_weights_4d = Eigen::Tensor<double, 4>(weights.dimension(0), weights.dimension(1), weights.dimension(2), weights.dimension(3));
        s_weights_4d.setZero();
    }
    if (s_biases_4d.dimension(0) != biases.dimension(0))
    {
        s_biases_4d = Eigen::Tensor<double, 1>(biases.dimension(0));
        s_biases_4d.setZero();
    }

    s_weights_4d = beta * s_weights_4d + (1.0 - beta) * d_weights.square();
    s_biases_4d = beta * s_biases_4d + (1.0 - beta) * d_biases.square();

    TensorOperations::applyUpdates(weights, d_weights / (s_weights_4d.sqrt() + epsilon), learning_rate);
    TensorOperations::applyUpdates(biases, d_biases / (s_biases_4d.sqrt() + epsilon), learning_rate);
}
