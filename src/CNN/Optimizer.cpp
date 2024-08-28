#include "Optimizer.hpp"
#include "TensorOperations.hpp"

// Initialize static members for Adam
double Adam::beta1 = 0.9;
double Adam::beta2 = 0.999;
double Adam::epsilon = 1e-7;
int Adam::t = 0;

// Initialize static members for SGDWithMomentum
double SGDWithMomentum::momentum = 0.9;

// Initialize static members for RMSprop
double RMSprop::beta = 0.9;
double RMSprop::epsilon = 1e-7;

// Factory method to create optimizers
std::shared_ptr<Optimizer> Optimizer::create(OptimizerType type,
                                             const std::unordered_map<std::string, double> &params)
{
    switch (type)
    {
    case OptimizerType::SGD:
        return std::make_shared<SGD>();
    case OptimizerType::SGDWithMomentum:
    {
        double momentum = params.at("momentum");
        SGDWithMomentum::setMomentum(momentum);
        return std::make_shared<SGDWithMomentum>(momentum);
    }
    case OptimizerType::Adam:
    {
        double beta1 = params.at("beta1");
        double beta2 = params.at("beta2");
        double epsilon = params.at("epsilon");
        Adam::setBeta1(beta1);
        Adam::setBeta2(beta2);
        Adam::setEpsilon(epsilon);
        return std::make_shared<Adam>(beta1, beta2, epsilon);
    }
    case OptimizerType::RMSprop:
    {
        double beta = params.at("beta");
        double epsilon = params.at("epsilon");
        RMSprop::setBeta(beta);
        RMSprop::setEpsilon(epsilon);
        return std::make_shared<RMSprop>(beta, epsilon);
    }
    default:
        throw std::invalid_argument("Unknown optimizer type");
    }
}

// SGD implementation
void SGD::update(Eigen::Tensor<double, 4> &weights,
                 Eigen::Tensor<double, 1> &biases,
                 const Eigen::Tensor<double, 4> &d_weights,
                 const Eigen::Tensor<double, 1> &d_biases,
                 double learning_rate)
{
    TensorOperations::applyUpdates(weights, d_weights, learning_rate);
    TensorOperations::applyUpdates(biases, d_biases, learning_rate);
}

// SGDWithMomentum implementation
SGDWithMomentum::SGDWithMomentum(double momentum)
{
    v_weights = Eigen::Tensor<double, 4>(1, 1, 1, 1);
    v_biases = Eigen::Tensor<double, 1>(1);
}

void SGDWithMomentum::update(Eigen::Tensor<double, 4> &weights,
                             Eigen::Tensor<double, 1> &biases,
                             const Eigen::Tensor<double, 4> &d_weights,
                             const Eigen::Tensor<double, 1> &d_biases,
                             double learning_rate)
{
    if (v_weights.dimension(0) != weights.dimension(0) || v_weights.dimension(1) != weights.dimension(1) ||
        v_weights.dimension(2) != weights.dimension(2) || v_weights.dimension(3) != weights.dimension(3))
    {
        v_weights = Eigen::Tensor<double, 4>(weights.dimension(0), weights.dimension(1), weights.dimension(2), weights.dimension(3));
        v_weights.setZero();
    }
    if (v_biases.dimension(0) != biases.dimension(0))
    {
        v_biases = Eigen::Tensor<double, 1>(biases.dimension(0));
        v_biases.setZero();
    }

    v_weights = momentum * v_weights + learning_rate * d_weights;
    v_biases = momentum * v_biases + learning_rate * d_biases;

    TensorOperations::applyUpdates(weights, v_weights, 1.0);
    TensorOperations::applyUpdates(biases, v_biases, 1.0);
}

Eigen::Tensor<double, 4> SGDWithMomentum::getVWeights() const
{
    return Eigen::Tensor<double, 4>(v_weights); // Return a copy
}

Eigen::Tensor<double, 1> SGDWithMomentum::getVBiases() const
{
    return Eigen::Tensor<double, 1>(v_biases); // Return a copy
}

void SGDWithMomentum::setVWeights(const Eigen::Tensor<double, 4> &v_weights)
{
    this->v_weights = Eigen::Tensor<double, 4>(v_weights); // Copy the input tensor
}

void SGDWithMomentum::setVBiases(const Eigen::Tensor<double, 1> &v_biases)
{
    this->v_biases = Eigen::Tensor<double, 1>(v_biases); // Copy the input tensor
}

double SGDWithMomentum::getMomentum()
{
    return momentum;
}

void SGDWithMomentum::setMomentum(double momentum)
{
    SGDWithMomentum::momentum = momentum;
}

// Adam implementation
Adam::Adam(double beta1,
           double beta2,
           double epsilon)
{
    m_weights = Eigen::Tensor<double, 4>(1, 1, 1, 1);
    v_weights = Eigen::Tensor<double, 4>(1, 1, 1, 1);
    m_biases = Eigen::Tensor<double, 1>(1);
    v_biases = Eigen::Tensor<double, 1>(1);
}

void Adam::update(Eigen::Tensor<double, 4> &weights,
                  Eigen::Tensor<double, 1> &biases,
                  const Eigen::Tensor<double, 4> &d_weights,
                  const Eigen::Tensor<double, 1> &d_biases,
                  double learning_rate)
{
    if (m_weights.dimension(0) != weights.dimension(0) || m_weights.dimension(1) != weights.dimension(1) ||
        m_weights.dimension(2) != weights.dimension(2) || m_weights.dimension(3) != weights.dimension(3))
    {
        m_weights = Eigen::Tensor<double, 4>(weights.dimension(0), weights.dimension(1), weights.dimension(2), weights.dimension(3));
        v_weights = Eigen::Tensor<double, 4>(weights.dimension(0), weights.dimension(1), weights.dimension(2), weights.dimension(3));
        m_weights.setZero();
        v_weights.setZero();
    }
    if (m_biases.dimension(0) != biases.dimension(0))
    {
        m_biases = Eigen::Tensor<double, 1>(biases.dimension(0));
        v_biases = Eigen::Tensor<double, 1>(biases.dimension(0));
        m_biases.setZero();
        v_biases.setZero();
    }

    m_weights = beta1 * m_weights + (1.0 - beta1) * d_weights;
    v_weights = beta2 * v_weights + (1.0 - beta2) * d_weights.square();

    auto m_hat_weights = m_weights / (1.0 - std::pow(beta1, t));
    auto v_hat_weights = v_weights / (1.0 - std::pow(beta2, t));

    m_biases = beta1 * m_biases + (1.0 - beta1) * d_biases;
    v_biases = beta2 * v_biases + (1.0 - beta2) * d_biases.square();

    auto m_hat_biases = m_biases / (1.0 - std::pow(beta1, t));
    auto v_hat_biases = v_biases / (1.0 - std::pow(beta2, t));

    TensorOperations::applyUpdates(weights, m_hat_weights / (v_hat_weights.sqrt() + epsilon), learning_rate);
    TensorOperations::applyUpdates(biases, m_hat_biases / (v_hat_biases.sqrt() + epsilon), learning_rate);
}

Eigen::Tensor<double, 4> Adam::getMWeights() const
{
    return Eigen::Tensor<double, 4>(m_weights); // Return a copy
}

Eigen::Tensor<double, 4> Adam::getVWeights() const
{
    return Eigen::Tensor<double, 4>(v_weights); // Return a copy
}

Eigen::Tensor<double, 1> Adam::getMBiases() const
{
    return Eigen::Tensor<double, 1>(m_biases); // Return a copy
}

Eigen::Tensor<double, 1> Adam::getVBiases() const
{
    return Eigen::Tensor<double, 1>(v_biases); // Return a copy
}

void Adam::setMWeights(const Eigen::Tensor<double, 4> &m_weights)
{
    this->m_weights = Eigen::Tensor<double, 4>(m_weights); // Copy the input tensor
}

void Adam::setVWeights(const Eigen::Tensor<double, 4> &v_weights)
{
    this->v_weights = Eigen::Tensor<double, 4>(v_weights); // Copy the input tensor
}

void Adam::setMBiases(const Eigen::Tensor<double, 1> &m_biases)
{
    this->m_biases = Eigen::Tensor<double, 1>(m_biases); // Copy the input tensor
}

void Adam::setVBiases(const Eigen::Tensor<double, 1> &v_biases)
{
    this->v_biases = Eigen::Tensor<double, 1>(v_biases); // Copy the input tensor
}

void Adam::incrementT()
{
    Adam::t++;
}

double Adam::getBeta1()
{
    return beta1;
}

double Adam::getBeta2()
{
    return beta2;
}

double Adam::getEpsilon()
{
    return epsilon;
}

void Adam::setEpsilon(double epsilon)
{
    Adam::epsilon = epsilon;
}

int Adam::getT()
{
    return t;
}

void Adam::setBeta1(double beta1)
{
    Adam::beta1 = beta1;
}

void Adam::setBeta2(double beta2)
{
    Adam::beta2 = beta2;
}

void Adam::setT(int t)
{
    Adam::t = t;
}

// RMSprop implementation
RMSprop::RMSprop(double beta,
                 double epsilon)
{
    s_weights = Eigen::Tensor<double, 4>(1, 1, 1, 1);
    s_biases = Eigen::Tensor<double, 1>(1);
}

void RMSprop::update(Eigen::Tensor<double, 4> &weights,
                     Eigen::Tensor<double, 1> &biases,
                     const Eigen::Tensor<double, 4> &d_weights,
                     const Eigen::Tensor<double, 1> &d_biases,
                     double learning_rate)
{
    if (s_weights.dimension(0) != weights.dimension(0) || s_weights.dimension(1) != weights.dimension(1) ||
        s_weights.dimension(2) != weights.dimension(2) || s_weights.dimension(3) != weights.dimension(3))
    {
        s_weights = Eigen::Tensor<double, 4>(weights.dimension(0), weights.dimension(1), weights.dimension(2), weights.dimension(3));
        s_weights.setZero();
    }
    if (s_biases.dimension(0) != biases.dimension(0))
    {
        s_biases = Eigen::Tensor<double, 1>(biases.dimension(0));
        s_biases.setZero();
    }

    s_weights = beta * s_weights + (1.0 - beta) * d_weights.square();
    s_biases = beta * s_biases + (1.0 - beta) * d_biases.square();

    TensorOperations::applyUpdates(weights, d_weights / (s_weights.sqrt() + epsilon), learning_rate);
    TensorOperations::applyUpdates(biases, d_biases / (s_biases.sqrt() + epsilon), learning_rate);
}

Eigen::Tensor<double, 4> RMSprop::getSWeights() const
{
    return Eigen::Tensor<double, 4>(s_weights); // Return a copy
}

Eigen::Tensor<double, 1> RMSprop::getSBiases() const
{
    return Eigen::Tensor<double, 1>(s_biases); // Return a copy
}

void RMSprop::setSWeights(const Eigen::Tensor<double, 4> &s_weights)
{
    this->s_weights = Eigen::Tensor<double, 4>(s_weights); // Copy the input tensor
}

void RMSprop::setSBiases(const Eigen::Tensor<double, 1> &s_biases)
{
    this->s_biases = Eigen::Tensor<double, 1>(s_biases); // Copy the input tensor
}

double RMSprop::getBeta()
{
    return beta;
}

double RMSprop::getEpsilon()
{
    return epsilon;
}

void RMSprop::setBeta(double beta)
{
    RMSprop::beta = beta;
}

void RMSprop::setEpsilon(double epsilon)
{
    RMSprop::epsilon = epsilon;
}
