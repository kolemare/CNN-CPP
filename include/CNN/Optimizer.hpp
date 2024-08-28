#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "Common.hpp"

/**
 * @class Optimizer
 * @brief Abstract base class for optimizers.
 *
 * This class defines the interface for all optimizers used in training neural networks.
 */
class Optimizer
{
public:
    virtual ~Optimizer() = default;

    /**
     * @brief Update the weights and biases.
     *
     * @param weights The weights to update (4D tensor).
     * @param biases The biases to update (1D tensor).
     * @param d_weights The gradients of the weights (4D tensor).
     * @param d_biases The gradients of the biases (1D tensor).
     * @param learning_rate The learning rate for the update.
     */
    virtual void update(Eigen::Tensor<double, 4> &weights,
                        Eigen::Tensor<double, 1> &biases,
                        const Eigen::Tensor<double, 4> &d_weights,
                        const Eigen::Tensor<double, 1> &d_biases,
                        double learning_rate) = 0;

    /**
     * @brief Factory method to create an optimizer.
     *
     * @param type The type of optimizer to create.
     * @param params Parameters for the optimizer.
     * @return A shared pointer to the created optimizer.
     */
    static std::shared_ptr<Optimizer> create(OptimizerType type,
                                             const std::unordered_map<std::string, double> &params = {});
};

/**
 * @class SGD
 * @brief Stochastic Gradient Descent optimizer.
 */
class SGD : public Optimizer
{
public:
    /**
     * @brief Update the weights and biases using SGD.
     *
     * @param weights The weights to update (4D tensor).
     * @param biases The biases to update (1D tensor).
     * @param d_weights The gradients of the weights (4D tensor).
     * @param d_biases The gradients of the biases (1D tensor).
     * @param learning_rate The learning rate for the update.
     */
    void update(Eigen::Tensor<double, 4> &weights,
                Eigen::Tensor<double, 1> &biases,
                const Eigen::Tensor<double, 4> &d_weights,
                const Eigen::Tensor<double, 1> &d_biases,
                double learning_rate) override;
};

/**
 * @class SGDWithMomentum
 * @brief SGD optimizer with momentum.
 */
class SGDWithMomentum : public Optimizer
{
public:
    /**
     * @brief Construct a new SGDWithMomentum object.
     *
     * @param momentum The momentum factor.
     */
    SGDWithMomentum(double momentum);

    /**
     * @brief Update the weights and biases using SGD with momentum.
     *
     * @param weights The weights to update (4D tensor).
     * @param biases The biases to update (1D tensor).
     * @param d_weights The gradients of the weights (4D tensor).
     * @param d_biases The gradients of the biases (1D tensor).
     * @param learning_rate The learning rate for the update.
     */
    void update(Eigen::Tensor<double, 4> &weights,
                Eigen::Tensor<double, 1> &biases,
                const Eigen::Tensor<double, 4> &d_weights,
                const Eigen::Tensor<double, 1> &d_biases,
                double learning_rate) override;

    /**
     * @brief Get the velocity for weights (4D tensor).
     *
     * @return A copy of the velocity for weights (4D tensor).
     */
    Eigen::Tensor<double, 4> getVWeights() const;

    /**
     * @brief Get the velocity for biases (1D tensor).
     *
     * @return A copy of the velocity for biases (1D tensor).
     */
    Eigen::Tensor<double, 1> getVBiases() const;

    /**
     * @brief Set the velocity for weights (4D tensor).
     *
     * @param v_weights The new velocity for weights (4D tensor).
     */
    void setVWeights(const Eigen::Tensor<double, 4> &v_weights);

    /**
     * @brief Set the velocity for biases (1D tensor).
     *
     * @param v_biases The new velocity for biases (1D tensor).
     */
    void setVBiases(const Eigen::Tensor<double, 1> &v_biases);

    /**
     * @brief Get the momentum factor.
     *
     * @return The momentum factor.
     */
    static double getMomentum();

    /**
     * @brief Set the momentum factor.
     *
     * @param momentum The new momentum factor.
     */
    static void setMomentum(double momentum);

private:
    static double momentum;
    Eigen::Tensor<double, 4> v_weights;
    Eigen::Tensor<double, 1> v_biases;
};

/**
 * @class Adam
 * @brief Adam optimizer.
 *
 * Adam is an adaptive learning rate optimization algorithm designed for training deep neural networks.
 */
class Adam : public Optimizer
{
public:
    /**
     * @brief Construct a new Adam object.
     *
     * @param beta1 The exponential decay rate for the first moment estimates.
     * @param beta2 The exponential decay rate for the second moment estimates.
     * @param epsilon A small constant for numerical stability.
     */
    Adam(double beta1, double beta2, double epsilon);

    /**
     * @brief Update the weights and biases using the Adam optimizer.
     *
     * @param weights The weights to update (4D tensor).
     * @param biases The biases to update (1D tensor).
     * @param d_weights The gradients of the weights (4D tensor).
     * @param d_biases The gradients of the biases (1D tensor).
     * @param learning_rate The learning rate for the update.
     */
    void update(Eigen::Tensor<double, 4> &weights,
                Eigen::Tensor<double, 1> &biases,
                const Eigen::Tensor<double, 4> &d_weights,
                const Eigen::Tensor<double, 1> &d_biases,
                double learning_rate) override;

    /**
     * @brief Get the first moment estimates for weights (4D tensor).
     *
     * @return A copy of the first moment estimates for weights (4D tensor).
     */
    Eigen::Tensor<double, 4> getMWeights() const;

    /**
     * @brief Get the second moment estimates for weights (4D tensor).
     *
     * @return A copy of the second moment estimates for weights (4D tensor).
     */
    Eigen::Tensor<double, 4> getVWeights() const;

    /**
     * @brief Get the first moment estimates for biases (1D tensor).
     *
     * @return A copy of the first moment estimates for biases (1D tensor).
     */
    Eigen::Tensor<double, 1> getMBiases() const;

    /**
     * @brief Get the second moment estimates for biases (1D tensor).
     *
     * @return A copy of the second moment estimates for biases (1D tensor).
     */
    Eigen::Tensor<double, 1> getVBiases() const;

    /**
     * @brief Set the first moment estimates for weights (4D tensor).
     *
     * @param m_weights The new first moment estimates (4D tensor).
     */
    void setMWeights(const Eigen::Tensor<double, 4> &m_weights);

    /**
     * @brief Set the second moment estimates for weights (4D tensor).
     *
     * @param v_weights The new second moment estimates (4D tensor).
     */
    void setVWeights(const Eigen::Tensor<double, 4> &v_weights);

    /**
     * @brief Set the first moment estimates for biases (1D tensor).
     *
     * @param m_biases The new first moment estimates (1D tensor).
     */
    void setMBiases(const Eigen::Tensor<double, 1> &m_biases);

    /**
     * @brief Set the second moment estimates for biases (1D tensor).
     *
     * @param v_biases The new second moment estimates (1D tensor).
     */
    void setVBiases(const Eigen::Tensor<double, 1> &v_biases);

    /**
     * @brief Increments the current time step (t).
     *
     * @return void.
     */
    static void incrementT();

    /**
     * @brief Get the beta1 parameter.
     *
     * @return The beta1 parameter.
     */
    static double getBeta1();

    /**
     * @brief Get the beta2 parameter.
     *
     * @return The beta2 parameter.
     */
    static double getBeta2();

    /**
     * @brief Get the epsilon parameter.
     *
     * @return The epsilon parameter.
     */
    static double getEpsilon();

    /**
     * @brief Set the epsilon parameter.
     *
     * @param epsilon The new epsilon parameter.
     */
    static void setEpsilon(double epsilon);

    /**
     * @brief Get the current time step (t).
     *
     * @return The current time step (t).
     */
    static int getT();

    /**
     * @brief Set the beta1 parameter.
     *
     * @param beta1 The new beta1 parameter.
     */
    static void setBeta1(double beta1);

    /**
     * @brief Set the beta2 parameter.
     *
     * @param beta2 The new beta2 parameter.
     */
    static void setBeta2(double beta2);

    /**
     * @brief Set the current time step (t).
     *
     * @param t The new time step (t).
     */
    static void setT(int t);

private:
    static double beta1;
    static double beta2;
    static double epsilon;
    static int t;
    Eigen::Tensor<double, 4> m_weights;
    Eigen::Tensor<double, 4> v_weights;
    Eigen::Tensor<double, 1> m_biases;
    Eigen::Tensor<double, 1> v_biases;
};

/**
 * @class RMSprop
 * @brief RMSprop optimizer.
 *
 * RMSprop is an adaptive learning rate optimization algorithm.
 */
class RMSprop : public Optimizer
{
public:
    /**
     * @brief Construct a new RMSprop object.
     *
     * @param beta The decay rate for the moving average.
     * @param epsilon A small constant for numerical stability.
     */
    RMSprop(double beta, double epsilon);

    /**
     * @brief Update the weights and biases using RMSprop.
     *
     * @param weights The weights to update (4D tensor).
     * @param biases The biases to update (1D tensor).
     * @param d_weights The gradients of the weights (4D tensor).
     * @param d_biases The gradients of the biases (1D tensor).
     * @param learning_rate The learning rate for the update.
     */
    void update(Eigen::Tensor<double, 4> &weights,
                Eigen::Tensor<double, 1> &biases,
                const Eigen::Tensor<double, 4> &d_weights,
                const Eigen::Tensor<double, 1> &d_biases,
                double learning_rate) override;

    /**
     * @brief Get the squared gradient averages for weights (4D tensor).
     *
     * @return A copy of the squared gradient averages for weights (4D tensor).
     */
    Eigen::Tensor<double, 4> getSWeights() const;

    /**
     * @brief Get the squared gradient averages for biases (1D tensor).
     *
     * @return A copy of the squared gradient averages for biases (1D tensor).
     */
    Eigen::Tensor<double, 1> getSBiases() const;

    /**
     * @brief Set the squared gradient averages for weights (4D tensor).
     *
     * @param s_weights The new squared gradient averages (4D tensor).
     */
    void setSWeights(const Eigen::Tensor<double, 4> &s_weights);

    /**
     * @brief Set the squared gradient averages for biases (1D tensor).
     *
     * @param s_biases The new squared gradient averages (1D tensor).
     */
    void setSBiases(const Eigen::Tensor<double, 1> &s_biases);

    /**
     * @brief Get the beta parameter.
     *
     * @return The beta parameter.
     */
    static double getBeta();

    /**
     * @brief Set the beta parameter.
     *
     * @param beta The new beta parameter.
     */
    static void setBeta(double beta);

    /**
     * @brief Get the epsilon parameter.
     *
     * @return The epsilon parameter.
     */
    static double getEpsilon();

    /**
     * @brief Set the epsilon parameter.
     *
     * @param epsilon The new epsilon parameter.
     */
    static void setEpsilon(double epsilon);

private:
    static double beta;
    static double epsilon;
    Eigen::Tensor<double, 4> s_weights;
    Eigen::Tensor<double, 1> s_biases;
};

#endif // OPTIMIZER_HPP
