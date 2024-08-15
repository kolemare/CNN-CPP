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
     * @param weights The weights to update (2D tensor).
     * @param biases The biases to update (1D tensor).
     * @param d_weights The gradients of the weights (2D tensor).
     * @param d_biases The gradients of the biases (1D tensor).
     * @param learning_rate The learning rate for the update.
     */
    virtual void update(Eigen::Tensor<double, 2> &weights,
                        Eigen::Tensor<double, 1> &biases,
                        const Eigen::Tensor<double, 2> &d_weights,
                        const Eigen::Tensor<double, 1> &d_biases,
                        double learning_rate) = 0;

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
     * @param weights The weights to update (2D tensor).
     * @param biases The biases to update (1D tensor).
     * @param d_weights The gradients of the weights (2D tensor).
     * @param d_biases The gradients of the biases (1D tensor).
     * @param learning_rate The learning rate for the update.
     */
    void update(Eigen::Tensor<double, 2> &weights,
                Eigen::Tensor<double, 1> &biases,
                const Eigen::Tensor<double, 2> &d_weights,
                const Eigen::Tensor<double, 1> &d_biases,
                double learning_rate) override;

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
     * @param weights The weights to update (2D tensor).
     * @param biases The biases to update (1D tensor).
     * @param d_weights The gradients of the weights (2D tensor).
     * @param d_biases The gradients of the biases (1D tensor).
     * @param learning_rate The learning rate for the update.
     */
    void update(Eigen::Tensor<double, 2> &weights,
                Eigen::Tensor<double, 1> &biases,
                const Eigen::Tensor<double, 2> &d_weights,
                const Eigen::Tensor<double, 1> &d_biases,
                double learning_rate) override;

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
     * @brief Get the velocity for weights (2D tensor).
     *
     * @return The velocity for weights (2D tensor).
     */
    Eigen::Tensor<double, 2> getVWeights2D() const;

    /**
     * @brief Get the velocity for biases (1D tensor).
     *
     * @return The velocity for biases (1D tensor).
     */
    Eigen::Tensor<double, 1> getVBiases2D() const;

    /**
     * @brief Get the velocity for weights (4D tensor).
     *
     * @return The velocity for weights (4D tensor).
     */
    Eigen::Tensor<double, 4> getVWeights4D() const;

    /**
     * @brief Get the velocity for biases (1D tensor).
     *
     * @return The velocity for biases (1D tensor).
     */
    Eigen::Tensor<double, 1> getVBiases4D() const;

    /**
     * @brief Set the velocity for weights (2D tensor).
     *
     * @param v_weights The new velocity for weights (2D tensor).
     */
    void setVWeights2D(const Eigen::Tensor<double, 2> &v_weights);

    /**
     * @brief Set the velocity for biases (1D tensor).
     *
     * @param v_biases The new velocity for biases (1D tensor).
     */
    void setVBiases2D(const Eigen::Tensor<double, 1> &v_biases);

    /**
     * @brief Set the velocity for weights (4D tensor).
     *
     * @param v_weights The new velocity for weights (4D tensor).
     */
    void setVWeights4D(const Eigen::Tensor<double, 4> &v_weights);

    /**
     * @brief Set the velocity for biases (1D tensor).
     *
     * @param v_biases The new velocity for biases (1D tensor).
     */
    void setVBiases4D(const Eigen::Tensor<double, 1> &v_biases);

private:
    double momentum;
    Eigen::Tensor<double, 2> v_weights_2d;
    Eigen::Tensor<double, 1> v_biases_2d;
    Eigen::Tensor<double, 4> v_weights_4d;
    Eigen::Tensor<double, 1> v_biases_4d;
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
     * @param weights The weights to update (2D tensor).
     * @param biases The biases to update (1D tensor).
     * @param d_weights The gradients of the weights (2D tensor).
     * @param d_biases The gradients of the biases (1D tensor).
     * @param learning_rate The learning rate for the update.
     */
    void update(Eigen::Tensor<double, 2> &weights,
                Eigen::Tensor<double, 1> &biases,
                const Eigen::Tensor<double, 2> &d_weights,
                const Eigen::Tensor<double, 1> &d_biases,
                double learning_rate) override;

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
     * @brief Get the first moment estimates for weights (2D tensor).
     *
     * @return The first moment estimates for weights (2D tensor).
     */
    Eigen::Tensor<double, 2> getMWeights2D() const;

    /**
     * @brief Get the second moment estimates for weights (2D tensor).
     *
     * @return The second moment estimates for weights (2D tensor).
     */
    Eigen::Tensor<double, 2> getVWeights2D() const;

    /**
     * @brief Get the first moment estimates for biases (1D tensor).
     *
     * @return The first moment estimates for biases (1D tensor).
     */
    Eigen::Tensor<double, 1> getMBiases2D() const;

    /**
     * @brief Get the second moment estimates for biases (1D tensor).
     *
     * @return The second moment estimates for biases (1D tensor).
     */
    Eigen::Tensor<double, 1> getVBiases2D() const;

    /**
     * @brief Get the first moment estimates for weights (4D tensor).
     *
     * @return The first moment estimates for weights (4D tensor).
     */
    Eigen::Tensor<double, 4> getMWeights4D() const;

    /**
     * @brief Get the second moment estimates for weights (4D tensor).
     *
     * @return The second moment estimates for weights (4D tensor).
     */
    Eigen::Tensor<double, 4> getVWeights4D() const;

    /**
     * @brief Get the first moment estimates for biases (1D tensor).
     *
     * @return The first moment estimates for biases (1D tensor).
     */
    Eigen::Tensor<double, 1> getMBiases4D() const;

    /**
     * @brief Get the second moment estimates for biases (1D tensor).
     *
     * @return The second moment estimates for biases (1D tensor).
     */
    Eigen::Tensor<double, 1> getVBiases4D() const;

    /**
     * @brief Set the first moment estimates for weights (2D tensor).
     *
     * @param m_weights The new first moment estimates (2D tensor).
     */
    void setMWeights2D(const Eigen::Tensor<double, 2> &m_weights);

    /**
     * @brief Set the second moment estimates for weights (2D tensor).
     *
     * @param v_weights The new second moment estimates (2D tensor).
     */
    void setVWeights2D(const Eigen::Tensor<double, 2> &v_weights);

    /**
     * @brief Set the first moment estimates for biases (1D tensor).
     *
     * @param m_biases The new first moment estimates (1D tensor).
     */
    void setMBiases2D(const Eigen::Tensor<double, 1> &m_biases);

    /**
     * @brief Set the second moment estimates for biases (1D tensor).
     *
     * @param v_biases The new second moment estimates (1D tensor).
     */
    void setVBiases2D(const Eigen::Tensor<double, 1> &v_biases);

    /**
     * @brief Set the first moment estimates for weights (4D tensor).
     *
     * @param m_weights The new first moment estimates (4D tensor).
     */
    void setMWeights4D(const Eigen::Tensor<double, 4> &m_weights);

    /**
     * @brief Set the second moment estimates for weights (4D tensor).
     *
     * @param v_weights The new second moment estimates (4D tensor).
     */
    void setVWeights4D(const Eigen::Tensor<double, 4> &v_weights);

    /**
     * @brief Set the first moment estimates for biases (1D tensor).
     *
     * @param m_biases The new first moment estimates (1D tensor).
     */
    void setMBiases4D(const Eigen::Tensor<double, 1> &m_biases);

    /**
     * @brief Set the second moment estimates for biases (1D tensor).
     *
     * @param v_biases The new second moment estimates (1D tensor).
     */
    void setVBiases4D(const Eigen::Tensor<double, 1> &v_biases);

private:
    double beta1;
    double beta2;
    double epsilon;
    int t;
    Eigen::Tensor<double, 2> m_weights_2d;
    Eigen::Tensor<double, 2> v_weights_2d;
    Eigen::Tensor<double, 1> m_biases_2d;
    Eigen::Tensor<double, 1> v_biases_2d;
    Eigen::Tensor<double, 4> m_weights_4d;
    Eigen::Tensor<double, 4> v_weights_4d;
    Eigen::Tensor<double, 1> m_biases_4d;
    Eigen::Tensor<double, 1> v_biases_4d;
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
     * @param weights The weights to update (2D tensor).
     * @param biases The biases to update (1D tensor).
     * @param d_weights The gradients of the weights (2D tensor).
     * @param d_biases The gradients of the biases (1D tensor).
     * @param learning_rate The learning rate for the update.
     */
    void update(Eigen::Tensor<double, 2> &weights,
                Eigen::Tensor<double, 1> &biases,
                const Eigen::Tensor<double, 2> &d_weights,
                const Eigen::Tensor<double, 1> &d_biases,
                double learning_rate) override;

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
     * @brief Get the squared gradient averages for weights (2D tensor).
     *
     * @return The squared gradient averages for weights (2D tensor).
     */
    Eigen::Tensor<double, 2> getSWeights2D() const;

    /**
     * @brief Get the squared gradient averages for biases (1D tensor).
     *
     * @return The squared gradient averages for biases (1D tensor).
     */
    Eigen::Tensor<double, 1> getSBiases2D() const;

    /**
     * @brief Get the squared gradient averages for weights (4D tensor).
     *
     * @return The squared gradient averages for weights (4D tensor).
     */
    Eigen::Tensor<double, 4> getSWeights4D() const;

    /**
     * @brief Get the squared gradient averages for biases (1D tensor).
     *
     * @return The squared gradient averages for biases (1D tensor).
     */
    Eigen::Tensor<double, 1> getSBiases4D() const;

    /**
     * @brief Set the squared gradient averages for weights (2D tensor).
     *
     * @param s_weights The new squared gradient averages (2D tensor).
     */
    void setSWeights2D(const Eigen::Tensor<double, 2> &s_weights);

    /**
     * @brief Set the squared gradient averages for biases (1D tensor).
     *
     * @param s_biases The new squared gradient averages (1D tensor).
     */
    void setSBiases2D(const Eigen::Tensor<double, 1> &s_biases);

    /**
     * @brief Set the squared gradient averages for weights (4D tensor).
     *
     * @param s_weights The new squared gradient averages (4D tensor).
     */
    void setSWeights4D(const Eigen::Tensor<double, 4> &s_weights);

    /**
     * @brief Set the squared gradient averages for biases (1D tensor).
     *
     * @param s_biases The new squared gradient averages (1D tensor).
     */
    void setSBiases4D(const Eigen::Tensor<double, 1> &s_biases);

private:
    double beta;
    double epsilon;
    Eigen::Tensor<double, 2> s_weights_2d;
    Eigen::Tensor<double, 1> s_biases_2d;
    Eigen::Tensor<double, 4> s_weights_4d;
    Eigen::Tensor<double, 1> s_biases_4d;
};

#endif // OPTIMIZER_HPP
