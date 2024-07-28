#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <unordered_map>
#include <string>
#include "Common.hpp"

class Optimizer
{
public:
    virtual ~Optimizer() = default;

    virtual void update(Eigen::Tensor<double, 2> &weights, Eigen::Tensor<double, 1> &biases, const Eigen::Tensor<double, 2> &d_weights, const Eigen::Tensor<double, 1> &d_biases, double learning_rate) = 0;
    virtual void update(Eigen::Tensor<double, 4> &weights, Eigen::Tensor<double, 1> &biases, const Eigen::Tensor<double, 4> &d_weights, const Eigen::Tensor<double, 1> &d_biases, double learning_rate) = 0;

    static std::shared_ptr<Optimizer> create(OptimizerType type, const std::unordered_map<std::string, double> &params = {});
};

class SGD : public Optimizer
{
public:
    void update(Eigen::Tensor<double, 2> &weights, Eigen::Tensor<double, 1> &biases, const Eigen::Tensor<double, 2> &d_weights, const Eigen::Tensor<double, 1> &d_biases, double learning_rate) override;
    void update(Eigen::Tensor<double, 4> &weights, Eigen::Tensor<double, 1> &biases, const Eigen::Tensor<double, 4> &d_weights, const Eigen::Tensor<double, 1> &d_biases, double learning_rate) override;
};

class SGDWithMomentum : public Optimizer
{
public:
    SGDWithMomentum(double momentum);
    void update(Eigen::Tensor<double, 2> &weights, Eigen::Tensor<double, 1> &biases, const Eigen::Tensor<double, 2> &d_weights, const Eigen::Tensor<double, 1> &d_biases, double learning_rate) override;
    void update(Eigen::Tensor<double, 4> &weights, Eigen::Tensor<double, 1> &biases, const Eigen::Tensor<double, 4> &d_weights, const Eigen::Tensor<double, 1> &d_biases, double learning_rate) override;

    Eigen::Tensor<double, 2> getVWeights2D() const;
    Eigen::Tensor<double, 1> getVBiases2D() const;
    Eigen::Tensor<double, 4> getVWeights4D() const;
    Eigen::Tensor<double, 1> getVBiases4D() const;

    void setVWeights2D(const Eigen::Tensor<double, 2> &v_weights);
    void setVBiases2D(const Eigen::Tensor<double, 1> &v_biases);
    void setVWeights4D(const Eigen::Tensor<double, 4> &v_weights);
    void setVBiases4D(const Eigen::Tensor<double, 1> &v_biases);

private:
    double momentum;
    Eigen::Tensor<double, 2> v_weights_2d;
    Eigen::Tensor<double, 1> v_biases_2d;
    Eigen::Tensor<double, 4> v_weights_4d;
    Eigen::Tensor<double, 1> v_biases_4d;
};

class Adam : public Optimizer
{
public:
    Adam(double beta1, double beta2, double epsilon);
    void update(Eigen::Tensor<double, 2> &weights, Eigen::Tensor<double, 1> &biases, const Eigen::Tensor<double, 2> &d_weights, const Eigen::Tensor<double, 1> &d_biases, double learning_rate) override;
    void update(Eigen::Tensor<double, 4> &weights, Eigen::Tensor<double, 1> &biases, const Eigen::Tensor<double, 4> &d_weights, const Eigen::Tensor<double, 1> &d_biases, double learning_rate) override;

    Eigen::Tensor<double, 2> getMWeights2D() const;
    Eigen::Tensor<double, 2> getVWeights2D() const;
    Eigen::Tensor<double, 1> getMBiases2D() const;
    Eigen::Tensor<double, 1> getVBiases2D() const;
    Eigen::Tensor<double, 4> getMWeights4D() const;
    Eigen::Tensor<double, 4> getVWeights4D() const;
    Eigen::Tensor<double, 1> getMBiases4D() const;
    Eigen::Tensor<double, 1> getVBiases4D() const;

    void setMWeights2D(const Eigen::Tensor<double, 2> &m_weights);
    void setVWeights2D(const Eigen::Tensor<double, 2> &v_weights);
    void setMBiases2D(const Eigen::Tensor<double, 1> &m_biases);
    void setVBiases2D(const Eigen::Tensor<double, 1> &v_biases);
    void setMWeights4D(const Eigen::Tensor<double, 4> &m_weights);
    void setVWeights4D(const Eigen::Tensor<double, 4> &v_weights);
    void setMBiases4D(const Eigen::Tensor<double, 1> &m_biases);
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

class RMSprop : public Optimizer
{
public:
    RMSprop(double beta, double epsilon);
    void update(Eigen::Tensor<double, 2> &weights, Eigen::Tensor<double, 1> &biases, const Eigen::Tensor<double, 2> &d_weights, const Eigen::Tensor<double, 1> &d_biases, double learning_rate) override;
    void update(Eigen::Tensor<double, 4> &weights, Eigen::Tensor<double, 1> &biases, const Eigen::Tensor<double, 4> &d_weights, const Eigen::Tensor<double, 1> &d_biases, double learning_rate) override;

    Eigen::Tensor<double, 2> getSWeights2D() const;
    Eigen::Tensor<double, 1> getSBiases2D() const;
    Eigen::Tensor<double, 4> getSWeights4D() const;
    Eigen::Tensor<double, 1> getSBiases4D() const;

    void setSWeights2D(const Eigen::Tensor<double, 2> &s_weights);
    void setSBiases2D(const Eigen::Tensor<double, 1> &s_biases);
    void setSWeights4D(const Eigen::Tensor<double, 4> &s_weights);
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
