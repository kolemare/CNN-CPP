#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <unordered_map>
#include <string>

class Optimizer
{
public:
    enum class Type
    {
        SGD,
        SGDWithMomentum,
        Adam,
        RMSprop
    };

    enum class TensorType
    {
        TENSOR2D,
        TENSOR4D
    };

    virtual ~Optimizer() = default;

    virtual void update(Eigen::Tensor<double, 2> &weights, Eigen::Tensor<double, 1> &biases, const Eigen::Tensor<double, 2> &d_weights, const Eigen::Tensor<double, 1> &d_biases, double learning_rate) = 0;
    virtual void update(Eigen::Tensor<double, 4> &weights, Eigen::Tensor<double, 1> &biases, const Eigen::Tensor<double, 4> &d_weights, const Eigen::Tensor<double, 1> &d_biases, double learning_rate) = 0;

    static std::unique_ptr<Optimizer> create(Type type, const std::unordered_map<std::string, double> &params = {});
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

private:
    double beta;
    double epsilon;
    Eigen::Tensor<double, 2> s_weights_2d;
    Eigen::Tensor<double, 1> s_biases_2d;
    Eigen::Tensor<double, 4> s_weights_4d;
    Eigen::Tensor<double, 1> s_biases_4d;
};

#endif // OPTIMIZER_HPP
