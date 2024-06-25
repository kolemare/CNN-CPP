#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <Eigen/Dense>
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

    virtual ~Optimizer() = default;
    virtual void update(Eigen::MatrixXd &weights, Eigen::VectorXd &biases, const Eigen::MatrixXd &d_weights, const Eigen::VectorXd &d_biases, double learning_rate) = 0;

    static std::unique_ptr<Optimizer> create(Type type, const std::unordered_map<std::string, double> &params = {});
};

class SGD : public Optimizer
{
public:
    void update(Eigen::MatrixXd &weights, Eigen::VectorXd &biases, const Eigen::MatrixXd &d_weights, const Eigen::VectorXd &d_biases, double learning_rate) override;
};

class SGDWithMomentum : public Optimizer
{
public:
    SGDWithMomentum(double momentum);
    void update(Eigen::MatrixXd &weights, Eigen::VectorXd &biases, const Eigen::MatrixXd &d_weights, const Eigen::VectorXd &d_biases, double learning_rate) override;

private:
    double momentum;
    Eigen::MatrixXd v_weights;
    Eigen::VectorXd v_biases;
};

class Adam : public Optimizer
{
public:
    Adam(double beta1, double beta2, double epsilon);
    void update(Eigen::MatrixXd &weights, Eigen::VectorXd &biases, const Eigen::MatrixXd &d_weights, const Eigen::VectorXd &d_biases, double learning_rate) override;

private:
    double beta1;
    double beta2;
    double epsilon;
    Eigen::MatrixXd m_weights;
    Eigen::MatrixXd v_weights;
    Eigen::VectorXd m_biases;
    Eigen::VectorXd v_biases;
    int t;
};

class RMSprop : public Optimizer
{
public:
    RMSprop(double beta, double epsilon);
    void update(Eigen::MatrixXd &weights, Eigen::VectorXd &biases, const Eigen::MatrixXd &d_weights, const Eigen::VectorXd &d_biases, double learning_rate) override;

private:
    double beta;
    double epsilon;
    Eigen::MatrixXd s_weights;
    Eigen::VectorXd s_biases;
};

#endif // OPTIMIZER_HPP
