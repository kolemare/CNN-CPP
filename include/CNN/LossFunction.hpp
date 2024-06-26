#ifndef LOSSFUNCTION_HPP
#define LOSSFUNCTION_HPP

#include <Eigen/Dense>
#include <memory>
#include <stdexcept>

enum class LossType
{
    BINARY_CROSS_ENTROPY,
    MEAN_SQUARED_ERROR,
    CATEGORICAL_CROSS_ENTROPY
};

class LossFunction
{
public:
    static std::unique_ptr<LossFunction> create(LossType type);

    virtual double compute(const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &targets) const = 0;
    virtual Eigen::MatrixXd derivative(const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &targets) const = 0;
    virtual ~LossFunction() = default;
};

class BinaryCrossEntropy : public LossFunction
{
public:
    double compute(const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &targets) const override;
    Eigen::MatrixXd derivative(const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &targets) const override;
};

class MeanSquaredError : public LossFunction
{
public:
    double compute(const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &targets) const override;
    Eigen::MatrixXd derivative(const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &targets) const override;
};

class CategoricalCrossEntropy : public LossFunction
{
public:
    double compute(const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &targets) const override;
    Eigen::MatrixXd derivative(const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &targets) const override;
};

#endif // LOSSFUNCTION_HPP
