#ifndef LOSSFUNCTION_HPP
#define LOSSFUNCTION_HPP

#include <unsupported/Eigen/CXX11/Tensor>
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

    virtual double compute(const Eigen::Tensor<double, 4> &predictions, const Eigen::Tensor<int, 2> &targets) const = 0;
    virtual Eigen::Tensor<double, 4> derivative(const Eigen::Tensor<double, 4> &predictions, const Eigen::Tensor<int, 2> &targets) const = 0;
    virtual ~LossFunction() = default;
};

class BinaryCrossEntropy : public LossFunction
{
public:
    double compute(const Eigen::Tensor<double, 4> &predictions, const Eigen::Tensor<int, 2> &targets) const override;
    Eigen::Tensor<double, 4> derivative(const Eigen::Tensor<double, 4> &predictions, const Eigen::Tensor<int, 2> &targets) const override;
};

class MeanSquaredError : public LossFunction
{
public:
    double compute(const Eigen::Tensor<double, 4> &predictions, const Eigen::Tensor<int, 2> &targets) const override;
    Eigen::Tensor<double, 4> derivative(const Eigen::Tensor<double, 4> &predictions, const Eigen::Tensor<int, 2> &targets) const override;
};

class CategoricalCrossEntropy : public LossFunction
{
public:
    double compute(const Eigen::Tensor<double, 4> &predictions, const Eigen::Tensor<int, 2> &targets) const override;
    Eigen::Tensor<double, 4> derivative(const Eigen::Tensor<double, 4> &predictions, const Eigen::Tensor<int, 2> &targets) const override;
};

#endif // LOSSFUNCTION_HPP
