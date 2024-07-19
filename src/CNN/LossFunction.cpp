#include "LossFunction.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <unsupported/Eigen/CXX11/Tensor>

std::unique_ptr<LossFunction> LossFunction::create(LossType type)
{
    switch (type)
    {
    case LossType::BINARY_CROSS_ENTROPY:
        return std::make_unique<BinaryCrossEntropy>();
    case LossType::MEAN_SQUARED_ERROR:
        return std::make_unique<MeanSquaredError>();
    case LossType::CATEGORICAL_CROSS_ENTROPY:
        return std::make_unique<CategoricalCrossEntropy>();
    default:
        throw std::invalid_argument("Unknown loss type");
    }
}

// Binary Cross Entropy
double BinaryCrossEntropy::compute(const Eigen::Tensor<double, 4> &predictions, const Eigen::Tensor<int, 2> &targets) const
{
    Eigen::Tensor<double, 4> log_preds = predictions.log();
    Eigen::Tensor<double, 4> log_one_minus_preds = (1.0 - predictions).log();
    Eigen::Tensor<double, 0> loss = -(targets.cast<double>() * log_preds + (1.0 - targets.cast<double>()) * log_one_minus_preds).mean();
    return loss();
}

Eigen::Tensor<double, 4> BinaryCrossEntropy::derivative(const Eigen::Tensor<double, 4> &predictions, const Eigen::Tensor<int, 2> &targets) const
{
    return (predictions - targets.cast<double>()) / (predictions * (1.0 - predictions));
}

// Mean Squared Error
double MeanSquaredError::compute(const Eigen::Tensor<double, 4> &predictions, const Eigen::Tensor<int, 2> &targets) const
{
    Eigen::Tensor<double, 0> loss = (predictions - targets.cast<double>()).square().mean();
    return loss();
}

Eigen::Tensor<double, 4> MeanSquaredError::derivative(const Eigen::Tensor<double, 4> &predictions, const Eigen::Tensor<int, 2> &targets) const
{
    return 2 * (predictions - targets.cast<double>()) / predictions.dimension(0);
}

// Categorical Cross Entropy
double CategoricalCrossEntropy::compute(const Eigen::Tensor<double, 4> &predictions, const Eigen::Tensor<int, 2> &targets) const
{
    Eigen::Tensor<double, 4> clipped_preds = predictions.cwiseMax(1e-10).cwiseMin(1.0 - 1e-10);
    Eigen::Tensor<double, 0> loss = -(targets.cast<double>() * clipped_preds.log()).mean();
    return loss();
}

Eigen::Tensor<double, 4> CategoricalCrossEntropy::derivative(const Eigen::Tensor<double, 4> &predictions, const Eigen::Tensor<int, 2> &targets) const
{
    Eigen::Tensor<double, 4> clipped_preds = predictions.cwiseMax(1e-10).cwiseMin(1.0 - 1e-10);
    return -(targets.cast<double>() / clipped_preds);
}
