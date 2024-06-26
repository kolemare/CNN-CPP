#include "LossFunction.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

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
double BinaryCrossEntropy::compute(const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &targets) const
{
    Eigen::MatrixXd log_preds = (predictions.array().log());
    Eigen::MatrixXd log_one_minus_preds = (1 - predictions.array()).log();
    return -(targets.array() * log_preds.array() + (1 - targets.array()) * log_one_minus_preds.array()).mean();
}

Eigen::MatrixXd BinaryCrossEntropy::derivative(const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &targets) const
{
    return (predictions - targets).array() / (predictions.array() * (1 - predictions.array()));
}

// Mean Squared Error
double MeanSquaredError::compute(const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &targets) const
{
    return (predictions - targets).array().square().mean();
}

Eigen::MatrixXd MeanSquaredError::derivative(const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &targets) const
{
    return 2 * (predictions - targets) / predictions.rows();
}

// Categorical Cross Entropy
double CategoricalCrossEntropy::compute(const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &targets) const
{
    Eigen::MatrixXd clipped_preds = predictions.array().max(1e-10).min(1.0 - 1e-10);
    return -(targets.array() * clipped_preds.array().log()).mean();
}

Eigen::MatrixXd CategoricalCrossEntropy::derivative(const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &targets) const
{
    Eigen::MatrixXd clipped_preds = predictions.array().max(1e-10).min(1.0 - 1e-10);
    return -(targets.array() / clipped_preds.array());
}
