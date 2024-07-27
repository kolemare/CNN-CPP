#include "LossFunction.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <unsupported/Eigen/CXX11/Tensor>

// Factory method to create the appropriate loss function object
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

// Binary Cross Entropy Loss

double BinaryCrossEntropy::compute(const Eigen::Tensor<double, 4> &predictions, const Eigen::Tensor<int, 2> &targets) const
{
    int batch_size = predictions.dimension(0);
    double loss = 0.0;

    // Iterate over each prediction in the batch
    for (int i = 0; i < batch_size; ++i)
    {
        double pred = std::max(std::min(predictions(i, 0, 0, 0), 1.0 - 1e-7), 1e-7); // Clipping
        int target = targets(i, 0);

        // Calculate the loss using the binary cross-entropy formula
        loss += -target * std::log(pred) - (1 - target) * std::log(1 - pred);
    }

    return loss / batch_size;
}

Eigen::Tensor<double, 4> BinaryCrossEntropy::derivative(const Eigen::Tensor<double, 4> &predictions, const Eigen::Tensor<int, 2> &targets) const
{
    int batch_size = predictions.dimension(0);
    Eigen::Tensor<double, 4> d_output(batch_size, 1, 1, 1);

    // Iterate over each prediction in the batch
    for (int i = 0; i < batch_size; ++i)
    {
        double pred = std::max(std::min(predictions(i, 0, 0, 0), 1.0 - 1e-7), 1e-7); // Clipping
        int target = targets(i, 0);

        // Calculate the gradient of the loss function
        d_output(i, 0, 0, 0) = (pred - target) / (pred * (1 - pred));
    }

    return d_output;
}

// Mean Squared Error Loss

double MeanSquaredError::compute(const Eigen::Tensor<double, 4> &predictions, const Eigen::Tensor<int, 2> &targets) const
{
    int batch_size = predictions.dimension(0);
    double loss = 0.0;

    // Iterate over each prediction in the batch
    for (int i = 0; i < batch_size; ++i)
    {
        double pred = predictions(i, 0, 0, 0);
        int target = targets(i, 0);

        // Calculate the loss using the mean squared error formula
        loss += std::pow(pred - target, 2);
    }

    return loss / batch_size;
}

Eigen::Tensor<double, 4> MeanSquaredError::derivative(const Eigen::Tensor<double, 4> &predictions, const Eigen::Tensor<int, 2> &targets) const
{
    int batch_size = predictions.dimension(0);
    Eigen::Tensor<double, 4> d_output(batch_size, 1, 1, 1);

    // Iterate over each prediction in the batch
    for (int i = 0; i < batch_size; ++i)
    {
        double pred = predictions(i, 0, 0, 0);
        int target = targets(i, 0);

        // Calculate the gradient of the loss function
        d_output(i, 0, 0, 0) = 2 * (pred - target) / batch_size;
    }

    return d_output;
}

// Categorical Cross Entropy Loss

double CategoricalCrossEntropy::compute(const Eigen::Tensor<double, 4> &predictions, const Eigen::Tensor<int, 2> &targets) const
{
    int batch_size = predictions.dimension(0);
    int num_classes = predictions.dimension(3);
    double loss = 0.0;

    // Iterate over each prediction in the batch
    for (int i = 0; i < batch_size; ++i)
    {
        for (int j = 0; j < num_classes; ++j)
        {
            double pred = std::max(std::min(predictions(i, 0, 0, j), 1.0 - 1e-7), 1e-7); // Clipping
            int target = targets(i, j);

            // Calculate the loss using the categorical cross-entropy formula
            loss += -target * std::log(pred);
        }
    }

    return loss / batch_size;
}

Eigen::Tensor<double, 4> CategoricalCrossEntropy::derivative(const Eigen::Tensor<double, 4> &predictions, const Eigen::Tensor<int, 2> &targets) const
{
    int batch_size = predictions.dimension(0);
    int num_classes = predictions.dimension(3);
    Eigen::Tensor<double, 4> d_output(batch_size, 1, 1, num_classes);

    // Iterate over each prediction in the batch
    for (int i = 0; i < batch_size; ++i)
    {
        for (int j = 0; j < num_classes; ++j)
        {
            double pred = std::max(std::min(predictions(i, 0, 0, j), 1.0 - 1e-7), 1e-7); // Clipping
            int target = targets(i, j);

            // Calculate the gradient of the loss function
            d_output(i, 0, 0, j) = -target / pred;
        }
    }

    return d_output;
}
