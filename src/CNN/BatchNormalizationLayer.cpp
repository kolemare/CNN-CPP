#include "BatchNormalizationLayer.hpp"
#include <iostream>
#include <cmath>

/**
 * @brief Constructs a BatchNormalizationLayer with the specified epsilon and momentum.
 *
 * @param epsilon A small constant to avoid division by zero during normalization.
 * @param momentum The momentum for the moving average of mean and variance.
 */
BatchNormalizationLayer::BatchNormalizationLayer(double epsilon, double momentum)
{
    this->epsilon = epsilon;
    this->momentum = momentum;
    this->initialized = false;
}

/**
 * @brief Returns false, indicating that batch normalization does not need an optimizer.
 *
 * @return False as batch normalization does not use an optimizer.
 */
bool BatchNormalizationLayer::needsOptimizer() const
{
    return false;
}

/**
 * @brief Sets the optimizer for batch normalization (not used).
 *
 * @param optimizer A shared pointer to an optimizer.
 */
void BatchNormalizationLayer::setOptimizer(std::shared_ptr<Optimizer> optimizer)
{
    return;
}

/**
 * @brief Returns a null pointer as batch normalization does not have an optimizer.
 *
 * @return A null pointer.
 */
std::shared_ptr<Optimizer> BatchNormalizationLayer::getOptimizer()
{
    return nullptr;
}

/**
 * @brief Initializes gamma, beta, moving mean, and moving variance.
 *
 * @param feature_size The size of the feature to initialize parameters.
 */
void BatchNormalizationLayer::initialize(int feature_size)
{
    gamma = Eigen::Tensor<double, 1>(feature_size);
    beta = Eigen::Tensor<double, 1>(feature_size);
    moving_mean = Eigen::Tensor<double, 1>(feature_size);
    moving_variance = Eigen::Tensor<double, 1>(feature_size);
    dgamma = Eigen::Tensor<double, 1>(feature_size);
    dbeta = Eigen::Tensor<double, 1>(feature_size);

    // Set initial values for parameters
    gamma.setConstant(1.0);           // Initialize scale to 1
    beta.setZero();                   // Initialize shift to 0
    moving_mean.setZero();            // Initialize moving mean to 0
    moving_variance.setConstant(1.0); // Initialize moving variance to 1

    initialized = true;
}

/**
 * @brief Forward pass: normalizes the input and scales it using gamma and beta.
 *
 * @param input_batch A 4D tensor containing the input data.
 * @return A 4D tensor containing the normalized output data.
 */
Eigen::Tensor<double, 4> BatchNormalizationLayer::forward(const Eigen::Tensor<double, 4> &input_batch)
{
    // Get dimensions of the input batch
    int batch_size = input_batch.dimension(0);
    int input_depth = input_batch.dimension(1);
    int height = input_batch.dimension(2);
    int width = input_batch.dimension(3);
    int feature_size = input_depth * height * width;

    // Initialize parameters if not already done
    if (!initialized)
    {
        initialize(feature_size);
    }

    // Compute mean and variance for the current batch
    Eigen::Tensor<double, 1> input_mean(feature_size);
    Eigen::Tensor<double, 1> input_variance(feature_size);
    input_mean.setZero();
    input_variance.setZero();

    // Compute mean
    for (int d = 0; d < input_depth; ++d)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                for (int n = 0; n < batch_size; ++n)
                {
                    input_mean(d * height * width + h * width + w) += input_batch(n, d, h, w);
                }
                input_mean(d * height * width + h * width + w) /= batch_size;
            }
        }
    }

    // Compute variance
    for (int d = 0; d < input_depth; ++d)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                for (int n = 0; n < batch_size; ++n)
                {
                    double diff = input_batch(n, d, h, w) - input_mean(d * height * width + h * width + w);
                    input_variance(d * height * width + h * width + w) += diff * diff;
                }
                input_variance(d * height * width + h * width + w) /= batch_size;
            }
        }
    }

    // Cache mean and variance for use in backward pass
    cache_mean = input_mean;
    cache_variance = input_variance;

    // Compute inverse square root of variance
    Eigen::Tensor<double, 1> inv_sqrt_variance(feature_size);
    for (int i = 0; i < feature_size; ++i)
    {
        inv_sqrt_variance(i) = 1.0 / std::sqrt(cache_variance(i) + epsilon);
    }

    // Normalize the input batch
    cache_normalized = input_batch;
    for (int n = 0; n < batch_size; ++n)
    {
        for (int d = 0; d < input_depth; ++d)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    int index = d * height * width + h * width + w;
                    cache_normalized(n, d, h, w) = (input_batch(n, d, h, w) - input_mean(index)) * inv_sqrt_variance(index);
                }
            }
        }
    }

    // Update moving averages of mean and variance
    for (int i = 0; i < feature_size; ++i)
    {
        moving_mean(i) = momentum * moving_mean(i) + (1.0 - momentum) * cache_mean(i);
        moving_variance(i) = momentum * moving_variance(i) + (1.0 - momentum) * cache_variance(i);
    }

    // Return the scaled and shifted output
    Eigen::Tensor<double, 4> output(batch_size, input_depth, height, width);
    for (int n = 0; n < batch_size; ++n)
    {
        for (int d = 0; d < input_depth; ++d)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    int index = d * height * width + h * width + w;
                    output(n, d, h, w) = cache_normalized(n, d, h, w) * gamma(index) + beta(index);
                }
            }
        }
    }

    return output;
}

/**
 * @brief Backward pass: computes the gradient of the loss with respect to the input.
 *
 * @param d_output_batch A 4D tensor containing the gradient of the loss with respect to the output.
 * @param input_batch A 4D tensor containing the input data.
 * @param learning_rate The learning rate for updating gamma and beta.
 * @return A 4D tensor containing the gradient of the loss with respect to the input.
 */
Eigen::Tensor<double, 4> BatchNormalizationLayer::backward(const Eigen::Tensor<double, 4> &d_output_batch,
                                                           const Eigen::Tensor<double, 4> &input_batch,
                                                           double learning_rate)
{
    // Get dimensions of the input batch
    int batch_size = input_batch.dimension(0);
    int input_depth = input_batch.dimension(1);
    int height = input_batch.dimension(2);
    int width = input_batch.dimension(3);
    int feature_size = input_depth * height * width;

    // Compute gradients for gamma and beta
    dgamma.setZero();
    dbeta.setZero();
    for (int n = 0; n < batch_size; ++n)
    {
        for (int d = 0; d < input_depth; ++d)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    int index = d * height * width + h * width + w;
                    dgamma(index) += d_output_batch(n, d, h, w) * cache_normalized(n, d, h, w);
                    dbeta(index) += d_output_batch(n, d, h, w);
                }
            }
        }
    }

    // Compute inverse square root of variance
    Eigen::Tensor<double, 1> inv_sqrt_variance(feature_size);
    for (int i = 0; i < feature_size; ++i)
    {
        inv_sqrt_variance(i) = 1.0 / std::sqrt(cache_variance(i) + epsilon);
    }

    // Compute gradient with respect to variance
    Eigen::Tensor<double, 1> d_variance(feature_size);
    d_variance.setZero();
    for (int n = 0; n < batch_size; ++n)
    {
        for (int d = 0; d < input_depth; ++d)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    int index = d * height * width + h * width + w;
                    double diff = input_batch(n, d, h, w) - cache_mean(index);
                    d_variance(index) += d_output_batch(n, d, h, w) * diff * inv_sqrt_variance(index);
                }
            }
        }
    }

    for (int i = 0; i < feature_size; ++i)
    {
        d_variance(i) *= -0.5 * std::pow(inv_sqrt_variance(i), 3);
    }

    // Compute gradient with respect to mean
    Eigen::Tensor<double, 1> d_mean(feature_size);
    d_mean.setZero();
    for (int n = 0; n < batch_size; ++n)
    {
        for (int d = 0; d < input_depth; ++d)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    int index = d * height * width + h * width + w;
                    d_mean(index) += d_output_batch(n, d, h, w) * inv_sqrt_variance(index);
                }
            }
        }
    }

    for (int i = 0; i < feature_size; ++i)
    {
        d_mean(i) += d_variance(i) * -2.0 * cache_mean(i) / batch_size;
    }

    // Compute final gradient with respect to input
    Eigen::Tensor<double, 4> d_input(batch_size, input_depth, height, width);
    for (int n = 0; n < batch_size; ++n)
    {
        for (int d = 0; d < input_depth; ++d)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    int index = d * height * width + h * width + w;
                    d_input(n, d, h, w) = d_output_batch(n, d, h, w) * inv_sqrt_variance(index) -
                                          d_mean(index) / batch_size -
                                          d_variance(index) * 2.0 * (input_batch(n, d, h, w) - cache_mean(index)) / batch_size;
                }
            }
        }
    }

    // Update gamma and beta parameters
    updateParameters(learning_rate);

    // Return the gradient with respect to the input batch
    return d_input;
}

/**
 * @brief Updates gamma and beta using the computed gradients.
 *
 * @param learning_rate The learning rate for updating gamma and beta.
 */
void BatchNormalizationLayer::updateParameters(double learning_rate)
{
    for (int i = 0; i < gamma.size(); ++i)
    {
        gamma(i) -= learning_rate * dgamma(i);
        beta(i) -= learning_rate * dbeta(i);
    }
}
