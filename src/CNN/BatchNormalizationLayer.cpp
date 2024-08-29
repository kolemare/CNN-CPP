#include "BatchNormalizationLayer.hpp"

// Initialize the static variable to Training mode
BNMode BatchNormalizationLayer::layerMode = BNMode::Training;

BatchNormalizationLayer::BatchNormalizationLayer(double epsilon, double momentum)
{
    this->epsilon = epsilon;
    this->momentum = momentum;
    this->initialized = false;
    this->target = BNTarget::None;
}

bool BatchNormalizationLayer::needsOptimizer() const
{
    return false; // Batch normalization does not use an optimizer
}

void BatchNormalizationLayer::setOptimizer(std::shared_ptr<Optimizer> optimizer)
{
    return; // No-op, as batch normalization does not use an optimizer
}

std::shared_ptr<Optimizer> BatchNormalizationLayer::getOptimizer()
{
    return nullptr; // Batch normalization does not have an optimizer
}

void BatchNormalizationLayer::setTarget(BNTarget target)
{
    this->target = target;
}

void BatchNormalizationLayer::setMode(BNMode mode)
{
    BatchNormalizationLayer::layerMode = mode;
}

Eigen::Tensor<double, 4> BatchNormalizationLayer::forward(const Eigen::Tensor<double, 4> &input_batch)
{
    // Forward pass based on the layer type
    if (BNTarget::ConvolutionLayer == target)
    {
        return normalizeConvLayer(input_batch);
    }
    else if (BNTarget::DenseLayer == target)
    {
        return normalizeDenseLayer(input_batch);
    }
    else
    {
        throw std::runtime_error("Unrecognized Layer For Batch Normalization (Possibly None)");
    }
}

Eigen::Tensor<double, 4> BatchNormalizationLayer::normalizeConvLayer(const Eigen::Tensor<double, 4> &input_batch)
{
    int batch_size = input_batch.dimension(0);
    int input_depth = input_batch.dimension(1);
    int height = input_batch.dimension(2);
    int width = input_batch.dimension(3);

    // Initialize parameters for the first time
    if (!initialized)
    {
        gamma = Eigen::Tensor<double, 1>(input_depth);
        beta = Eigen::Tensor<double, 1>(input_depth);
        moving_mean = Eigen::Tensor<double, 1>(input_depth);
        moving_variance = Eigen::Tensor<double, 1>(input_depth);
        dgamma = Eigen::Tensor<double, 1>(input_depth);
        dbeta = Eigen::Tensor<double, 1>(input_depth);
        cache_mean = Eigen::Tensor<double, 1>(input_depth);
        cache_variance = Eigen::Tensor<double, 1>(input_depth);
        cache_normalized = Eigen::Tensor<double, 4>(batch_size, input_depth, height, width);

        // Set initial values for parameters
        gamma.setConstant(1.0);           // Initialize scale to 1
        beta.setZero();                   // Initialize shift to 0
        moving_mean.setZero();            // Initialize moving mean to 0
        moving_variance.setConstant(1.0); // Initialize moving variance to 1

        initialized = true;
    }

    // Compute mean and variance for the current batch (per channel)
    Eigen::Tensor<double, 1> input_mean(input_depth);
    Eigen::Tensor<double, 1> input_variance(input_depth);

    if (BNMode::Training == BatchNormalizationLayer::layerMode)
    {
        // Compute mean and variance for the current batch (per channel)
        input_mean.setZero();
        input_variance.setZero();

        // Calculate mean for each channel
        for (int d = 0; d < input_depth; ++d)
        {
            for (int n = 0; n < batch_size; ++n)
            {
                for (int h = 0; h < height; ++h)
                {
                    for (int w = 0; w < width; ++w)
                    {
                        input_mean(d) += input_batch(n, d, h, w);
                    }
                }
            }
            input_mean(d) /= (batch_size * height * width);
        }

        // Calculate variance for each channel
        for (int d = 0; d < input_depth; ++d)
        {
            for (int n = 0; n < batch_size; ++n)
            {
                for (int h = 0; h < height; ++h)
                {
                    for (int w = 0; w < width; ++w)
                    {
                        double diff = input_batch(n, d, h, w) - input_mean(d);
                        input_variance(d) += diff * diff;
                    }
                }
            }
            input_variance(d) /= (batch_size * height * width);
        }

        // Cache the computed mean and variance for backward pass
        cache_mean = input_mean;
        cache_variance = input_variance;

        // Update moving mean and variance
        moving_mean = momentum * moving_mean + (1 - momentum) * input_mean;
        moving_variance = momentum * moving_variance + (1 - momentum) * input_variance;
    }
    else if (BNMode::Inference == BatchNormalizationLayer::layerMode)
    {
        // Use moving mean and variance during inference
        input_mean = moving_mean;
        input_variance = moving_variance;
    }
    else
    {
        throw std::runtime_error("Unknown BNMode");
    }

    // Normalize
    Eigen::Tensor<double, 4> normalized(batch_size, input_depth, height, width);
    for (int d = 0; d < input_depth; ++d)
    {
        double inv_sqrt_variance = 1.0 / std::sqrt(input_variance(d) + epsilon);
        for (int n = 0; n < batch_size; ++n)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    normalized(n, d, h, w) = (input_batch(n, d, h, w) - input_mean(d)) * inv_sqrt_variance;
                }
            }
        }
    }

    cache_normalized = normalized;

    // Scale and shift
    Eigen::Tensor<double, 4> output(batch_size, input_depth, height, width);
    for (int d = 0; d < input_depth; ++d)
    {
        for (int n = 0; n < batch_size; ++n)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    output(n, d, h, w) = normalized(n, d, h, w) * gamma(d) + beta(d);
                }
            }
        }
    }

    return output;
}

Eigen::Tensor<double, 4> BatchNormalizationLayer::normalizeDenseLayer(const Eigen::Tensor<double, 4> &input_batch)
{
    int batch_size = input_batch.dimension(0);
    int input_depth = input_batch.dimension(3); // Use the last dimension for fully connected layers

    // Initialize parameters for the first time
    if (!initialized)
    {
        gamma = Eigen::Tensor<double, 1>(input_depth);
        beta = Eigen::Tensor<double, 1>(input_depth);
        moving_mean = Eigen::Tensor<double, 1>(input_depth);
        moving_variance = Eigen::Tensor<double, 1>(input_depth);
        dgamma = Eigen::Tensor<double, 1>(input_depth);
        dbeta = Eigen::Tensor<double, 1>(input_depth);
        cache_mean = Eigen::Tensor<double, 1>(input_depth);
        cache_variance = Eigen::Tensor<double, 1>(input_depth);
        cache_normalized = Eigen::Tensor<double, 4>(batch_size, 1, 1, input_depth);

        // Set initial values for parameters
        gamma.setConstant(1.0);           // Initialize scale to 1
        beta.setZero();                   // Initialize shift to 0
        moving_mean.setZero();            // Initialize moving mean to 0
        moving_variance.setConstant(1.0); // Initialize moving variance to 1

        initialized = true;
    }

    // Compute mean and variance for the current batch (per channel)
    Eigen::Tensor<double, 1> input_mean(input_depth);
    Eigen::Tensor<double, 1> input_variance(input_depth);

    if (BNMode::Training == BatchNormalizationLayer::layerMode)
    {
        // Compute mean and variance for the current batch (per channel)
        input_mean.setZero();
        input_variance.setZero();

        // Calculate mean for each feature
        for (int d = 0; d < input_depth; ++d)
        {
            for (int n = 0; n < batch_size; ++n)
            {
                input_mean(d) += input_batch(n, 0, 0, d);
            }
            input_mean(d) /= batch_size;
        }

        // Calculate variance for each feature
        for (int d = 0; d < input_depth; ++d)
        {
            for (int n = 0; n < batch_size; ++n)
            {
                double diff = input_batch(n, 0, 0, d) - input_mean(d);
                input_variance(d) += diff * diff;
            }
            input_variance(d) /= batch_size;
        }

        // Cache the computed mean and variance for backward pass
        cache_mean = input_mean;
        cache_variance = input_variance;

        // Update moving mean and variance
        moving_mean = momentum * moving_mean + (1 - momentum) * input_mean;
        moving_variance = momentum * moving_variance + (1 - momentum) * input_variance;
    }
    else if (BNMode::Inference == BatchNormalizationLayer::layerMode)
    {
        // Use moving mean and variance during inference
        input_mean = moving_mean;
        input_variance = moving_variance;
    }
    else
    {
        throw std::runtime_error("Unknown BNMode");
    }

    // Normalize
    Eigen::Tensor<double, 4> normalized(batch_size, 1, 1, input_depth);
    for (int d = 0; d < input_depth; ++d)
    {
        double inv_sqrt_variance = 1.0 / std::sqrt(input_variance(d) + epsilon);
        for (int n = 0; n < batch_size; ++n)
        {
            normalized(n, 0, 0, d) = (input_batch(n, 0, 0, d) - input_mean(d)) * inv_sqrt_variance;
        }
    }

    cache_normalized = normalized;

    // Scale and shift
    Eigen::Tensor<double, 4> output(batch_size, 1, 1, input_depth);
    for (int d = 0; d < input_depth; ++d)
    {
        for (int n = 0; n < batch_size; ++n)
        {
            output(n, 0, 0, d) = normalized(n, 0, 0, d) * gamma(d) + beta(d);
        }
    }

    return output;
}

Eigen::Tensor<double, 4> BatchNormalizationLayer::backward(const Eigen::Tensor<double, 4> &d_output_batch,
                                                           const Eigen::Tensor<double, 4> &input_batch,
                                                           double learning_rate)
{
    if (BNTarget::ConvolutionLayer == target)
    {
        return backwardConvLayer(d_output_batch, input_batch, learning_rate);
    }
    else if (BNTarget::DenseLayer == target)
    {
        return backwardDenseLayer(d_output_batch, input_batch, learning_rate);
    }
    else
    {
        throw std::runtime_error("Unrecognized Layer For Batch Normalization (Possibly None)");
    }
}

Eigen::Tensor<double, 4> BatchNormalizationLayer::backwardConvLayer(const Eigen::Tensor<double, 4> &d_output_batch,
                                                                    const Eigen::Tensor<double, 4> &input_batch,
                                                                    double learning_rate)
{
    int batch_size = input_batch.dimension(0);
    int input_depth = input_batch.dimension(1);
    int height = input_batch.dimension(2);
    int width = input_batch.dimension(3);

    dgamma.setZero();
    dbeta.setZero();

    for (int d = 0; d < input_depth; ++d)
    {
        for (int n = 0; n < batch_size; ++n)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    dgamma(d) += d_output_batch(n, d, h, w) * cache_normalized(n, d, h, w);
                    dbeta(d) += d_output_batch(n, d, h, w);
                }
            }
        }
    }

    Eigen::Tensor<double, 1> d_variance(input_depth);
    d_variance.setZero();

    for (int d = 0; d < input_depth; ++d)
    {
        double inv_sqrt_variance = 1.0 / std::sqrt(cache_variance(d) + epsilon);

        for (int n = 0; n < batch_size; ++n)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    double diff = input_batch(n, d, h, w) - cache_mean(d);
                    d_variance(d) += d_output_batch(n, d, h, w) * diff * inv_sqrt_variance;
                }
            }
        }

        d_variance(d) *= -0.5 * std::pow(inv_sqrt_variance, 3);
    }

    Eigen::Tensor<double, 1> d_mean(input_depth);
    d_mean.setZero();

    for (int d = 0; d < input_depth; ++d)
    {
        double inv_sqrt_variance = 1.0 / std::sqrt(cache_variance(d) + epsilon);

        for (int n = 0; n < batch_size; ++n)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    d_mean(d) += d_output_batch(n, d, h, w) * inv_sqrt_variance;
                }
            }
        }

        for (int n = 0; n < batch_size; ++n)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    d_mean(d) += d_variance(d) * -2.0 * (input_batch(n, d, h, w) - cache_mean(d)) / (batch_size * height * width);
                }
            }
        }
    }

    Eigen::Tensor<double, 4> d_input(batch_size, input_depth, height, width);

    for (int d = 0; d < input_depth; ++d)
    {
        double inv_sqrt_variance = 1.0 / std::sqrt(cache_variance(d) + epsilon);

        for (int n = 0; n < batch_size; ++n)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    d_input(n, d, h, w) = d_output_batch(n, d, h, w) * inv_sqrt_variance -
                                          d_mean(d) / (batch_size * height * width) -
                                          d_variance(d) * 2.0 * (input_batch(n, d, h, w) - cache_mean(d)) / (batch_size * height * width);
                }
            }
        }
    }

    updateParameters(learning_rate);
    return d_input;
}

Eigen::Tensor<double, 4> BatchNormalizationLayer::backwardDenseLayer(const Eigen::Tensor<double, 4> &d_output_batch,
                                                                     const Eigen::Tensor<double, 4> &input_batch,
                                                                     double learning_rate)
{
    int batch_size = input_batch.dimension(0);
    int input_depth = input_batch.dimension(3);

    dgamma.setZero();
    dbeta.setZero();

    for (int d = 0; d < input_depth; ++d)
    {
        for (int n = 0; n < batch_size; ++n)
        {
            dgamma(d) += d_output_batch(n, 0, 0, d) * cache_normalized(n, 0, 0, d);
            dbeta(d) += d_output_batch(n, 0, 0, d);
        }
    }

    Eigen::Tensor<double, 1> d_variance(input_depth);
    d_variance.setZero();

    for (int d = 0; d < input_depth; ++d)
    {
        double inv_sqrt_variance = 1.0 / std::sqrt(cache_variance(d) + epsilon);

        for (int n = 0; n < batch_size; ++n)
        {
            double diff = input_batch(n, 0, 0, d) - cache_mean(d);
            d_variance(d) += d_output_batch(n, 0, 0, d) * diff * inv_sqrt_variance;
        }

        d_variance(d) *= -0.5 * std::pow(inv_sqrt_variance, 3);
    }

    Eigen::Tensor<double, 1> d_mean(input_depth);
    d_mean.setZero();

    for (int d = 0; d < input_depth; ++d)
    {
        double inv_sqrt_variance = 1.0 / std::sqrt(cache_variance(d) + epsilon);

        for (int n = 0; n < batch_size; ++n)
        {
            d_mean(d) += d_output_batch(n, 0, 0, d) * inv_sqrt_variance;
        }

        for (int n = 0; n < batch_size; ++n)
        {
            d_mean(d) += d_variance(d) * -2.0 * (input_batch(n, 0, 0, d) - cache_mean(d)) / batch_size;
        }
    }

    Eigen::Tensor<double, 4> d_input(batch_size, 1, 1, input_depth);

    for (int d = 0; d < input_depth; ++d)
    {
        double inv_sqrt_variance = 1.0 / std::sqrt(cache_variance(d) + epsilon);

        for (int n = 0; n < batch_size; ++n)
        {
            d_input(n, 0, 0, d) = d_output_batch(n, 0, 0, d) * inv_sqrt_variance -
                                  d_mean(d) / batch_size -
                                  d_variance(d) * 2.0 * (input_batch(n, 0, 0, d) - cache_mean(d)) / batch_size;
        }
    }

    updateParameters(learning_rate);
    return d_input;
}

void BatchNormalizationLayer::updateParameters(double learning_rate)
{
    for (int i = 0; i < gamma.size(); ++i)
    {
        gamma(i) -= learning_rate * dgamma(i);
        beta(i) -= learning_rate * dbeta(i);
    }
}

Eigen::Tensor<double, 1> BatchNormalizationLayer::getGamma() const
{
    return gamma;
}

void BatchNormalizationLayer::setGamma(const Eigen::Tensor<double, 1> &gamma)
{
    this->gamma = gamma;
}

Eigen::Tensor<double, 1> BatchNormalizationLayer::getBeta() const
{
    return beta;
}

void BatchNormalizationLayer::setBeta(const Eigen::Tensor<double, 1> &beta)
{
    this->beta = beta;
}
