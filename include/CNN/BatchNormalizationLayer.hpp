#ifndef BATCHNORMALIZATIONLAYER_HPP
#define BATCHNORMALIZATIONLAYER_HPP

#include "Layer.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

class BatchNormalizationLayer : public Layer
{
public:
    // Constructor with optional epsilon and momentum parameters
    BatchNormalizationLayer(double epsilon, double momentum);

    // Forward pass of batch normalization
    Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4> &input_batch) override;

    // Backward pass of batch normalization
    Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4> &d_output_batch, const Eigen::Tensor<double, 4> &input_batch, double learning_rate) override;

    // Determines if the layer needs an optimizer
    bool needsOptimizer() const override;

    // Set optimizer for the layer
    void setOptimizer(std::shared_ptr<Optimizer> optimizer) override;

    // Get the optimizer from the layer
    std::shared_ptr<Optimizer> getOptimizer() override;

private:
    // Initialize parameters for batch normalization
    void initialize(int feature_size);

    // Update gamma and beta parameters
    void updateParameters(double learning_rate);

    double epsilon;   // Small constant to prevent division by zero
    double momentum;  // Momentum for moving averages
    bool initialized; // Flag to check if the layer is initialized

    Eigen::Tensor<double, 1> gamma, beta;                  // Scale and shift parameters
    Eigen::Tensor<double, 1> moving_mean, moving_variance; // Moving averages for mean and variance
    Eigen::Tensor<double, 1> dgamma, dbeta;                // Gradients for gamma and beta

    Eigen::Tensor<double, 4> cache_normalized;           // Cached normalized input
    Eigen::Tensor<double, 1> cache_mean, cache_variance; // Cached mean and variance
};

#endif // BATCHNORMALIZATIONLAYER_HPP
