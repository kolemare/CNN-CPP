#ifndef BATCHNORMALIZATIONLAYER_HPP
#define BATCHNORMALIZATIONLAYER_HPP

#include "Layer.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

/**
 * @class BatchNormalizationLayer
 * @brief A class that implements batch normalization for neural network layers.
 */
class BatchNormalizationLayer : public Layer
{
public:
    /**
     * @brief Constructs a BatchNormalizationLayer with the specified epsilon and momentum.
     *
     * @param epsilon A small constant to avoid division by zero during normalization.
     * @param momentum The momentum for the moving average of mean and variance.
     */
    BatchNormalizationLayer(double epsilon,
                            double momentum);

    /**
     * @brief Forward pass of batch normalization.
     *
     * @param input_batch A 4D tensor containing the input data.
     * @return A 4D tensor containing the normalized output data.
     */
    Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4> &input_batch) override;

    /**
     * @brief Backward pass of batch normalization.
     *
     * @param d_output_batch A 4D tensor containing the gradient of the loss with respect to the output.
     * @param input_batch A 4D tensor containing the input data.
     * @param learning_rate The learning rate for updating gamma and beta.
     * @return A 4D tensor containing the gradient of the loss with respect to the input.
     */
    Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4> &d_output_batch,
                                      const Eigen::Tensor<double, 4> &input_batch,
                                      double learning_rate) override;

    /**
     * @brief Determines if the layer needs an optimizer.
     *
     * @return False as batch normalization does not use an optimizer.
     */
    bool needsOptimizer() const override;

    /**
     * @brief Set optimizer for the layer.
     *
     * @param optimizer A shared pointer to an optimizer.
     */
    void setOptimizer(std::shared_ptr<Optimizer> optimizer) override;

    /**
     * @brief Get the optimizer from the layer.
     *
     * @return A null pointer as batch normalization does not have an optimizer.
     */
    std::shared_ptr<Optimizer> getOptimizer() override;

private:
    /**
     * @brief Initialize parameters for batch normalization.
     *
     * @param feature_size The size of the feature to initialize parameters.
     */
    void initialize(int feature_size);

    /**
     * @brief Update gamma and beta parameters.
     *
     * @param learning_rate The learning rate for updating gamma and beta.
     */
    void updateParameters(double learning_rate);

    double epsilon;   ///< Small constant to prevent division by zero.
    double momentum;  ///< Momentum for moving averages.
    bool initialized; ///< Flag to check if the layer is initialized.

    Eigen::Tensor<double, 1> gamma, beta;                  ///< Scale and shift parameters.
    Eigen::Tensor<double, 1> moving_mean, moving_variance; ///< Moving averages for mean and variance.
    Eigen::Tensor<double, 1> dgamma, dbeta;                ///< Gradients for gamma and beta.

    Eigen::Tensor<double, 4> cache_normalized;           ///< Cached normalized input.
    Eigen::Tensor<double, 1> cache_mean, cache_variance; ///< Cached mean and variance.
};

#endif // BATCHNORMALIZATIONLAYER_HPP
