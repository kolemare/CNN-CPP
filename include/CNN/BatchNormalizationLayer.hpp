/*
MIT License
Copyright (c) 2024 Marko Kostić

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

This project is the CNN-CPP Framework. Usage of this code is free, and 
uploading and using the code is also free, with a humble request to mention 
the origin of the implementation, the author Marko Kostić, and the repository 
link: https://github.com/kolemare/CNN-CPP.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
*/

#ifndef BATCHNORMALIZATIONLAYER_HPP
#define BATCHNORMALIZATIONLAYER_HPP

#include "Layer.hpp"

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

    /**
     * @brief Get the gamma parameter (scale factor).
     *
     * @return A copy of the gamma tensor.
     */
    Eigen::Tensor<double, 1> getGamma() const;

    /**
     * @brief Set the gamma parameter (scale factor).
     *
     * @param gamma A tensor to copy values from.
     */
    void setGamma(const Eigen::Tensor<double, 1> &gamma);

    /**
     * @brief Get the beta parameter (shift factor).
     *
     * @return A copy of the beta tensor.
     */
    Eigen::Tensor<double, 1> getBeta() const;

    /**
     * @brief Set the beta parameter (shift factor).
     *
     * @param beta A tensor to copy values from.
     */
    void setBeta(const Eigen::Tensor<double, 1> &beta);

    /**
     * @brief Set the target layer type (Convolutional or Dense).
     *
     * @param target The target layer type.
     */
    void setTarget(BNTarget target);

    /**
     * @brief Set the layer mode (Inference or Training).
     *
     * @param target The requested mode.
     */
    static void setMode(BNMode mode);

private:
    /**
     * @brief Update gamma and beta parameters.
     *
     * @param learning_rate The learning rate for updating gamma and beta.
     */
    void updateParameters(double learning_rate);

    /**
     * @brief Normalize the input for convolutional layers.
     *
     * @param input_batch The input tensor for normalization.
     * @return A normalized tensor.
     */
    Eigen::Tensor<double, 4> normalizeConvLayer(const Eigen::Tensor<double, 4> &input_batch);

    /**
     * @brief Normalize the input for fully connected layers.
     *
     * @param input_batch The input tensor for normalization.
     * @return A normalized tensor.
     */
    Eigen::Tensor<double, 4> normalizeDenseLayer(const Eigen::Tensor<double, 4> &input_batch);

    /**
     * @brief Compute the backward pass for convolutional layers.
     */
    Eigen::Tensor<double, 4> backwardConvLayer(const Eigen::Tensor<double, 4> &d_output_batch,
                                               const Eigen::Tensor<double, 4> &input_batch,
                                               double learning_rate);

    /**
     * @brief Compute the backward pass for fully connected layers.
     */
    Eigen::Tensor<double, 4> backwardDenseLayer(const Eigen::Tensor<double, 4> &d_output_batch,
                                                const Eigen::Tensor<double, 4> &input_batch,
                                                double learning_rate);

    double epsilon;             ///< Small constant to prevent division by zero.
    double momentum;            ///< Momentum for moving averages.
    bool initialized;           ///< Flag to check if the layer is initialized.
    BNTarget target;            ///< The target type (Convolution or Dense).
    static BNMode layerMode;    ///< Mode druing forward propagation (Inference or Training).

    Eigen::Tensor<double, 1> gamma, beta;                  ///< Scale and shift parameters.
    Eigen::Tensor<double, 1> moving_mean, moving_variance; ///< Moving averages for mean and variance.
    Eigen::Tensor<double, 1> dgamma, dbeta;                ///< Gradients for gamma and beta.

    Eigen::Tensor<double, 4> cache_normalized;           ///< Cached normalized input.
    Eigen::Tensor<double, 1> cache_mean, cache_variance; ///< Cached mean and variance.
};

#endif // BATCHNORMALIZATIONLAYER_HPP
