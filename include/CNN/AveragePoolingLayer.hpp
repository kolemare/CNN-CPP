#ifndef AVERAGE_POOLING_LAYER_HPP
#define AVERAGE_POOLING_LAYER_HPP

#include "Layer.hpp"

/**
 * @brief A class representing an Average Pooling Layer in a neural network.
 */
class AveragePoolingLayer : public Layer
{
public:
    /**
     * @brief Constructs an AveragePoolingLayer with specified pool size and stride.
     *
     * @param pool_size The size of the pooling window.
     * @param stride The stride of the pooling operation.
     */
    AveragePoolingLayer(int pool_size,
                        int stride);

    /**
     * @brief Performs the forward pass using average pooling on the input batch.
     *
     * @param input_batch A 4D tensor representing the input batch.
     * @return A 4D tensor with the pooled output.
     */
    Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4> &input_batch) override;

    /**
     * @brief Performs the backward pass, computing gradients with respect to the input batch.
     *
     * @param d_output_batch A 4D tensor of gradients with respect to the output.
     * @param input_batch A 4D tensor representing the original input batch.
     * @param learning_rate The learning rate used for weight updates (not used here).
     * @return A 4D tensor of gradients with respect to the input.
     */
    Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4> &d_output_batch,
                                      const Eigen::Tensor<double, 4> &input_batch,
                                      double learning_rate) override;

    /**
     * @brief Indicates whether the layer needs an optimizer.
     *
     * @return False, as average pooling layers do not require an optimizer.
     */
    bool needsOptimizer() const override;

    /**
     * @brief Sets the optimizer for the layer.
     *
     * Average pooling layers do not use optimizers, so this function does nothing.
     *
     * @param optimizer A shared pointer to the optimizer.
     */
    void setOptimizer(std::shared_ptr<Optimizer> optimizer) override;

    /**
     * @brief Gets the optimizer associated with the layer.
     *
     * @return A nullptr, as average pooling layers do not use optimizers.
     */
    std::shared_ptr<Optimizer> getOptimizer() override;

    /**
     * @brief Gets the pool size used in the average pooling operation.
     *
     * @return The size of the pooling window.
     */
    int getPoolSize();

    /**
     * @brief Gets the stride used in the average pooling operation.
     *
     * @return The stride of the pooling operation.
     */
    int getStride();

private:
    int pool_size; /**< The size of the pooling window. */
    int stride;    /**< The stride of the pooling operation. */
};

#endif // AVERAGE_POOLING_LAYER_HPP
