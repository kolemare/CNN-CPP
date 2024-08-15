#ifndef LAYER_HPP
#define LAYER_HPP

#include "Common.hpp"
#include "Optimizer.hpp"

/**
 * @class Layer
 * @brief An abstract base class representing a layer in a neural network.
 *
 * This class defines the interface for all types of layers used in neural networks,
 * including methods for forward and backward passes, optimizer management, and
 * destructor.
 */
class Layer
{
public:
    /**
     * @brief Perform the forward pass of the layer.
     *
     * This method processes the input data through the layer and produces the output.
     *
     * @param input_batch A 4D tensor representing the input batch to the layer.
     * @return A 4D tensor representing the output from the layer.
     */
    virtual Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4> &input_batch) = 0;

    /**
     * @brief Perform the backward pass of the layer.
     *
     * This method computes the gradient of the loss with respect to the input of the layer,
     * updating internal parameters if necessary.
     *
     * @param d_output_batch A 4D tensor representing the gradient of the loss with respect to the output of the layer.
     * @param input_batch A 4D tensor representing the input batch to the layer.
     * @param learning_rate The learning rate to use for updating parameters.
     * @return A 4D tensor representing the gradient of the loss with respect to the input of the layer.
     */
    virtual Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4> &d_output_batch,
                                              const Eigen::Tensor<double, 4> &input_batch,
                                              double learning_rate) = 0;

    /**
     * @brief Check if the layer requires an optimizer.
     *
     * @return True if the layer requires an optimizer, false otherwise.
     */
    virtual bool needsOptimizer() const = 0;

    /**
     * @brief Set the optimizer for the layer.
     *
     * This method associates an optimizer with the layer for use during training.
     *
     * @param optimizer A shared pointer to an optimizer object.
     */
    virtual void setOptimizer(std::shared_ptr<Optimizer> optimizer) = 0;

    /**
     * @brief Get the optimizer associated with the layer.
     *
     * @return A shared pointer to the optimizer object used by the layer.
     */
    virtual std::shared_ptr<Optimizer> getOptimizer() = 0;

    /**
     * @brief Virtual destructor for the layer.
     *
     * Ensures that derived class destructors are called properly.
     */
    virtual ~Layer() = default;
};

#endif // LAYER_HPP
