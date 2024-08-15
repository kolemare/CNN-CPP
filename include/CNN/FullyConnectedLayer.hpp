#ifndef FULLYCONNECTEDLAYER_HPP
#define FULLYCONNECTEDLAYER_HPP

#include "Layer.hpp"

/**
 * @brief Class representing a fully connected layer in a neural network.
 *
 * This class implements the functionalities for a fully connected layer,
 * including initialization, forward pass, backward pass, and parameter management.
 */
class FullyConnectedLayer : public Layer
{
public:
    /**
     * @brief Construct a new FullyConnectedLayer object.
     *
     * Initializes a fully connected layer with the specified output size, weight initialization,
     * bias initialization, and random seed.
     *
     * @param output_size The number of output neurons in the layer.
     * @param weight_init The method to initialize the weights (default is Xavier).
     * @param bias_init The method to initialize the biases (default is Zero).
     * @param seed The random seed for initializing weights and biases (default is 42).
     */
    FullyConnectedLayer(int output_size,
                        DenseWeightInitialization weight_init = DenseWeightInitialization::XAVIER,
                        DenseBiasInitialization bias_init = DenseBiasInitialization::ZERO,
                        unsigned int seed = 42);

    /**
     * @brief Perform the forward pass for the fully connected layer.
     *
     * Computes the output of the layer by performing matrix multiplication with the input
     * and adding biases.
     *
     * @param input_batch The input tensor with dimensions (batch_size, depth, height, width).
     * @return Eigen::Tensor<double, 4> The output tensor with dimensions (batch_size, 1, 1, output_size).
     */
    Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4> &input_batch) override;

    /**
     * @brief Perform the backward pass for the fully connected layer.
     *
     * Computes the gradients with respect to the weights, biases, and input,
     * and updates the weights and biases using the optimizer.
     *
     * @param d_output_batch The gradient tensor with dimensions (batch_size, 1, 1, output_size).
     * @param input_batch The original input tensor with dimensions (batch_size, depth, height, width).
     * @param learning_rate The learning rate for updating the weights and biases.
     * @return Eigen::Tensor<double, 4> The gradient tensor with dimensions (batch_size, depth, height, width).
     */
    Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4> &d_output_batch,
                                      const Eigen::Tensor<double, 4> &input_batch,
                                      double learning_rate) override;

    /**
     * @brief Set the weights of the layer.
     *
     * @param new_weights The new weights tensor with dimensions (output_size, input_size).
     */
    void setWeights(const Eigen::Tensor<double, 2> &new_weights);

    /**
     * @brief Get the weights of the layer.
     *
     * @return Eigen::Tensor<double, 2> A copy of the weights tensor.
     */
    Eigen::Tensor<double, 2> getWeights() const;

    /**
     * @brief Set the biases of the layer.
     *
     * @param new_biases The new biases tensor with dimensions (output_size).
     */
    void setBiases(const Eigen::Tensor<double, 1> &new_biases);

    /**
     * @brief Get the biases of the layer.
     *
     * @return Eigen::Tensor<double, 1> A copy of the biases tensor.
     */
    Eigen::Tensor<double, 1> getBiases() const;

    /**
     * @brief Set the input size and initialize weights and biases.
     *
     * This function sets the input size for the layer and initializes the weights
     * and biases based on the specified initialization methods.
     *
     * @param input_size The size of the input to the layer.
     */
    void setInputSize(int input_size);

    /**
     * @brief Get the output size of the layer.
     *
     * @return int The number of output neurons in the layer.
     */
    int getOutputSize() const;

    /**
     * @brief Check if an optimizer is needed for the layer.
     *
     * @return true, as fully connected layers typically require an optimizer for training.
     */
    bool needsOptimizer() const override;

    /**
     * @brief Set the optimizer for the layer.
     *
     * @param optimizer A shared pointer to the optimizer object.
     */
    void setOptimizer(std::shared_ptr<Optimizer> optimizer) override;

    /**
     * @brief Get the optimizer used by the layer.
     *
     * @return std::shared_ptr<Optimizer> The optimizer used by the layer.
     */
    std::shared_ptr<Optimizer> getOptimizer() override;

private:
    int input_size;                   ///< Input size of the layer
    int output_size;                  ///< Output size of the layer
    Eigen::Tensor<double, 2> weights; ///< Weights of the layer
    Eigen::Tensor<double, 1> biases;  ///< Biases of the layer

    DenseWeightInitialization weight_init; ///< Weight initialization method
    DenseBiasInitialization bias_init;     ///< Bias initialization method
    unsigned int seed;                     ///< Seed for random number generation

    std::shared_ptr<Optimizer> optimizer; ///< Optimizer

    /**
     * @brief Initialize the weights of the layer.
     *
     * This function initializes the weights using the specified initialization method.
     */
    void initializeWeights();

    /**
     * @brief Initialize the biases of the layer.
     *
     * This function initializes the biases using the specified initialization method.
     */
    void initializeBiases();
};

#endif // FULLYCONNECTEDLAYER_HPP
