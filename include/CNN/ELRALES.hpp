#ifndef ELRALES_HPP
#define ELRALES_HPP

#include "ConvolutionLayer.hpp"
#include "FullyConnectedLayer.hpp"
#include "BatchNormalizationLayer.hpp"

/**
 * @class ELRALES
 * @brief A class for managing learning rate adaptation and network state saving & loading in neural networks.
 *
 * ELRALES (Epoch Loss Recovery Adaptive Learning Early Stopping) is designed to manage the learning rate and save the
 * model state based on epoch performance. It helps to handle epochs with failures and decide when to recover the model
 * to best previous state or perform early stopping.
 *
 */
class ELRALES
{
public:
    /**
     * @brief Constructs an ELRALES object to manage learning rate and model state.
     *
     * This constructor initializes the ELRALES system with a specified learning rate coefficient,
     * maximum allowable epoch failures, tolerance, and a list of layers to manage.
     *
     * @param learning_rate_coef Coefficient for adjusting the learning rate.
     * @param maxSuccessiveEpochFailures Maximum number of successive epochs allowed to fail.
     * @param maxEpochFailures Maximum total number of epochs allowed to fail.
     * @param tolerance The tolerance level for considering an epoch as successful.
     * @param layers The layers of the neural network to be managed.
     *
     * @throws std::runtime_error if learning_rate_coef or tolerance is out of the range [0, 1].
     */
    ELRALES(double learning_rate_coef,
            int maxSuccessiveEpochFailures,
            int maxEpochFailures,
            double tolerance,
            const std::vector<std::shared_ptr<Layer>> &layers);

    /**
     * @brief Updates the state of the ELRALES system based on current epoch loss.
     *
     * Determines the state of the training process, adjusts learning rates if necessary,
     * and restores or saves states based on epoch success or failure.
     *
     * @param current_loss The loss observed in the current epoch.
     * @param layers The layers of the neural network to be managed.
     * @param learning_rate Reference to the current learning rate to be adjusted if needed.
     * @param mode The current mode of the ELRALES state machine, adjusted based on epoch outcome.
     *
     * @return An ELRALES_Retval indicating the result of the epoch (e.g., successful, wasted, end learning).
     */
    ELRALES_Retval updateState(double current_loss,
                               std::vector<std::shared_ptr<Layer>> &layers,
                               double &learning_rate,
                               ELRALES_StateMachine &mode);

private:
    double learning_rate_coef;      ///< Coefficient for learning rate adjustment.
    int maxSuccessiveEpochFailures; ///< Max number of consecutive failed epochs allowed.
    int maxEpochFailures;           ///< Max total number of failed epochs allowed.
    double tolerance;               ///< Tolerance level for determining epoch success.

    double best_loss;            ///< Best observed loss for saving state.
    double previous_loss;        ///< Previous epoch's loss for comparison.
    int successiveEpochFailures; ///< Count of successive failed epochs.
    int totalEpochFailures;      ///< Total count of failed epochs.

    /**
     * @struct OptimizerState
     * @brief Stores optimizer state information for saving and restoring.
     */
    struct OptimizerState
    {
        Eigen::Tensor<double, 2> v_weights_2d; ///< Velocity for 2D weights.
        Eigen::Tensor<double, 1> v_biases_2d;  ///< Velocity for 2D biases.
        Eigen::Tensor<double, 4> v_weights_4d; ///< Velocity for 4D weights.
        Eigen::Tensor<double, 1> v_biases_4d;  ///< Velocity for 4D biases.

        Eigen::Tensor<double, 2> m_weights_2d; ///< Momentum for 2D weights.
        Eigen::Tensor<double, 1> m_biases_2d;  ///< Momentum for 2D biases.
        Eigen::Tensor<double, 4> m_weights_4d; ///< Momentum for 4D weights.
        Eigen::Tensor<double, 1> m_biases_4d;  ///< Momentum for 4D biases.

        Eigen::Tensor<double, 2> s_weights_2d; ///< Squared gradients for 2D weights.
        Eigen::Tensor<double, 1> s_biases_2d;  ///< Squared gradients for 2D biases.
        Eigen::Tensor<double, 4> s_weights_4d; ///< Squared gradients for 4D weights.
        Eigen::Tensor<double, 1> s_biases_4d;  ///< Squared gradients for 4D biases.
    };

    /**
     * @struct ConvolutionLayerState
     * @brief Stores state of convolutional layers including weights, biases, and optimizer state.
     */
    struct ConvolutionLayerState
    {
        Eigen::Tensor<double, 4> kernels; ///< Kernels for convolution layer.
        Eigen::Tensor<double, 1> biases;  ///< Biases for convolution layer.
        OptimizerState optimizer_state;   ///< Optimizer state for convolution layer.
    };

    /**
     * @struct FullyConnectedLayerState
     * @brief Stores state of fully connected layers including weights, biases, and optimizer state.
     */
    struct FullyConnectedLayerState
    {
        Eigen::Tensor<double, 2> weights; ///< Weights for fully connected layer.
        Eigen::Tensor<double, 1> biases;  ///< Biases for fully connected layer.
        OptimizerState optimizer_state;   ///< Optimizer state for fully connected layer.
    };

    /**
     * @struct BatchNormalizationLayerState
     * @brief Stores state of batch normalization layers including gamma and beta parameters.
     */
    struct BatchNormalizationLayerState
    {
        Eigen::Tensor<double, 1> gamma; ///< Gamma (scale) for batch normalization layer.
        Eigen::Tensor<double, 1> beta;  ///< Beta (shift) for batch normalization layer.
    };

    std::vector<ConvolutionLayerState> savedConvLayerStates;        ///< Saved states for convolutional layers.
    std::vector<FullyConnectedLayerState> savedFCLayerStates;       ///< Saved states for fully connected layers.
    std::vector<BatchNormalizationLayerState> savedBatchNormStates; ///< Saved states for batch normalization layers.

    /**
     * @brief Saves the state of the layers for potential restoration.
     *
     * Captures the current state of the convolutional, fully connected, and batch normalization layers,
     * including their weights, biases, gamma, beta, and optimizer states, to allow restoring them if necessary.
     *
     * @param layers The layers of the neural network to save.
     */
    void saveState(const std::vector<std::shared_ptr<Layer>> &layers);

    /**
     * @brief Restores the state of the layers to a previously saved state.
     *
     * Restores the weights, biases, gamma, beta, and optimizer states of the convolutional,
     * fully connected, and batch normalization layers to a previously saved best state.
     *
     * @param layers The layers of the neural network to restore.
     */
    void restoreState(std::vector<std::shared_ptr<Layer>> &layers);
};

#endif // ELRALES_HPP
