#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "Layer.hpp"
#include "LossFunction.hpp"
#include "Optimizer.hpp"
#include "ImageContainer.hpp"
#include "BatchManager.hpp"
#include "ConvolutionLayer.hpp"
#include "MaxPoolingLayer.hpp"
#include "AveragePoolingLayer.hpp"
#include "FlattenLayer.hpp"
#include "FullyConnectedLayer.hpp"
#include "ActivationLayer.hpp"
#include "BatchNormalizationLayer.hpp"
#include "GradientClipping.hpp"
#include "NNLogger.hpp"
#include "ELRALES.hpp"
#include "LearningDecay.hpp"

/**
 * @class NeuralNetwork
 * @brief Represents a configurable neural network with various layers and training capabilities.
 *
 * This class provides methods to construct, train, and evaluate a neural network.
 * It supports multiple types of layers, loss functions, optimizers, and learning rate strategies.
 */
class NeuralNetwork
{
public:
    /**
     * @brief Constructs a new NeuralNetwork object.
     */
    NeuralNetwork();

    /**
     * @brief Adds a convolutional layer to the neural network.
     *
     * @param filters The number of filters in the convolutional layer.
     * @param kernel_size The size of the convolution kernel.
     * @param stride The stride of the convolution (default is 1).
     * @param padding The amount of zero-padding to add to the input (default is 1).
     * @param kernel_init The method used to initialize the convolution kernels (default is Xavier initialization).
     * @param bias_init The method used to initialize the biases (default is zero initialization).
     */
    void addConvolutionLayer(int filters,
                             int kernel_size,
                             int stride = 1,
                             int padding = 1,
                             ConvKernelInitialization kernel_init = ConvKernelInitialization::XAVIER,
                             ConvBiasInitialization bias_init = ConvBiasInitialization::ZERO);

    /**
     * @brief Adds a max pooling layer to the neural network.
     *
     * @param pool_size The size of the pooling window.
     * @param stride The stride of the pooling operation.
     */
    void addMaxPoolingLayer(int pool_size,
                            int stride);

    /**
     * @brief Adds an average pooling layer to the neural network.
     *
     * @param pool_size The size of the pooling window.
     * @param stride The stride of the pooling operation.
     */
    void addAveragePoolingLayer(int pool_size,
                                int stride);

    /**
     * @brief Adds a flatten layer to the neural network.
     *
     * This layer reshapes the input into a 1D vector, typically used before fully connected layers.
     */
    void addFlattenLayer();

    /**
     * @brief Adds a fully connected layer to the neural network.
     *
     * @param output_size The number of output neurons in the layer.
     * @param weight_init The method used to initialize the weights (default is Xavier initialization).
     * @param bias_init The method used to initialize the biases (default is zero initialization).
     */
    void addFullyConnectedLayer(int output_size,
                                DenseWeightInitialization weight_init = DenseWeightInitialization::XAVIER,
                                DenseBiasInitialization bias_init = DenseBiasInitialization::ZERO);

    /**
     * @brief Adds an activation layer to the neural network.
     *
     * @param type The type of activation function to use.
     */
    void addActivationLayer(ActivationType type);

    /**
     * @brief Adds a batch normalization layer to the neural network.
     *
     * @param epsilon A small constant to avoid division by zero (default is 1e-5).
     * @param momentum The momentum for the moving average of mean and variance (default is 0.9).
     */
    void addBatchNormalizationLayer(double epsilon = 1e-5, double momentum = 0.9);

    /**
     * @brief Sets the loss function for the neural network.
     *
     * @param type The type of loss function to use.
     */
    void setLossFunction(LossType type);

    /**
     * @brief Sets the log level for the neural network.
     *
     * @param level The desired log level.
     */
    void setLogLevel(LogLevel level);

    /**
     * @brief Sets the progress level for the neural network.
     *
     * @param level The desired progress level.
     */
    void setProgressLevel(ProgressLevel level);

    /**
     * @brief Enables gradient clipping for the neural network.
     *
     * @param clipValue The maximum allowed value for gradients (default is 1.0).
     * @param mode The mode for gradient clipping (default is enabled).
     */
    void enableGradientClipping(double clipValue = 1.0,
                                GradientClippingMode mode = GradientClippingMode::ENABLED);

    /**
     * @brief Compiles the neural network by setting up the optimizer and preparing layers.
     *
     * @param optimizerType The type of optimizer to use.
     * @param optimizer_params A map of parameters specific to the optimizer type.
     */
    void compile(OptimizerType optimizerType,
                 const std::unordered_map<std::string, double> &optimizer_params = {});

    /**
     * @brief Performs a forward pass through the neural network.
     *
     * @param input A 4D tensor representing the input data.
     * @return A 4D tensor containing the output of the network.
     */
    Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4> &input);

    /**
     * @brief Performs a backward pass through the neural network.
     *
     * @param d_output A 4D tensor representing the gradient of the loss with respect to the output.
     * @param learning_rate The learning rate for updating parameters.
     */
    void backward(const Eigen::Tensor<double, 4> &d_output,
                  double learning_rate);

    /**
     * @brief Trains the neural network using the provided data.
     *
     * @param imageContainer The container holding the training images and labels.
     * @param epochs The number of training epochs.
     * @param batch_size The size of each training batch.
     * @param learning_rate The initial learning rate for training (default is 0.001).
     */
    void train(const ImageContainer &imageContainer,
               int epochs,
               int batch_size,
               double learning_rate = 0.001);

    /**
     * @brief Evaluates the neural network on the provided data.
     *
     * @param imageContainer The container holding the test images and labels.
     * @return A tuple containing the accuracy and loss on the test data.
     */
    std::tuple<double, double> evaluate(const ImageContainer &imageContainer);

    /**
     * @brief Makes predictions on single images and outputs the results.
     *
     * This method retrieves a batch of images intended for single prediction,
     * performs forward propagation, and outputs the predicted category and confidence
     * for each image.
     *
     * @param imageContainer The container holding the images to predict.
     */
    void makeSinglePredictions(const ImageContainer &imageContainer);

    /**
     * @brief Sets the input image size for the neural network.
     *
     * @param targetWidth The target width of the input images.
     * @param targetHeight The target height of the input images.
     */
    void setImageSize(const int targetWidth, const int targetHeight);

    /**
     * @brief Sets the path to csv that will be created for training and testing metrics.
     *
     * @param outputCSV Relative path to csv that will be created for training and testing metrics.
     */
    void setCSVPath(std::string outputCSV);

    /**
     * @brief Enables the ELRALES algorithm for adaptive learning and early stopping.
     *
     * @param learning_rate_coef Coefficient for adjusting the learning rate (default is 0.5).
     * @param maxSuccessiveEpochFailures Maximum number of successive epochs allowed to fail (default is 3).
     * @param maxEpochFails Maximum number of total epochs allowed to fail (default is 10).
     * @param tolerance The tolerance level for considering an epoch successful (default is 0.0).
     * @param mode The mode of operation for ELRALES (default is enabled).
     */
    void enableELRALES(double learning_rate_coef = 0.5,
                       int maxSuccessiveEpochFailures = 3,
                       int maxEpochFails = 10,
                       double tolerance = 0.0,
                       ELRALES_Mode mode = ELRALES_Mode::ENABLED);

    /**
     * @brief Enables learning rate decay for the neural network.
     *
     * @param decayType The type of learning rate decay to use.
     * @param params A map of parameters specific to the decay type.
     */
    void enableLearningDecay(LearningDecayType decayType,
                             const std::unordered_map<std::string, double> &params);

private:
    std::vector<std::shared_ptr<Layer>> layers;        ///< Vector of layers in the network.
    std::unique_ptr<LossFunction> lossFunction;        ///< The loss function used by the network.
    std::vector<Eigen::Tensor<double, 4>> layerInputs; ///< Cached inputs for each layer.

    bool flattenAdded;  ///< Flag indicating if a flatten layer has been added.
    bool clippingSet;   ///< Flag indicating if gradient clipping has been set.
    bool elralesSet;    ///< Flag indicating if ELRALES has been set.
    bool compiled;      ///< Flag indicating if the network has been compiled.
    bool trained;       ///< Flag indicating if the network has been trained.
    bool adamIncrement; ///< Flag indicating if the network used at least 1 Adam.

    int currentDepth; ///< Current depth of the input as it passes through layers.
    int inputSize;    ///< Input size for fully connected layers.
    int inputHeight;  ///< Input image height.
    int inputWidth;   ///< Input image width.
    int batchSize;    ///< Size of the batch used for predictions.

    LogLevel logLevel;           ///< Logging level for the network.
    ProgressLevel progressLevel; ///< Progress level for training output.

    double clipValue; ///< Value for gradient clipping.

    GradientClippingMode clippingMode = GradientClippingMode::DISABLED;      ///< Mode for gradient clipping.
    ELRALES_Mode elralesMode = ELRALES_Mode::DISABLED;                       ///< Mode for ELRALES.
    ELRALES_StateMachine elralesStateMachine = ELRALES_StateMachine::NORMAL; ///< Current state of the ELRALES state machine.
    std::vector<ELRALES_StateMachine> elralesStateMachineTimeLine{};         ///< Timeline of ELRALES state machine states.

    std::unique_ptr<LearningDecay> learningDecay;                  ///< Learning decay strategy.
    LearningDecayType learningDecayMode = LearningDecayType::NONE; ///< Type of learning decay used.

    // ELRALES parameters
    double learning_rate_coef;                  ///< Learning rate coefficient for ELRALES.
    int maxSuccessiveEpochFailures;             ///< Max successive epoch failures for ELRALES.
    int maxEpochFailures;                       ///< Max total epoch failures for ELRALES.
    double tolerance;                           ///< Tolerance for ELRALES.
    std::unique_ptr<ELRALES> elrales = nullptr; ///< ELRALES instance for adaptive learning and early stopping.
    std::string outputCSV;                      ///< Relative path to the output csv of training and testing metrics

    /**
     * @brief Resets the members when making changes to network to their initial state.
     *
     * This method sets compiled, trained, currentDepth and batchSize to initial values.
     * hardReset assures that no further changes are allowed to network after method compile.
     * It is possible to call compile multiple times, but there is no necessity for that.
     *
     */
    void hardReset();
};

#endif // NEURALNETWORK_HPP
