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

#ifndef COMMON_HPP
#define COMMON_HPP

#include <cmath>
#include <tuple>
#include <mutex>
#include <queue>
#include <thread>
#include <future>
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <random>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <filesystem>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <opencv2/opencv.hpp>
#include <condition_variable>
#include <unsupported/Eigen/CXX11/Tensor>

// Macro definitions

/**
 * @def LOADING_PROGRESS
 * @brief Macro to enable loading progress display.
 */
// #define LOADING_PROGRESS

/**
 * @def AUGMENT_PROGRESS
 * @brief Macro to enable augmentation progress display.
 */
// #define AUGMENT_PROGRESS

/**
 * @def SAVE_BATCHES
 * @brief Macro to enable baching saves to disk.
 */
// #define SAVE_BATCHES

// Enum class declarations

/**
 * @enum ConvKernelInitialization
 * @brief Enumeration for convolutional kernel initialization methods.
 */
enum class ConvKernelInitialization
{
    HE,           /**< He initialization */
    XAVIER,       /**< Xavier initialization */
    RANDOM_NORMAL /**< Random normal initialization */
};

/**
 * @enum ConvBiasInitialization
 * @brief Enumeration for convolutional bias initialization methods.
 */
enum class ConvBiasInitialization
{
    ZERO,          /**< Initialize biases to zero */
    RANDOM_NORMAL, /**< Random normal initialization */
    NONE           /**< No initialization */
};

/**
 * @enum DenseWeightInitialization
 * @brief Enumeration for dense layer weight initialization methods.
 */
enum class DenseWeightInitialization
{
    XAVIER,       /**< Xavier initialization */
    HE,           /**< He initialization */
    RANDOM_NORMAL /**< Random normal initialization */
};

/**
 * @enum DenseBiasInitialization
 * @brief Enumeration for dense layer bias initialization methods.
 */
enum class DenseBiasInitialization
{
    ZERO,          /**< Initialize biases to zero */
    RANDOM_NORMAL, /**< Random normal initialization */
    NONE           /**< No initialization */
};

/**
 * @enum ActivationType
 * @brief Enumeration for activation function types.
 */
enum class ActivationType
{
    RELU,       /**< Rectified Linear Unit activation */
    LEAKY_RELU, /**< Leaky ReLU activation */
    SIGMOID,    /**< Sigmoid activation */
    TANH,       /**< Hyperbolic tangent activation */
    SOFTMAX,    /**< Softmax activation */
    ELU         /**< Exponential Linear Unit activation */
};

/**
 * @enum OptimizerType
 * @brief Enumeration for optimizer types.
 */
enum class OptimizerType
{
    SGD,             /**< Stochastic Gradient Descent */
    SGDWithMomentum, /**< SGD with momentum */
    Adam,            /**< Adam optimizer */
    RMSprop          /**< RMSprop optimizer */
};

/**
 * @enum LossType
 * @brief Enumeration for loss function types.
 */
enum class LossType
{
    BINARY_CROSS_ENTROPY,     /**< Binary cross entropy loss */
    MEAN_SQUARED_ERROR,       /**< Mean squared error loss */
    CATEGORICAL_CROSS_ENTROPY /**< Categorical cross entropy loss */
};

/**
 * @enum BNTarget
 * @brief An enumeration to define the target type for batch normalization: Convolutional or Fully Connected layer.
 */
enum class BNTarget
{
    ConvolutionLayer, /**< Batch Normalization for Convolution Layer */
    DenseLayer,       /**< Batch Normalization for FullyConnected Layer */
    None              /**< No layer specified => Represents Error */
};

/**
 * @enum BNMode
 * @brief An enumeration to define whether the Batch Normalization is doing inference of training.
 */
enum class BNMode
{
    Inference, /**< Inference Mode */
    Training,  /**< Training Mode */
};

/**
 * @enum PropagationType
 * @brief Enumeration for propagation types during training.
 */
enum class PropagationType
{
    FORWARD, /**< Forward propagation */
    BACK     /**< Backward propagation */
};

/**
 * @enum GradientClippingMode
 * @brief Enumeration for gradient clipping modes.
 */
enum class GradientClippingMode
{
    ENABLED, /**< Gradient clipping enabled */
    DISABLED /**< Gradient clipping disabled */
};

/**
 * @enum BatchType
 * @brief Enumeration for batch types.
 */
enum class BatchType
{
    Training, /**< Training batch */
    Testing   /**< Testing batch */
};

/**
 * @enum BatchMode
 * @brief Enumeration for batch organizations.
 */
enum class BatchMode
{
    UniformDistribution,
    ShuffleOnly
};

/**
 * @enum ELRALES_Mode
 * @brief Enumeration for ELRALES modes.
 */
enum class ELRALES_Mode
{
    ENABLED, /**< ELRALES enabled */
    DISABLED /**< ELRALES disabled */
};

/**
 * @enum ELRALES_StateMachine
 * @brief Enumeration for ELRALES state machine states.
 */
enum class ELRALES_StateMachine
{
    NORMAL,        /**< Normal state */
    RECOVERY,      /**< Recovery state */
    LOSING,        /**< Losing state */
    DONE,          /**< Done state */
    EARLY_STOPPING /**< Early stopping state */
};

/**
 * @enum ELRALES_Retval
 * @brief Enumeration for ELRALES return values.
 */
enum class ELRALES_Retval
{
    WASTED_EPOCH,     /**< Wasted epoch */
    SUCCESSFUL_EPOCH, /**< Successful epoch */
    END_LEARNING      /**< End learning */
};

/**
 * @enum LogLevel
 * @brief Enumeration for log levels.
 */
enum class LogLevel
{
    None,         /**< No logging */
    LayerSummary, /**< Log layer forward and backward summary */
    FullTensor    /**< Log full tensors */
};

/**
 * @enum ProgressLevel
 * @brief Enumeration for progress levels.
 */
enum class ProgressLevel
{
    None,        /**< No progress display */
    Time,        /**< Display time only */
    Progress,    /**< Display progress only */
    ProgressTime /**< Display both progress and time */
};

/**
 * @enum AugmentTarget
 * @brief Enumeration for data augmentation targets.
 */
enum class AugmentTarget
{
    TRAIN_DATASET,     /**< Augment training dataset */
    TEST_DATASET,      /**< Augment testing dataset */
    SINGLE_PREDICTION, /**< Augment single prediction images */
    WHOLE_DATASET,     /**< Augment whole dataset */
    NONE               /**< No augmentation */
};

/**
 * @enum LearningDecayType
 * @brief Enumeration for learning rate decay types.
 */
enum class LearningDecayType
{
    NONE,         /**< No decay */
    EXPONENTIAL,  /**< Exponential decay */
    STEP,         /**< Step decay */
    POLYNOMIAL,   /**< Polynomial decay */
    INVERSE_TIME, /**< Inverse time decay */
    COSINE        /**< Cosine decay */
};

// Function declarations for converting enums to strings

/**
 * @brief Convert ConvKernelInitialization enum to string.
 * @param value The ConvKernelInitialization value.
 * @return The string representation of the value.
 */
std::string toString(ConvKernelInitialization value);

/**
 * @brief Convert ConvBiasInitialization enum to string.
 * @param value The ConvBiasInitialization value.
 * @return The string representation of the value.
 */
std::string toString(ConvBiasInitialization value);

/**
 * @brief Convert DenseWeightInitialization enum to string.
 * @param value The DenseWeightInitialization value.
 * @return The string representation of the value.
 */
std::string toString(DenseWeightInitialization value);

/**
 * @brief Convert DenseBiasInitialization enum to string.
 * @param value The DenseBiasInitialization value.
 * @return The string representation of the value.
 */
std::string toString(DenseBiasInitialization value);

/**
 * @brief Convert ActivationType enum to string.
 * @param value The ActivationType value.
 * @return The string representation of the value.
 */
std::string toString(ActivationType value);

/**
 * @brief Convert OptimizerType enum to string.
 * @param value The OptimizerType value.
 * @return The string representation of the value.
 */
std::string toString(OptimizerType value);

/**
 * @brief Convert LossType enum to string.
 * @param value The LossType value.
 * @return The string representation of the value.
 */
std::string toString(LossType value);

/**
 * @brief Convert BNTarget enum to string.
 * @param value The BNTarget value.
 * @return The string representation of the value.
 */
std::string toString(BNTarget value);

/**
 * @brief Convert BNMode enum to string.
 * @param value The BNMode value.
 * @return The string representation of the value.
 */
std::string toString(BNMode value);

/**
 * @brief Convert PropagationType enum to string.
 * @param value The PropagationType value.
 * @return The string representation of the value.
 */
std::string toString(PropagationType value);

/**
 * @brief Convert GradientClippingMode enum to string.
 * @param value The GradientClippingMode value.
 * @return The string representation of the value.
 */
std::string toString(GradientClippingMode value);

/**
 * @brief Convert BatchType enum to string.
 * @param value The BatchType value.
 * @return The string representation of the value.
 */
std::string toString(BatchType value);

/**
 * @brief Convert BatchMode enum to string.
 * @param value The BatchMode value.
 * @return The string representation of the value.
 */
std::string toString(BatchMode value);

/**
 * @brief Convert ELRALES_Mode enum to string.
 * @param value The ELRALES_Mode value.
 * @return The string representation of the value.
 */
std::string toString(ELRALES_Mode value);

/**
 * @brief Convert ELRALES_StateMachine enum to string.
 * @param value The ELRALES_StateMachine value.
 * @return The string representation of the value.
 */
std::string toString(ELRALES_StateMachine value);

/**
 * @brief Convert ELRALES_Retval enum to string.
 * @param value The ELRALES_Retval value.
 * @return The string representation of the value.
 */
std::string toString(ELRALES_Retval value);

/**
 * @brief Convert LogLevel enum to string.
 * @param value The LogLevel value.
 * @return The string representation of the value.
 */
std::string toString(LogLevel value);

/**
 * @brief Convert ProgressLevel enum to string.
 * @param value The ProgressLevel value.
 * @return The string representation of the value.
 */
std::string toString(ProgressLevel value);

/**
 * @brief Convert AugmentTarget enum to string.
 * @param value The AugmentTarget value.
 * @return The string representation of the value.
 */
std::string toString(AugmentTarget value);

/**
 * @brief Convert LearningDecayType enum to string.
 * @param value The LearningDecayType value.
 * @return The string representation of the value.
 */
std::string toString(LearningDecayType value);

#endif // COMMON_HPP
