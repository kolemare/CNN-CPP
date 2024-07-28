#ifndef COMMON_HPP
#define COMMON_HPP

#include <iostream>
#include <string>

// Macro definitions

#define LOADING_PROGRESS
#define AUGMENT_PROGRESS

// Enum class declarations

enum class ActivationType
{
    RELU,
    LEAKY_RELU,
    SIGMOID,
    TANH,
    SOFTMAX,
    ELU
};

enum class OptimizerType
{
    SGD,
    SGDWithMomentum,
    Adam,
    RMSprop
};

enum class BatchType
{
    Training,
    Testing
};

enum class ConvKernelInitialization
{
    HE,
    XAVIER,
    RANDOM_NORMAL
};

enum class ConvBiasInitialization
{
    ZERO,
    RANDOM_NORMAL,
    NONE
};

enum class DenseWeightInitialization
{
    XAVIER,
    HE,
    RANDOM_NORMAL
};

enum class DenseBiasInitialization
{
    ZERO,
    RANDOM_NORMAL,
    NONE
};

enum class LossType
{
    BINARY_CROSS_ENTROPY,
    MEAN_SQUARED_ERROR,
    CATEGORICAL_CROSS_ENTROPY
};

enum class LogLevel
{
    None,
    LayerOutputs,
    All
};

enum class ProgressLevel
{
    None,
    Time,
    Progress,
    ProgressTime
};

enum class PropagationType
{
    FORWARD,
    BACK
};

enum class GradientClippingMode
{
    ENABLED,
    DISABLED
};

enum class AugmentTarget
{
    TRAIN_DATASET,
    TEST_DATASET,
    WHOLE_DATASET,
    NONE
};

#endif // COMMON_HPP
