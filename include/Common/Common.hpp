#ifndef COMMON_HPP
#define COMMON_HPP

#include <iostream>
#include <string>

// Macro definitions

#define LOADING_PROGRESS
#define AUGMENT_PROGRESS

// Enum class declarations

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

enum class LossType
{
    BINARY_CROSS_ENTROPY,
    MEAN_SQUARED_ERROR,
    CATEGORICAL_CROSS_ENTROPY
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

enum class BatchType
{
    Training,
    Testing
};

enum class ELRALES_Mode
{
    ENABLED,
    DISABLED
};

enum class ELRALES_StateMachine
{
    NORMAL,
    RECOVERY,
    LOSING,
    DONE,
    EARLY_STOPPING
};

enum class ELRALES_Retval
{
    WASTED_EPOCH,
    SUCCESSFUL_EPOCH,
    END_LEARNING
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

enum class AugmentTarget
{
    TRAIN_DATASET,
    TEST_DATASET,
    WHOLE_DATASET,
    NONE
};

enum class LearningDecayType
{
    NONE,
    EXPONENTIAL,
    STEP,
    POLYNOMIAL,
    INVERSE_TIME,
    COSINE
};

// Function declarations for converting enums to strings
std::string toString(ConvKernelInitialization value);
std::string toString(ConvBiasInitialization value);
std::string toString(DenseWeightInitialization value);
std::string toString(DenseBiasInitialization value);
std::string toString(ActivationType value);
std::string toString(OptimizerType value);
std::string toString(LossType value);
std::string toString(PropagationType value);
std::string toString(GradientClippingMode value);
std::string toString(BatchType value);
std::string toString(ELRALES_Mode value);
std::string toString(ELRALES_StateMachine value);
std::string toString(ELRALES_Retval value);
std::string toString(LogLevel value);
std::string toString(ProgressLevel value);
std::string toString(AugmentTarget value);
std::string toString(LearningDecayType value);

#endif // COMMON_HPP
