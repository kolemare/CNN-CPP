#include "Common.hpp"

// Function to convert ConvKernelInitialization to string
std::string toString(ConvKernelInitialization value)
{
    switch (value)
    {
    case ConvKernelInitialization::HE:
        return "HE";
    case ConvKernelInitialization::XAVIER:
        return "XAVIER";
    case ConvKernelInitialization::RANDOM_NORMAL:
        return "RANDOM_NORMAL";
    default:
        return "UNKNOWN";
    }
}

// Function to convert ConvBiasInitialization to string
std::string toString(ConvBiasInitialization value)
{
    switch (value)
    {
    case ConvBiasInitialization::ZERO:
        return "ZERO";
    case ConvBiasInitialization::RANDOM_NORMAL:
        return "RANDOM_NORMAL";
    case ConvBiasInitialization::NONE:
        return "NONE";
    default:
        return "UNKNOWN";
    }
}

// Function to convert DenseWeightInitialization to string
std::string toString(DenseWeightInitialization value)
{
    switch (value)
    {
    case DenseWeightInitialization::XAVIER:
        return "XAVIER";
    case DenseWeightInitialization::HE:
        return "HE";
    case DenseWeightInitialization::RANDOM_NORMAL:
        return "RANDOM_NORMAL";
    default:
        return "UNKNOWN";
    }
}

// Function to convert DenseBiasInitialization to string
std::string toString(DenseBiasInitialization value)
{
    switch (value)
    {
    case DenseBiasInitialization::ZERO:
        return "ZERO";
    case DenseBiasInitialization::RANDOM_NORMAL:
        return "RANDOM_NORMAL";
    case DenseBiasInitialization::NONE:
        return "NONE";
    default:
        return "UNKNOWN";
    }
}

// Function to convert ActivationType to string
std::string toString(ActivationType value)
{
    switch (value)
    {
    case ActivationType::RELU:
        return "RELU";
    case ActivationType::LEAKY_RELU:
        return "LEAKY_RELU";
    case ActivationType::SIGMOID:
        return "SIGMOID";
    case ActivationType::TANH:
        return "TANH";
    case ActivationType::SOFTMAX:
        return "SOFTMAX";
    case ActivationType::ELU:
        return "ELU";
    default:
        return "UNKNOWN";
    }
}

// Function to convert OptimizerType to string
std::string toString(OptimizerType value)
{
    switch (value)
    {
    case OptimizerType::SGD:
        return "SGD";
    case OptimizerType::SGDWithMomentum:
        return "SGDWithMomentum";
    case OptimizerType::Adam:
        return "Adam";
    case OptimizerType::RMSprop:
        return "RMSprop";
    default:
        return "UNKNOWN";
    }
}

// Function to convert LossType to string
std::string toString(LossType value)
{
    switch (value)
    {
    case LossType::BINARY_CROSS_ENTROPY:
        return "BINARY_CROSS_ENTROPY";
    case LossType::MEAN_SQUARED_ERROR:
        return "MEAN_SQUARED_ERROR";
    case LossType::CATEGORICAL_CROSS_ENTROPY:
        return "CATEGORICAL_CROSS_ENTROPY";
    default:
        return "UNKNOWN";
    }
}

// Function to convert PropagationType to string
std::string toString(PropagationType value)
{
    switch (value)
    {
    case PropagationType::FORWARD:
        return "FORWARD";
    case PropagationType::BACK:
        return "BACK";
    default:
        return "UNKNOWN";
    }
}

// Function to convert GradientClippingMode to string
std::string toString(GradientClippingMode value)
{
    switch (value)
    {
    case GradientClippingMode::ENABLED:
        return "ENABLED";
    case GradientClippingMode::DISABLED:
        return "DISABLED";
    default:
        return "UNKNOWN";
    }
}

// Function to convert BatchType to string
std::string toString(BatchType value)
{
    switch (value)
    {
    case BatchType::Training:
        return "Training";
    case BatchType::Testing:
        return "Testing";
    default:
        return "UNKNOWN";
    }
}

// Function to convert ELRALES_Mode to string
std::string toString(ELRALES_Mode value)
{
    switch (value)
    {
    case ELRALES_Mode::ENABLED:
        return "ENABLED";
    case ELRALES_Mode::DISABLED:
        return "DISABLED";
    default:
        return "UNKNOWN";
    }
}

// Function to convert ELRALES_StateMachine to string
std::string toString(ELRALES_StateMachine value)
{
    switch (value)
    {
    case ELRALES_StateMachine::NORMAL:
        return "NORMAL";
    case ELRALES_StateMachine::RECOVERY:
        return "RECOVERY";
    case ELRALES_StateMachine::LOSING:
        return "LOSING";
    case ELRALES_StateMachine::DONE:
        return "DONE";
    case ELRALES_StateMachine::EARLY_STOPPING:
        return "EARLY_STOPPING";
    default:
        return "UNKNOWN";
    }
}

// Function to convert ELRALES_Retval to string
std::string toString(ELRALES_Retval value)
{
    switch (value)
    {
    case ELRALES_Retval::WASTED_EPOCH:
        return "WASTED_EPOCH";
    case ELRALES_Retval::SUCCESSFUL_EPOCH:
        return "SUCCESSFUL_EPOCH";
    case ELRALES_Retval::END_LEARNING:
        return "END_LEARNING";
    default:
        return "UNKNOWN";
    }
}

// Function to convert LogLevel to string
std::string toString(LogLevel value)
{
    switch (value)
    {
    case LogLevel::None:
        return "None";
    case LogLevel::LayerOutputs:
        return "LayerOutputs";
    case LogLevel::All:
        return "All";
    default:
        return "UNKNOWN";
    }
}

// Function to convert ProgressLevel to string
std::string toString(ProgressLevel value)
{
    switch (value)
    {
    case ProgressLevel::None:
        return "None";
    case ProgressLevel::Time:
        return "Time";
    case ProgressLevel::Progress:
        return "Progress";
    case ProgressLevel::ProgressTime:
        return "ProgressTime";
    default:
        return "UNKNOWN";
    }
}

// Function to convert AugmentTarget to string
std::string toString(AugmentTarget value)
{
    switch (value)
    {
    case AugmentTarget::TRAIN_DATASET:
        return "TRAIN_DATASET";
    case AugmentTarget::TEST_DATASET:
        return "TEST_DATASET";
    case AugmentTarget::WHOLE_DATASET:
        return "WHOLE_DATASET";
    case AugmentTarget::NONE:
        return "NONE";
    default:
        return "UNKNOWN";
    }
}

// Function to convert LearningDecayType to string
std::string toString(LearningDecayType value)
{
    switch (value)
    {
    case LearningDecayType::NONE:
        return "NONE";
    case LearningDecayType::EXPONENTIAL:
        return "EXPONENTIAL";
    case LearningDecayType::STEP:
        return "STEP";
    case LearningDecayType::POLYNOMIAL:
        return "POLYNOMIAL";
    case LearningDecayType::INVERSE_TIME:
        return "INVERSE_TIME";
    case LearningDecayType::COSINE:
        return "COSINE";
    default:
        return "UNKNOWN";
    }
}
