#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <vector>
#include <memory>
#include <Eigen/Dense>
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

enum PropagationType
{
    FORWARD,
    BACK
};

class NeuralNetwork
{
public:
    NeuralNetwork();

    void addConvolutionLayer(int filters, int kernel_size, int stride, int padding, ConvKernelInitialization kernel_init = ConvKernelInitialization::HE, ConvBiasInitialization bias_init = ConvBiasInitialization::ZERO);
    void addMaxPoolingLayer(int pool_size, int stride);
    void addAveragePoolingLayer(int pool_size, int stride);
    void addBatchNormalizationLayer(double epsilon = 1e-5, double momentum = 0.9);
    void addFlattenLayer();
    void addFullyConnectedLayer(int output_size, DenseWeightInitialization weight_init = DenseWeightInitialization::XAVIER, DenseBiasInitialization bias_init = DenseBiasInitialization::ZERO);
    void addActivationLayer(ActivationType type);
    void setLossFunction(LossType type);
    void setLogLevel(LogLevel level);
    void setProgressLevel(ProgressLevel level);

    void compile(std::unique_ptr<Optimizer> optimizer);
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input);
    void backward(const Eigen::MatrixXd &d_output, double learning_rate);
    void train(const ImageContainer &imageContainer, int epochs, double learning_rate, int batch_size, const std::vector<std::string> &categories);
    void evaluate(const ImageContainer &imageContainer, const std::vector<std::string> &categories);
    void setImageSize(const int targetWidth, const int targetHeight);

private:
    std::vector<std::shared_ptr<Layer>> layers;
    std::unique_ptr<LossFunction> lossFunction;
    std::unique_ptr<Optimizer> optimizer;
    std::vector<Eigen::MatrixXd> layerInputs;
    bool flattenAdded;
    int currentDepth;
    int inputSize;
    int inputHeight;
    int inputWidth;
    LogLevel logLevel;
    ProgressLevel progressLevel;

    void printMatrixSummary(const Eigen::MatrixXd &matrix, const std::string &layerType, PropagationType propagationType);
    void printFullMatrix(const Eigen::MatrixXd &matrix, const std::string &layerType, PropagationType propagationType);
    void printProgress(int epoch, int epochs, int batch, int totalBatches, std::chrono::steady_clock::time_point start, double batch_loss);
};

#endif // NEURALNETWORK_HPP
