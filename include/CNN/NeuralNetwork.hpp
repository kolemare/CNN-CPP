#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <vector>
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>
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
#include "GradientClipping.hpp"
#include "NNLogger.hpp"

class NeuralNetwork
{
public:
    NeuralNetwork();

    void addConvolutionLayer(int filters, int kernel_size, int stride, int padding, ConvKernelInitialization kernel_init = ConvKernelInitialization::HE, ConvBiasInitialization bias_init = ConvBiasInitialization::ZERO);
    void addMaxPoolingLayer(int pool_size, int stride);
    void addAveragePoolingLayer(int pool_size, int stride);
    void addFlattenLayer();
    void addFullyConnectedLayer(int output_size, DenseWeightInitialization weight_init = DenseWeightInitialization::XAVIER, DenseBiasInitialization bias_init = DenseBiasInitialization::ZERO);
    void addActivationLayer(ActivationType type);
    void setLossFunction(LossType type);
    void setLogLevel(LogLevel level);
    void setProgressLevel(ProgressLevel level);
    void setGradientClipping(GradientClippingMode clipping = GradientClippingMode::ENABLED, double clipValue = 1.0);

    void compile(OptimizerType optimizerType, const std::unordered_map<std::string, double> &optimizer_params = {});
    Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4> &input);
    void backward(const Eigen::Tensor<double, 4> &d_output, double learning_rate);
    void train(const ImageContainer &imageContainer, int epochs, int batch_size, double learning_rate = 0.001);
    void evaluate(const ImageContainer &imageContainer);
    void setImageSize(const int targetWidth, const int targetHeight);

private:
    std::vector<std::shared_ptr<Layer>> layers;
    std::unique_ptr<LossFunction> lossFunction;
    std::shared_ptr<Optimizer> optimizer;
    std::vector<Eigen::Tensor<double, 4>> layerInputs;
    bool flattenAdded;
    bool clippingSet;
    int currentDepth;
    int inputSize;
    int inputHeight;
    int inputWidth;
    LogLevel logLevel;
    ProgressLevel progressLevel;
    GradientClippingMode clippingMode;
    double clipValue;
};

#endif // NEURALNETWORK_HPP
