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

enum PropagationType
{
    FORWARD,
    BACK
};

class NeuralNetwork
{
public:
    NeuralNetwork();

    void addConvolutionLayer(int filters, int kernel_size, int input_depth, int stride, int padding, const Eigen::VectorXd &biases);
    void addMaxPoolingLayer(int pool_size, int stride);
    void addAveragePoolingLayer(int pool_size, int stride);
    void addFlattenLayer();
    void addFullyConnectedLayer(int input_size, int output_size, std::unique_ptr<Optimizer> optimizer);
    void addActivationLayer(ActivationType type);
    void setLossFunction(LossType type);

    Eigen::MatrixXd forward(const Eigen::MatrixXd &input);
    void backward(const Eigen::MatrixXd &d_output, double learning_rate);
    void train(const ImageContainer &imageContainer, int epochs, double learning_rate, int batch_size, const std::vector<std::string> &categories);
    void evaluate(const ImageContainer &imageContainer, const std::vector<std::string> &categories);

private:
    std::vector<std::shared_ptr<Layer>> layers;
    std::unique_ptr<LossFunction> lossFunction;
    std::vector<Eigen::MatrixXd> layerInputs;
    bool flattenAdded;
    int current_depth;
    int input_size; // Assuming square input dimensions, can be modified to handle non-square inputs

    void printMatrixSummary(const Eigen::MatrixXd &matrix, const std::string &layerType, PropagationType propagationType);
};

#endif
