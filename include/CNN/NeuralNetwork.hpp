#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "ConvolutionLayer.hpp"
#include "FlattenLayer.hpp"
#include "FullyConnectedLayer.hpp"
#include "ActivationLayer.hpp"
#include "MaxPoolingLayer.hpp"
#include "AveragePoolingLayer.hpp"
#include "LossFunction.hpp"
#include "BatchManager.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <random>

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
    void evaluate(const std::vector<Eigen::MatrixXd> &inputs, const std::vector<Eigen::MatrixXd> &labels);

private:
    std::vector<std::shared_ptr<Layer>> layers;
    std::vector<Eigen::MatrixXd> layerInputs;
    bool flattenAdded;
    std::unique_ptr<LossFunction> lossFunction;
};

#endif // NEURALNETWORK_HPP
