#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "Layer.hpp"
#include "Optimizer.hpp"
#include "ActivationLayer.hpp"
#include <vector>
#include <memory>
#include <algorithm>

class NeuralNetwork
{
public:
    NeuralNetwork();

    void addConvolutionLayer(int filters, int kernel_size, int input_depth, int stride, int padding, const Eigen::VectorXd &biases);
    void addFlattenLayer();
    void addFullyConnectedLayer(int input_size, int output_size, std::unique_ptr<Optimizer> optimizer);
    void addActivationLayer(ActivationType type);

    Eigen::MatrixXd forward(const Eigen::MatrixXd &input);
    void backward(const Eigen::MatrixXd &d_output, double learning_rate);
    void train(const std::vector<Eigen::MatrixXd> &inputs, const std::vector<Eigen::MatrixXd> &labels, int epochs, double learning_rate, int batch_size);

private:
    std::vector<std::shared_ptr<Layer>> layers;
    std::vector<Eigen::MatrixXd> layerInputs;
    bool flattenAdded;
};

#endif // NEURALNETWORK_HPP