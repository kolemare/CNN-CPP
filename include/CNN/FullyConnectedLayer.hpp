#ifndef FULLYCONNECTEDLAYER_HPP
#define FULLYCONNECTEDLAYER_HPP

#include "Layer.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <random>

class FullyConnectedLayer : public Layer
{
public:
    // Constructor
    FullyConnectedLayer(int output_size, DenseWeightInitialization weight_init = DenseWeightInitialization::XAVIER,
                        DenseBiasInitialization bias_init = DenseBiasInitialization::ZERO, unsigned int seed = 42);

    // Forward and backward pass functions
    Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4> &input_batch) override;
    Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4> &d_output_batch, const Eigen::Tensor<double, 4> &input_batch, double learning_rate) override;

    // Functions to set and get weights and biases
    void setWeights(const Eigen::Tensor<double, 2> &new_weights);
    Eigen::Tensor<double, 2> getWeights() const;

    void setBiases(const Eigen::Tensor<double, 1> &new_biases);
    Eigen::Tensor<double, 1> getBiases() const;

    void setInputSize(int input_size);
    int getOutputSize() const;

    // Functions to manage optimizer
    bool needsOptimizer() const override;
    void setOptimizer(std::shared_ptr<Optimizer> optimizer) override;
    std::shared_ptr<Optimizer> getOptimizer() override;

private:
    int input_size;                   // Input size of the layer
    int output_size;                  // Output size of the layer
    Eigen::Tensor<double, 2> weights; // Weights of the layer
    Eigen::Tensor<double, 1> biases;  // Biases of the layer

    DenseWeightInitialization weight_init; // Weight initialization method
    DenseBiasInitialization bias_init;     // Bias initialization method
    unsigned int seed;                     // Seed for random number generation

    std::shared_ptr<Optimizer> optimizer; // Optimizer

    // Functions to initialize weights and biases
    void initializeWeights();
    void initializeBiases();
};

#endif // FULLYCONNECTEDLAYER_HPP
