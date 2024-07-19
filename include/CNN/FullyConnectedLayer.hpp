#ifndef FULLYCONNECTEDLAYER_HPP
#define FULLYCONNECTEDLAYER_HPP

#include "Layer.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <random>

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

class FullyConnectedLayer : public Layer
{
public:
    FullyConnectedLayer(int output_size, DenseWeightInitialization weight_init = DenseWeightInitialization::XAVIER,
                        DenseBiasInitialization bias_init = DenseBiasInitialization::ZERO, unsigned int seed = 42);

    Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4> &input_batch) override;
    Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4> &d_output_batch, const Eigen::Tensor<double, 4> &input_batch, double learning_rate) override;

    void setWeights(const Eigen::Tensor<double, 2> &new_weights);
    Eigen::Tensor<double, 2> getWeights() const;

    void setBiases(const Eigen::Tensor<double, 1> &new_biases);
    Eigen::Tensor<double, 1> getBiases() const;

    void setInputSize(int input_size);
    int getOutputSize() const;

    bool needsOptimizer() const override;
    void setOptimizer(std::shared_ptr<Optimizer> optimizer) override;

private:
    int input_size;
    int output_size;
    Eigen::Tensor<double, 2> weights;
    Eigen::Tensor<double, 1> biases;

    DenseWeightInitialization weight_init;
    DenseBiasInitialization bias_init;
    unsigned int seed;

    std::shared_ptr<Optimizer> optimizer;

    void initializeWeights();
    void initializeBiases();
};

#endif // FULLYCONNECTEDLAYER_HPP
