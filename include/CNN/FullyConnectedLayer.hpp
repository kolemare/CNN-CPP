#ifndef FULLYCONNECTEDLAYER_HPP
#define FULLYCONNECTEDLAYER_HPP

#include "Layer.hpp"
#include <Eigen/Dense>
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

    Eigen::MatrixXd forward(const Eigen::MatrixXd &input_batch) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd &d_output_batch, const Eigen::MatrixXd &input_batch, double learning_rate) override;

    void setWeights(const Eigen::MatrixXd &new_weights);
    Eigen::MatrixXd getWeights() const;

    void setBiases(const Eigen::VectorXd &new_biases);
    Eigen::VectorXd getBiases() const;

    void setInputSize(int input_size);
    int getOutputSize() const;

private:
    int input_size;
    int output_size;
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;

    DenseWeightInitialization weight_init;
    DenseBiasInitialization bias_init;
    unsigned int seed;

    void initializeWeights();
    void initializeBiases();
};

#endif // FULLYCONNECTEDLAYER_HPP
