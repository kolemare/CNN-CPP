#ifndef FULLYCONNECTEDLAYER_HPP
#define FULLYCONNECTEDLAYER_HPP

#include "Layer.hpp"
#include "Optimizer.hpp"
#include <Eigen/Dense>
#include <memory>

class FullyConnectedLayer : public Layer
{
public:
    FullyConnectedLayer(int input_size, int output_size, std::unique_ptr<Optimizer> optimizer, unsigned int seed = 0);

    Eigen::MatrixXd forward(const Eigen::MatrixXd &input_batch) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd &d_output_batch, const Eigen::MatrixXd &input_batch, double learning_rate) override;

    void setWeights(const Eigen::MatrixXd &new_weights);
    Eigen::MatrixXd getWeights() const;

    void setBiases(const Eigen::VectorXd &new_biases);
    Eigen::VectorXd getBiases() const;

private:
    int input_size;
    int output_size;
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;
    std::unique_ptr<Optimizer> optimizer;
};

#endif // FULLYCONNECTEDLAYER_HPP
