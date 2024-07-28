#ifndef ELRAL_HPP
#define ELRAL_HPP

#include <vector>
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>
#include "Layer.hpp"
#include "Optimizer.hpp"
#include "ConvolutionLayer.hpp"
#include "FullyConnectedLayer.hpp"

class ELRAL
{
public:
    ELRAL(double learning_rate_coef, int maxSuccessiveFailures, int maxFails, double tolerance, const std::vector<std::shared_ptr<Layer>> &layers);

    bool updateState(double current_loss, std::vector<std::shared_ptr<Layer>> &layers, double &learning_rate, ELRALMode &mode);

private:
    double learning_rate_coef;
    int maxSuccessiveEpochFailures;
    int maxEpochFails;
    double tolerance;

    double best_loss;
    int successiveEpochFailures;
    int totalEpochFailures;

    struct OptimizerState
    {
        Eigen::Tensor<double, 2> v_weights_2d;
        Eigen::Tensor<double, 1> v_biases_2d;
        Eigen::Tensor<double, 4> v_weights_4d;
        Eigen::Tensor<double, 1> v_biases_4d;

        Eigen::Tensor<double, 2> m_weights_2d;
        Eigen::Tensor<double, 1> m_biases_2d;
        Eigen::Tensor<double, 4> m_weights_4d;
        Eigen::Tensor<double, 1> m_biases_4d;

        Eigen::Tensor<double, 2> s_weights_2d;
        Eigen::Tensor<double, 1> s_biases_2d;
        Eigen::Tensor<double, 4> s_weights_4d;
        Eigen::Tensor<double, 1> s_biases_4d;
    };

    struct ConvolutionLayerState
    {
        Eigen::Tensor<double, 4> kernels;
        Eigen::Tensor<double, 1> biases;
        OptimizerState optimizer_state;
    };

    struct FullyConnectedLayerState
    {
        Eigen::Tensor<double, 2> weights;
        Eigen::Tensor<double, 1> biases;
        OptimizerState optimizer_state;
    };

    std::vector<ConvolutionLayerState> savedConvLayerStates;
    std::vector<FullyConnectedLayerState> savedFCLayerStates;

    void saveState(const std::vector<std::shared_ptr<Layer>> &layers);
    void restoreState(std::vector<std::shared_ptr<Layer>> &layers);
};

#endif // ELRAL_HPP
