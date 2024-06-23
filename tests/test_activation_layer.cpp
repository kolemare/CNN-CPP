#include <gtest/gtest.h>
#include "ActivationLayer.hpp"
#include <Eigen/Dense>

class ActivationLayerTest : public ::testing::Test
{
protected:
    ActivationLayer *reluLayer;
    ActivationLayer *leakyReluLayer;
    ActivationLayer *sigmoidLayer;
    ActivationLayer *tanhLayer;
    ActivationLayer *softmaxLayer;
    ActivationLayer *eluLayer;

    Eigen::MatrixXd input;
    Eigen::MatrixXd expectedOutput;

    virtual void SetUp()
    {
        reluLayer = new ActivationLayer(ActivationLayer::RELU);
        leakyReluLayer = new ActivationLayer(ActivationLayer::LEAKY_RELU);
        sigmoidLayer = new ActivationLayer(ActivationLayer::SIGMOID);
        tanhLayer = new ActivationLayer(ActivationLayer::TANH);
        softmaxLayer = new ActivationLayer(ActivationLayer::SOFTMAX);
        eluLayer = new ActivationLayer(ActivationLayer::ELU);

        input.resize(2, 2);
    }

    virtual void TearDown()
    {
        delete reluLayer;
        delete leakyReluLayer;
        delete sigmoidLayer;
        delete tanhLayer;
        delete softmaxLayer;
        delete eluLayer;
    }
};

TEST_F(ActivationLayerTest, ReLUActivation)
{
    input << -1, 2, -3, 4;
    expectedOutput.resize(2, 2);
    expectedOutput << 0, 2, 0, 4;
    ASSERT_EQ(reluLayer->forward(input), expectedOutput);
}

TEST_F(ActivationLayerTest, LeakyReLUActivation)
{
    input << -1, 2, -3, 4;
    expectedOutput.resize(2, 2);
    expectedOutput << -0.01, 2, -0.03, 4;
    ASSERT_EQ(leakyReluLayer->forward(input), expectedOutput);
}

TEST_F(ActivationLayerTest, SigmoidActivation)
{
    input << 0, 2, -2, 4;
    expectedOutput.resize(2, 2);
    expectedOutput << 0.5, 1 / (1 + std::exp(-2)), 1 / (1 + std::exp(2)), 1 / (1 + std::exp(-4));
    ASSERT_TRUE((sigmoidLayer->forward(input) - expectedOutput).norm() < 1e-5);
}

TEST_F(ActivationLayerTest, TanhActivation)
{
    input << 0, 2, -2, 4;
    expectedOutput.resize(2, 2);
    expectedOutput << 0, std::tanh(2), std::tanh(-2), std::tanh(4);
    ASSERT_TRUE((tanhLayer->forward(input) - expectedOutput).norm() < 1e-5);
}

TEST_F(ActivationLayerTest, SoftmaxActivation)
{
    input << 1, 2, 3, 4;
    expectedOutput.resize(2, 2);
    expectedOutput << std::exp(1) / (std::exp(1) + std::exp(2)), std::exp(2) / (std::exp(1) + std::exp(2)),
        std::exp(3) / (std::exp(3) + std::exp(4)), std::exp(4) / (std::exp(3) + std::exp(4));
    ASSERT_TRUE((softmaxLayer->forward(input) - expectedOutput).norm() < 1e-5);
}

TEST_F(ActivationLayerTest, ELUActivation)
{
    input << -1, 2, -3, 4;
    expectedOutput.resize(2, 2);
    expectedOutput << std::exp(-1) - 1, 2, std::exp(-3) - 1, 4;
    ASSERT_TRUE((eluLayer->forward(input) - expectedOutput).norm() < 1e-5);
}