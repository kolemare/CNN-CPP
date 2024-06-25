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
        reluLayer = new ActivationLayer(ActivationType::RELU);
        leakyReluLayer = new ActivationLayer(ActivationType::LEAKY_RELU);
        sigmoidLayer = new ActivationLayer(ActivationType::SIGMOID);
        tanhLayer = new ActivationLayer(ActivationType::TANH);
        softmaxLayer = new ActivationLayer(ActivationType::SOFTMAX);
        eluLayer = new ActivationLayer(ActivationType::ELU);

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

TEST_F(ActivationLayerTest, LeakyReLUActivationDefaultAlpha)
{
    input << -1, 2, -3, 4;
    expectedOutput.resize(2, 2);
    expectedOutput << -0.01, 2, -0.03, 4;
    ASSERT_EQ(leakyReluLayer->forward(input), expectedOutput);
}

TEST_F(ActivationLayerTest, LeakyReLUActivationCustomAlpha)
{
    input << -1, 2, -3, 4;
    expectedOutput.resize(2, 2);
    expectedOutput << -0.1, 2, -0.3, 4;
    leakyReluLayer->setAlpha(0.1);
    Eigen::MatrixXd output = leakyReluLayer->forward(input);
    ASSERT_TRUE((output - expectedOutput).norm() < 1e-5) << "Difference:\n"
                                                         << (output - expectedOutput);
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
    double exp1 = std::exp(1);
    double exp2 = std::exp(2);
    double exp3 = std::exp(3);
    double exp4 = std::exp(4);
    expectedOutput << exp1 / (exp1 + exp2), exp2 / (exp1 + exp2),
        exp3 / (exp3 + exp4), exp4 / (exp3 + exp4);
    ASSERT_TRUE((softmaxLayer->forward(input) - expectedOutput).norm() < 1e-5);
}

TEST_F(ActivationLayerTest, ELUActivationDefaultAlpha)
{
    input << -1, 2, -3, 4;
    expectedOutput.resize(2, 2);
    expectedOutput << std::exp(-1) - 1, 2, std::exp(-3) - 1, 4;
    ASSERT_TRUE((eluLayer->forward(input) - expectedOutput).norm() < 1e-5);
}

TEST_F(ActivationLayerTest, ELUActivationCustomAlpha)
{
    input << -1, 2, -3, 4;
    expectedOutput.resize(2, 2);
    expectedOutput << 2 * (std::exp(-1) - 1), 2, 2 * (std::exp(-3) - 1), 4;
    eluLayer->setAlpha(2.0);
    ASSERT_TRUE((eluLayer->forward(input) - expectedOutput).norm() < 1e-5);
}

TEST_F(ActivationLayerTest, GetAlpha)
{
    ASSERT_EQ(leakyReluLayer->getAlpha(), 0.01);
    ASSERT_EQ(eluLayer->getAlpha(), 1.0);

    leakyReluLayer->setAlpha(0.1);
    ASSERT_EQ(leakyReluLayer->getAlpha(), 0.1);

    eluLayer->setAlpha(2.0);
    ASSERT_EQ(eluLayer->getAlpha(), 2.0);
}