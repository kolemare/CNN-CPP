#include <gtest/gtest.h>
#include "FullyConnectedLayer.hpp"
#include "Optimizer.hpp"
#include <Eigen/Dense>
#include <stdexcept>

class FullyConnectedLayerTest : public ::testing::Test
{
protected:
    FullyConnectedLayer *layer;
    Eigen::MatrixXd input;
    Eigen::MatrixXd expectedOutput;
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;
    std::unique_ptr<Optimizer> optimizer;

    virtual void SetUp()
    {
        // Set up optimizer
        optimizer = Optimizer::create(Optimizer::Type::SGD);

        // Create layer with optimizer and seed for reproducibility
        layer = new FullyConnectedLayer(3, 2, std::move(optimizer), 42);

        // Initialize input
        input.resize(2, 3);
        input << 1, 2, 3,
            4, 5, 6;

        // Initialize weights and biases
        weights.resize(2, 3);
        weights << 1, 0, -1,
            0, 1, -1;

        biases.resize(2);
        biases << 1, -1;

        // Setting weights and biases
        layer->setWeights(weights);
        layer->setBiases(biases);

        // Expected output based on input, weights, and biases
        expectedOutput.resize(2, 2);
        expectedOutput << -1, -2,
            -1, -2;
    }

    virtual void TearDown()
    {
        delete layer;
    }
};

TEST_F(FullyConnectedLayerTest, ForwardPass)
{
    Eigen::MatrixXd output = layer->forward(input);
    ASSERT_EQ(output.rows(), expectedOutput.rows());
    ASSERT_EQ(output.cols(), expectedOutput.cols());
    ASSERT_TRUE(output.isApprox(expectedOutput, 1e-5)) << "Output:\n"
                                                       << output << "\nExpected:\n"
                                                       << expectedOutput;
}

TEST_F(FullyConnectedLayerTest, SetGetWeights)
{
    layer->setWeights(weights);
    Eigen::MatrixXd retrievedWeights = layer->getWeights();
    ASSERT_EQ(retrievedWeights.rows(), weights.rows());
    ASSERT_EQ(retrievedWeights.cols(), weights.cols());
    ASSERT_TRUE(retrievedWeights.isApprox(weights, 1e-5));
}

TEST_F(FullyConnectedLayerTest, SetGetBiases)
{
    layer->setBiases(biases);
    Eigen::VectorXd retrievedBiases = layer->getBiases();
    ASSERT_EQ(retrievedBiases.size(), biases.size());
    ASSERT_TRUE(retrievedBiases.isApprox(biases, 1e-5));
}

TEST_F(FullyConnectedLayerTest, InvalidInputSize)
{
    Eigen::MatrixXd invalidInput(2, 4);
    ASSERT_THROW(layer->forward(invalidInput), std::invalid_argument);
}

TEST_F(FullyConnectedLayerTest, InvalidWeightsSize)
{
    Eigen::MatrixXd invalidWeights(3, 4);
    ASSERT_THROW(layer->setWeights(invalidWeights), std::invalid_argument);
}

TEST_F(FullyConnectedLayerTest, InvalidBiasesSize)
{
    Eigen::VectorXd invalidBiases(3);
    ASSERT_THROW(layer->setBiases(invalidBiases), std::invalid_argument);
}

TEST_F(FullyConnectedLayerTest, ConstructorInvalidSize)
{
    std::unique_ptr<Optimizer> optimizer = Optimizer::create(Optimizer::Type::SGD);
    ASSERT_THROW(FullyConnectedLayer(-1, 3, std::move(optimizer), 42), std::invalid_argument);
    optimizer = Optimizer::create(Optimizer::Type::SGD);
    ASSERT_THROW(FullyConnectedLayer(3, -1, std::move(optimizer), 42), std::invalid_argument);
}

TEST_F(FullyConnectedLayerTest, BackwardPass)
{
    Eigen::MatrixXd d_output(2, 2);
    d_output << 1, 2,
        3, 4;
    Eigen::MatrixXd expected_d_input = d_output * weights;
    Eigen::MatrixXd d_input = layer->backward(d_output, input, 0.01);
    ASSERT_EQ(d_input.rows(), input.rows());
    ASSERT_EQ(d_input.cols(), input.cols());
    ASSERT_TRUE(d_input.isApprox(expected_d_input, 1e-5)) << "d_input:\n"
                                                          << d_input << "\nExpected:\n"
                                                          << expected_d_input;
}

TEST_F(FullyConnectedLayerTest, UpdateWeightsAndBiases)
{
    Eigen::MatrixXd d_output(2, 2);
    d_output << 1, 2,
        3, 4;
    layer->backward(d_output, input, 0.01);

    Eigen::MatrixXd new_weights = layer->getWeights();
    Eigen::VectorXd new_biases = layer->getBiases();

    ASSERT_FALSE(new_weights.isApprox(weights, 1e-5)) << "Weights did not update correctly.";
    ASSERT_FALSE(new_biases.isApprox(biases, 1e-5)) << "Biases did not update correctly.";
}
