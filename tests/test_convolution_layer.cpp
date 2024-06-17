#include <gtest/gtest.h>
#include "convolution_layer.hpp"

class ConvolutionLayerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Code here will be called immediately after the constructor (right before each test).
        conv = new ConvolutionLayer(32, 3);
    }

    void TearDown() override
    {
        // Code here will be called immediately after each test (right before the destructor).
        delete conv;
    }

    // Objects declared here can be used by all tests in the test suite for ConvolutionLayerTest.
    ConvolutionLayer *conv;
};

TEST_F(ConvolutionLayerTest, ForwardPass)
{
    Eigen::MatrixXd input(150, 150);
    input.setRandom();
    Eigen::MatrixXd output = conv->forward(input);
    ASSERT_EQ(output.rows(), 0); // Assuming a kernel size of 3 and stride of 1
    ASSERT_EQ(output.cols(), 0);
}
