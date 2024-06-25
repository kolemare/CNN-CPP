#include <gtest/gtest.h>
#include "ConvolutionLayer.hpp"
#include <Eigen/Dense>

class ConvolutionLayerTest : public ::testing::Test
{
protected:
    ConvolutionLayer *convLayer;
    std::vector<Eigen::MatrixXd> input;
    std::vector<Eigen::MatrixXd> expectedOutput;

    virtual void SetUp()
    {
        // Initialize a basic convolution layer for initial test
        Eigen::VectorXd biases = Eigen::VectorXd::Zero(1);
        convLayer = new ConvolutionLayer(1, 3, 1, 1, 1, biases);
        input.resize(1);
        input[0].resize(5, 5);
        input[0] << 1, 1, 1, 0, 0,
            0, 1, 1, 1, 0,
            0, 0, 1, 1, 1,
            0, 0, 1, 1, 0,
            0, 1, 1, 0, 0;

        expectedOutput.resize(1);
        expectedOutput[0].resize(5, 5);

        // Set specific kernel values for predictability
        std::vector<std::vector<Eigen::MatrixXd>> kernels = {{Eigen::MatrixXd::Constant(3, 3, 1.0)}};
        convLayer->setKernels(kernels);

        // Expected output calculated with constant kernel values
        expectedOutput[0] << 3, 5, 5, 3, 1,
            3, 6, 7, 6, 3,
            1, 4, 7, 7, 4,
            1, 4, 6, 6, 3,
            1, 3, 4, 3, 1;
    }

    virtual void
    TearDown()
    {
        delete convLayer;
    }
};

TEST_F(ConvolutionLayerTest, ForwardPass)
{
    std::vector<Eigen::MatrixXd> output = convLayer->forward(input);
    ASSERT_EQ(output.size(), expectedOutput.size());

    for (size_t i = 0; i < output.size(); ++i)
    {
        Eigen::MatrixXd diff = output[i] - expectedOutput[i];
        ASSERT_TRUE(diff.norm() < 1e-5) << "Difference at index " << i << ":\n"
                                        << diff;
    }
}

TEST_F(ConvolutionLayerTest, ForwardPass_Bias)
{
    Eigen::VectorXd biases(1);
    biases << 3;
    convLayer->setBiases(biases);

    expectedOutput[0] << 6, 8, 8, 6, 4,
        6, 9, 10, 9, 6,
        4, 7, 10, 10, 7,
        4, 7, 9, 9, 6,
        4, 6, 7, 6, 4;

    std::vector<Eigen::MatrixXd> output = convLayer->forward(input);
    ASSERT_EQ(output.size(), expectedOutput.size());

    for (size_t i = 0; i < output.size(); ++i)
    {
        Eigen::MatrixXd diff = output[i] - expectedOutput[i];
        ASSERT_TRUE(diff.norm() < 1e-5) << "Difference at index " << i << ":\n"
                                        << diff;
    }
}

TEST_F(ConvolutionLayerTest, DifferentKernelSize)
{
    delete convLayer;
    Eigen::VectorXd biases = Eigen::VectorXd::Zero(1);
    convLayer = new ConvolutionLayer(1, 2, 1, 1, 0, biases); // 2x2 kernel, no padding
    input[0] << 1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25;

    // Set specific kernel values for predictability
    std::vector<std::vector<Eigen::MatrixXd>> kernels = {{Eigen::MatrixXd::Constant(2, 2, 1.0)}};
    convLayer->setKernels(kernels);

    expectedOutput[0].resize(4, 4);
    expectedOutput[0] << 16, 20, 24, 28,
        36, 40, 44, 48,
        56, 60, 64, 68,
        76, 80, 84, 88;

    std::vector<Eigen::MatrixXd> output = convLayer->forward(input);
    ASSERT_EQ(output.size(), expectedOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        ASSERT_TRUE((output[i] - expectedOutput[i]).norm() < 1e-5);
    }
}

TEST_F(ConvolutionLayerTest, DifferentKernelSize_Bias)
{
    delete convLayer;
    Eigen::VectorXd biases(1);
    biases << 3;
    convLayer = new ConvolutionLayer(1, 2, 1, 1, 0, biases); // 2x2 kernel, no padding
    input[0] << 1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25;

    // Set specific kernel values for predictability
    std::vector<std::vector<Eigen::MatrixXd>> kernels = {{Eigen::MatrixXd::Constant(2, 2, 1.0)}};
    convLayer->setKernels(kernels);

    expectedOutput[0].resize(4, 4);
    expectedOutput[0] << 19, 23, 27, 31,
        39, 43, 47, 51,
        59, 63, 67, 71,
        79, 83, 87, 91;

    std::vector<Eigen::MatrixXd> output = convLayer->forward(input);
    ASSERT_EQ(output.size(), expectedOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        ASSERT_TRUE((output[i] - expectedOutput[i]).norm() < 1e-5);
    }
}

TEST_F(ConvolutionLayerTest, MultipleFilters)
{
    delete convLayer;
    Eigen::VectorXd biases = Eigen::VectorXd::Zero(2);
    convLayer = new ConvolutionLayer(2, 3, 1, 1, 1, biases); // 2 filters, 3x3 kernel, padding 1
    input[0] << 1, 1, 1, 0, 0,
        0, 1, 1, 1, 0,
        0, 0, 1, 1, 1,
        0, 0, 1, 1, 0,
        0, 1, 1, 0, 0;

    // Set specific kernel values for predictability
    std::vector<std::vector<Eigen::MatrixXd>> kernels = {{Eigen::MatrixXd::Constant(3, 3, 1.0)},
                                                         {Eigen::MatrixXd::Constant(3, 3, 1.0)}};
    convLayer->setKernels(kernels);

    expectedOutput.resize(2);
    expectedOutput[0].resize(5, 5);
    expectedOutput[0] << 3, 5, 5, 3, 1,
        3, 6, 7, 6, 3,
        1, 4, 7, 7, 4,
        1, 4, 6, 6, 3,
        1, 3, 4, 3, 1;
    expectedOutput[1] = expectedOutput[0]; // Same expected output for both filters

    std::vector<Eigen::MatrixXd> output = convLayer->forward(input);
    ASSERT_EQ(output.size(), expectedOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        ASSERT_TRUE((output[i] - expectedOutput[i]).norm() < 1e-5);
    }
}

TEST_F(ConvolutionLayerTest, MultipleFilters_Bias)
{
    delete convLayer;
    Eigen::VectorXd biases(2);
    biases << 3, 3;
    convLayer = new ConvolutionLayer(2, 3, 1, 1, 1, biases); // 2 filters, 3x3 kernel, padding 1
    input[0] << 1, 1, 1, 0, 0,
        0, 1, 1, 1, 0,
        0, 0, 1, 1, 1,
        0, 0, 1, 1, 0,
        0, 1, 1, 0, 0;

    // Set specific kernel values for predictability
    std::vector<std::vector<Eigen::MatrixXd>> kernels = {{Eigen::MatrixXd::Constant(3, 3, 1.0)},
                                                         {Eigen::MatrixXd::Constant(3, 3, 1.0)}};
    convLayer->setKernels(kernels);

    expectedOutput.resize(2);
    expectedOutput[0].resize(5, 5);
    expectedOutput[0] << 6, 8, 8, 6, 4,
        6, 9, 10, 9, 6,
        4, 7, 10, 10, 7,
        4, 7, 9, 9, 6,
        4, 6, 7, 6, 4;
    expectedOutput[1] = expectedOutput[0]; // Same expected output for both filters

    std::vector<Eigen::MatrixXd> output = convLayer->forward(input);
    ASSERT_EQ(output.size(), expectedOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        ASSERT_TRUE((output[i] - expectedOutput[i]).norm() < 1e-5);
    }
}

TEST_F(ConvolutionLayerTest, DifferentStride)
{
    delete convLayer;
    Eigen::VectorXd biases = Eigen::VectorXd::Zero(1);
    convLayer = new ConvolutionLayer(1, 3, 1, 2, 1, biases); // 3x3 kernel, stride 2, padding 1
    input[0] << 1, 1, 1, 0, 0,
        0, 1, 1, 1, 0,
        0, 0, 1, 1, 1,
        0, 0, 1, 1, 0,
        0, 1, 1, 0, 0;

    // Set specific kernel values for predictability
    std::vector<std::vector<Eigen::MatrixXd>> kernels = {{Eigen::MatrixXd::Constant(3, 3, 1.0)}};
    convLayer->setKernels(kernels);

    expectedOutput[0].resize(3, 3);
    expectedOutput[0] << 3, 5, 1,
        1, 7, 4,
        1, 4, 1;

    std::vector<Eigen::MatrixXd> output = convLayer->forward(input);
    ASSERT_EQ(output.size(), expectedOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        ASSERT_TRUE((output[i] - expectedOutput[i]).norm() < 1e-5);
    }
}

TEST_F(ConvolutionLayerTest, DifferentStride_Bias)
{
    delete convLayer;
    Eigen::VectorXd biases(1);
    biases << 3;
    convLayer = new ConvolutionLayer(1, 3, 1, 2, 1, biases); // 3x3 kernel, stride 2, padding 1
    input[0] << 1, 1, 1, 0, 0,
        0, 1, 1, 1, 0,
        0, 0, 1, 1, 1,
        0, 0, 1, 1, 0,
        0, 1, 1, 0, 0;

    // Set specific kernel values for predictability
    std::vector<std::vector<Eigen::MatrixXd>> kernels = {{Eigen::MatrixXd::Constant(3, 3, 1.0)}};
    convLayer->setKernels(kernels);

    expectedOutput[0].resize(3, 3);
    expectedOutput[0] << 6, 8, 4,
        4, 10, 7,
        4, 7, 4;

    std::vector<Eigen::MatrixXd> output = convLayer->forward(input);
    ASSERT_EQ(output.size(), expectedOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        ASSERT_TRUE((output[i] - expectedOutput[i]).norm() < 1e-5);
    }
}

TEST_F(ConvolutionLayerTest, MultipleInputDepth)
{
    delete convLayer;
    Eigen::VectorXd biases = Eigen::VectorXd::Zero(1);
    convLayer = new ConvolutionLayer(1, 3, 2, 1, 1, biases); // 3x3 kernel, 1 filter, 2 input depth channels, padding 1
    input.resize(2);
    input[0].resize(5, 5);
    input[1].resize(5, 5);
    input[0] << 1, 1, 1, 0, 0,
        0, 1, 1, 1, 0,
        0, 0, 1, 1, 1,
        0, 0, 1, 1, 0,
        0, 1, 1, 0, 0;
    input[1] << 1, 1, 1, 0, 0,
        0, 1, 1, 1, 0,
        0, 0, 1, 1, 1,
        0, 0, 1, 1, 0,
        0, 1, 1, 0, 0;

    // Set specific kernel values for predictability
    std::vector<std::vector<Eigen::MatrixXd>> kernels = {{Eigen::MatrixXd::Constant(3, 3, 1.0), Eigen::MatrixXd::Constant(3, 3, 1.0)}};
    convLayer->setKernels(kernels);

    expectedOutput[0].resize(5, 5);
    expectedOutput[0] << 6, 10, 10, 6, 2,
        6, 12, 14, 12, 6,
        2, 8, 14, 14, 8,
        2, 8, 12, 12, 6,
        2, 6, 8, 6, 2;

    std::vector<Eigen::MatrixXd> output = convLayer->forward(input);
    ASSERT_EQ(output.size(), expectedOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        ASSERT_TRUE((output[i] - expectedOutput[i]).norm() < 1e-5);
    }
}

TEST_F(ConvolutionLayerTest, MultipleInputDepth_Bias)
{
    delete convLayer;
    Eigen::VectorXd biases(1);
    biases << 3;
    convLayer = new ConvolutionLayer(1, 3, 2, 1, 1, biases); // 3x3 kernel, 1 filter, 2 input depth channels, padding 1
    input.resize(2);
    input[0].resize(5, 5);
    input[1].resize(5, 5);
    input[0] << 1, 1, 1, 0, 0,
        0, 1, 1, 1, 0,
        0, 0, 1, 1, 1,
        0, 0, 1, 1, 0,
        0, 1, 1, 0, 0;
    input[1] << 1, 1, 1, 0, 0,
        0, 1, 1, 1, 0,
        0, 0, 1, 1, 1,
        0, 0, 1, 1, 0,
        0, 1, 1, 0, 0;

    // Set specific kernel values for predictability
    std::vector<std::vector<Eigen::MatrixXd>> kernels = {{Eigen::MatrixXd::Constant(3, 3, 1.0), Eigen::MatrixXd::Constant(3, 3, 1.0)}};
    convLayer->setKernels(kernels);

    expectedOutput[0].resize(5, 5);
    expectedOutput[0] << 9, 13, 13, 9, 5,
        9, 15, 17, 15, 9,
        5, 11, 17, 17, 11,
        5, 11, 15, 15, 9,
        5, 9, 11, 9, 5;

    std::vector<Eigen::MatrixXd> output = convLayer->forward(input);
    ASSERT_EQ(output.size(), expectedOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        ASSERT_TRUE((output[i] - expectedOutput[i]).norm() < 1e-5);
    }
}

TEST_F(ConvolutionLayerTest, ZeroPadding)
{
    delete convLayer;
    Eigen::VectorXd biases = Eigen::VectorXd::Zero(1);
    convLayer = new ConvolutionLayer(1, 3, 1, 1, 0, biases); // 3x3 kernel, no padding
    input[0].resize(5, 5);
    input[0] << 1, 1, 1, 0, 0,
        0, 1, 1, 1, 0,
        0, 0, 1, 1, 1,
        0, 0, 1, 1, 0,
        0, 1, 1, 0, 0;

    // Set specific kernel values for predictability
    std::vector<std::vector<Eigen::MatrixXd>> kernels = {{Eigen::MatrixXd::Constant(3, 3, 1.0)}};
    convLayer->setKernels(kernels);

    expectedOutput[0].resize(3, 3);
    expectedOutput[0] << 6, 7, 6,
        4, 7, 7,
        4, 6, 6;

    std::vector<Eigen::MatrixXd> output = convLayer->forward(input);
    ASSERT_EQ(output.size(), expectedOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        ASSERT_TRUE((output[i] - expectedOutput[i]).norm() < 1e-5) << "Difference at index " << i << ":\n"
                                                                   << output[i] << "\nExpected:\n"
                                                                   << expectedOutput[i];
    }
}

TEST_F(ConvolutionLayerTest, LargeStride)
{
    delete convLayer;
    Eigen::VectorXd biases = Eigen::VectorXd::Zero(1);
    convLayer = new ConvolutionLayer(1, 3, 1, 4, 1, biases); // 3x3 kernel, stride 4, padding 1
    input[0].resize(5, 5);
    input[0] << 1, 1, 1, 0, 0,
        0, 1, 1, 1, 0,
        0, 0, 1, 1, 1,
        0, 0, 1, 1, 0,
        0, 1, 1, 0, 0;

    std::vector<std::vector<Eigen::MatrixXd>> kernels = {{Eigen::MatrixXd::Constant(3, 3, 1.0)}};
    convLayer->setKernels(kernels);

    expectedOutput[0].resize(2, 2);
    expectedOutput[0] << 3, 1,
        1, 1;

    std::vector<Eigen::MatrixXd> output = convLayer->forward(input);
    ASSERT_EQ(output.size(), expectedOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        ASSERT_TRUE((output[i] - expectedOutput[i]).norm() < 1e-5);
    }
}

TEST_F(ConvolutionLayerTest, LargeInputSize)
{
    delete convLayer;
    Eigen::VectorXd biases = Eigen::VectorXd::Zero(1);
    convLayer = new ConvolutionLayer(1, 3, 1, 1, 1, biases); // 3x3 kernel, padding 1
    input[0].resize(10, 10);
    input[0].setRandom(); // Random values for large input

    std::vector<std::vector<Eigen::MatrixXd>> kernels = {{Eigen::MatrixXd::Constant(3, 3, 1.0)}};
    convLayer->setKernels(kernels);

    int output_size = (10 - 3 + 2 * 1) / 1 + 1;
    expectedOutput[0].resize(output_size, output_size);
    expectedOutput[0].setZero(); // Expected output is not known, we just want to ensure it runs

    std::vector<Eigen::MatrixXd> output = convLayer->forward(input);
    ASSERT_EQ(output.size(), expectedOutput.size());
}