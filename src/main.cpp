#include <iostream>
#include <opencv2/opencv.hpp>
#include "convolution_layer.hpp"
#include <gtest/gtest.h>

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    if (argc > 1 && std::string(argv[1]) == "--tests")
    {
        return RUN_ALL_TESTS();
    }
    else
    {
        // Example usage of ConvolutionLayer
        ConvolutionLayer conv(32, 3);
        cv::Mat image = cv::imread("path_to_image.jpg", cv::IMREAD_GRAYSCALE);
        Eigen::MatrixXd input = Eigen::Map<Eigen::MatrixXd>(image.ptr<double>(), image.rows, image.cols);

        Eigen::MatrixXd output = conv.forward(input);
        std::cout << "Output shape: " << output.rows() << " x " << output.cols() << std::endl;
        return 0;
    }
}
