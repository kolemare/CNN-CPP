#include "FlattenLayer.hpp"
#include <iostream>
#include <stdexcept>

Eigen::MatrixXd FlattenLayer::forward(const Eigen::MatrixXd &input)
{
    batch_size = input.rows();
    original_size = input.cols();

    throw std::runtime_error("END OF DEBUGGING");

    // Add debugging output
    std::cout << "Flatten Layer forward pass input dimensions: " << input.rows() << "x" << input.cols() << std::endl;
    std::cout << "Flatten Layer forward pass input: " << input << std::endl;

    Eigen::MatrixXd output = Eigen::Map<const Eigen::MatrixXd>(input.data(), batch_size, original_size);

    std::cout << "Flatten Layer forward pass output dimensions: " << output.rows() << "x" << output.cols() << std::endl;
    std::cout << "Flatten Layer forward pass output: " << output << std::endl;

    return output;
}

Eigen::MatrixXd FlattenLayer::backward(const Eigen::MatrixXd &d_output, const Eigen::MatrixXd &input, double learning_rate)
{
    return Eigen::Map<const Eigen::MatrixXd>(d_output.data(), batch_size, original_size);
}
