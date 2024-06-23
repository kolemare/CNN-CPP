#include "ConvolutionLayer.hpp"
#include <iostream>

ConvolutionLayer::ConvolutionLayer(int filters, int kernel_size, int input_depth, int stride, int padding, const Eigen::VectorXd &biases)
    : filters(filters), kernel_size(kernel_size), input_depth(input_depth), stride(stride), padding(padding), biases(biases)
{
    // Initialize filters with fixed values for testing
    kernels.resize(filters);
    for (int f = 0; f < filters; ++f)
    {
        kernels[f].resize(input_depth);
        for (int d = 0; d < input_depth; ++d)
        {
            kernels[f][d] = Eigen::MatrixXd::Constant(kernel_size, kernel_size, 1.0); // Using constant value for predictability
        }
    }
    if (biases.size() != filters)
    {
        this->biases = Eigen::VectorXd::Zero(filters); // Initialize biases to zero if size mismatch
    }
}

std::vector<Eigen::MatrixXd> ConvolutionLayer::forward(const std::vector<Eigen::MatrixXd> &input)
{
    int input_size = input[0].rows();
    int output_size = (input_size - kernel_size + 2 * padding) / stride + 1;
    std::vector<Eigen::MatrixXd> output(filters, Eigen::MatrixXd::Zero(output_size, output_size));

    for (int f = 0; f < filters; ++f)
    {
        Eigen::MatrixXd feature_map = Eigen::MatrixXd::Zero(output_size, output_size);

        for (int d = 0; d < input_depth; ++d)
        {
            Eigen::MatrixXd padded_input = padInput(input[d], padding);
            for (int i = 0; i < output_size; ++i)
            {
                for (int j = 0; j < output_size; ++j)
                {
                    int row_start = i * stride;
                    int col_start = j * stride;
                    double conv_sum = convolve(padded_input, kernels[f][d], row_start, col_start);
                    feature_map(i, j) += conv_sum;
                }
            }
        }
        feature_map.array() += biases(f);
        output[f] = feature_map;
    }
    return output;
}

void ConvolutionLayer::setBiases(const Eigen::VectorXd &new_biases)
{
    if (new_biases.size() == filters)
    {
        biases = new_biases;
    }
}

Eigen::MatrixXd ConvolutionLayer::padInput(const Eigen::MatrixXd &input, int pad)
{
    if (pad == 0)
        return input;

    int padded_size = input.rows() + 2 * pad;
    Eigen::MatrixXd padded_input = Eigen::MatrixXd::Zero(padded_size, padded_size);
    padded_input.block(pad, pad, input.rows(), input.cols()) = input;
    return padded_input;
}

double ConvolutionLayer::convolve(const Eigen::MatrixXd &input, const Eigen::MatrixXd &kernel, int start_row, int start_col)
{
    double sum = 0.0;
    for (int i = 0; i < kernel_size; ++i)
    {
        for (int j = 0; j < kernel_size; ++j)
        {
            sum += input(start_row + i, start_col + j) * kernel(i, j);
        }
    }
    return sum;
}
