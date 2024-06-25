#include "ConvolutionLayer.hpp"
#include <iostream>
#include <Eigen/Dense>

ConvolutionLayer::ConvolutionLayer(int filters, int kernel_size, int input_depth, int stride, int padding, const Eigen::VectorXd &biases)
    : filters(filters), kernel_size(kernel_size), input_depth(input_depth), stride(stride), padding(padding), biases(biases)
{
    kernels.resize(filters);
    for (int f = 0; f < filters; ++f)
    {
        kernels[f].resize(input_depth);
        for (int d = 0; d < input_depth; ++d)
        {
            kernels[f][d] = Eigen::MatrixXd::Random(kernel_size, kernel_size);
        }
    }
    if (biases.size() != filters)
    {
        std::cerr << "Warning: Mismatch in biases size, initializing to zero.\n";
        this->biases = Eigen::VectorXd::Zero(filters);
    }
}

std::vector<Eigen::MatrixXd> ConvolutionLayer::forward(const std::vector<Eigen::MatrixXd> &input)
{
    if (input.size() != input_depth)
    {
        throw std::invalid_argument("Input depth does not match the expected depth.");
    }

    int input_size = input[0].rows();
    if (input_size < kernel_size || (input_size - kernel_size + 2 * padding) / stride + 1 <= 0)
    {
        throw std::invalid_argument("Invalid input size relative to kernel size and stride.");
    }

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
    else
    {
        std::cerr << "Error: The size of new_biases must match the number of filters.\n";
    }
}

Eigen::VectorXd ConvolutionLayer::getBiases() const
{
    return biases;
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

void ConvolutionLayer::setKernels(const std::vector<std::vector<Eigen::MatrixXd>> &new_kernels)
{
    if (new_kernels.size() == filters && new_kernels[0].size() == input_depth)
    {
        kernels = new_kernels;
    }
    else
    {
        std::cerr << "Error: The size of new_kernels must match the number of filters and input depth." << std::endl;
    }
}
