#include "ConvolutionLayer.hpp"
#include "MaxPoolingLayer.hpp"
#include <iostream>
#include <stdexcept>
#include <cmath>

ConvolutionLayer::ConvolutionLayer(int filters, int kernel_size, int input_depth, int stride, int padding, const Eigen::VectorXd &biases)
    : filters(filters), kernel_size(kernel_size), input_depth(input_depth), stride(stride), padding(padding), biases(biases)
{
    kernels.resize(filters);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0, std::sqrt(2.0 / (input_depth * kernel_size * kernel_size)));

    for (int f = 0; f < filters; ++f)
    {
        kernels[f].resize(input_depth);
        for (int d = 0; d < input_depth; ++d)
        {
            Eigen::MatrixXd kernel(kernel_size, kernel_size);
            for (int i = 0; i < kernel_size; ++i)
            {
                for (int j = 0; j < kernel_size; ++j)
                {
                    kernel(i, j) = dis(gen);
                }
            }
            kernels[f][d] = kernel;
        }
    }
    if (biases.size() != filters)
    {
        std::cerr << "Warning: Mismatch in biases size, initializing to zero.\n";
        this->biases = Eigen::VectorXd::Zero(filters);
    }
}

Eigen::MatrixXd ConvolutionLayer::forward(const Eigen::MatrixXd &input_batch)
{
    int batch_size = input_batch.rows();
    int input_size = std::sqrt(input_batch.cols() / input_depth);
    if (input_size * input_size * input_depth != input_batch.cols())
    {
        throw std::invalid_argument("Input dimensions do not match the expected depth and size.");
    }

    int output_size = (input_size - kernel_size + 2 * padding) / stride + 1;
    Eigen::MatrixXd output_batch(batch_size, filters * output_size * output_size);

    for (int b = 0; b < batch_size; ++b)
    {
        Eigen::Map<const Eigen::MatrixXd> input(input_batch.row(b).data(), input_size, input_size * input_depth);

        for (int f = 0; f < filters; ++f)
        {
            Eigen::MatrixXd feature_map = Eigen::MatrixXd::Zero(output_size, output_size);

            for (int d = 0; d < input_depth; ++d)
            {
                Eigen::MatrixXd input_slice = input.middleCols(d * input_size, input_size);
                Eigen::MatrixXd padded_input = padInput(input_slice, padding);

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

            // Apply biases
            feature_map.array() += biases(f);

            // Copy feature_map to output_batch
            output_batch.block(b, f * output_size * output_size, 1, output_size * output_size) = Eigen::Map<Eigen::RowVectorXd>(feature_map.data(), feature_map.size());
        }
    }

    // Update the static variables in MaxPoolingLayer
    MaxPoolingLayer::setInputSize(output_size);
    MaxPoolingLayer::setInputDepth(filters);

    return output_batch;
}

Eigen::MatrixXd ConvolutionLayer::backward(const Eigen::MatrixXd &d_output_batch, const Eigen::MatrixXd &input_batch, double learning_rate)
{
    int batch_size = input_batch.rows();
    int input_size = std::sqrt(input_batch.cols() / input_depth);
    int output_size = (input_size - kernel_size + 2 * padding) / stride + 1;

    std::vector<std::vector<Eigen::MatrixXd>> d_kernels(filters, std::vector<Eigen::MatrixXd>(input_depth, Eigen::MatrixXd::Zero(kernel_size, kernel_size)));
    Eigen::VectorXd d_biases = Eigen::VectorXd::Zero(filters);
    Eigen::MatrixXd d_input_batch = Eigen::MatrixXd::Zero(batch_size, input_size * input_size * input_depth);

    for (int b = 0; b < batch_size; ++b)
    {
        Eigen::Map<const Eigen::MatrixXd> input(input_batch.row(b).data(), input_size, input_size * input_depth);
        Eigen::Map<const Eigen::MatrixXd> d_output(d_output_batch.row(b).data(), filters, output_size * output_size);

        for (int f = 0; f < filters; ++f)
        {
            Eigen::MatrixXd d_output_reshaped = Eigen::Map<const Eigen::MatrixXd>(d_output.row(f).data(), output_size, output_size);

            for (int d = 0; d < input_depth; ++d)
            {
                Eigen::MatrixXd input_slice = input.middleCols(d * input_size, input_size);
                Eigen::MatrixXd padded_input = padInput(input_slice, padding);
                Eigen::MatrixXd d_input_slice = Eigen::MatrixXd::Zero(input_size, input_size);

                for (int i = 0; i < output_size; ++i)
                {
                    for (int j = 0; j < output_size; ++j)
                    {
                        int row_start = i * stride;
                        int col_start = j * stride;
                        for (int k = 0; k < kernel_size; ++k)
                        {
                            for (int l = 0; l < kernel_size; ++l)
                            {
                                d_kernels[f][d](k, l) += d_output_reshaped(i, j) * padded_input(row_start + k, col_start + l);
                                d_input_slice(row_start + k, col_start + l) += d_output_reshaped(i, j) * kernels[f][d](k, l);
                            }
                        }
                    }
                }
                Eigen::Map<Eigen::RowVectorXd>(d_input_batch.row(b).data() + d * input_size * input_size, input_size * input_size) = Eigen::Map<Eigen::RowVectorXd>(d_input_slice.data(), d_input_slice.size());
            }
            d_biases(f) += d_output_reshaped.sum();
        }
    }

    for (int f = 0; f < filters; ++f)
    {
        for (int d = 0; d < input_depth; ++d)
        {
            kernels[f][d] -= learning_rate * d_kernels[f][d] / batch_size;
        }
        biases(f) -= learning_rate * d_biases(f) / batch_size;
    }

    return d_input_batch;
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
