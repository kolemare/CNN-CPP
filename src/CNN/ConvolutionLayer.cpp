#include "ConvolutionLayer.hpp"
#include "MaxPoolingLayer.hpp"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <future>
#include <mutex>

// Constructor with specified kernel and bias initializations
ConvolutionLayer::ConvolutionLayer(int filters, int kernel_size, int stride, int padding, ConvKernelInitialization kernel_init, ConvBiasInitialization bias_init)
    : filters(filters), kernel_size(kernel_size), input_depth(0), stride(stride), padding(padding), forwardThreadPool(std::thread::hardware_concurrency()), backwardThreadPool(std::thread::hardware_concurrency())
{
    initializeKernels(kernel_init);
    initializeBiases(bias_init);
}

void ConvolutionLayer::setInputDepth(int depth)
{
    input_depth = depth;
    initializeKernels(ConvKernelInitialization::HE); // Reinitialize kernels with new input depth
}

int ConvolutionLayer::getStride() const
{
    return stride;
}

int ConvolutionLayer::getFilters() const
{
    return filters;
}

int ConvolutionLayer::getKernelSize() const
{
    return kernel_size;
}

int ConvolutionLayer::getPadding() const
{
    return padding;
}

void ConvolutionLayer::initializeKernels(ConvKernelInitialization kernel_init)
{
    kernels.resize(filters);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis;

    switch (kernel_init)
    {
    case ConvKernelInitialization::HE:
        dis = std::normal_distribution<>(0, std::sqrt(2.0 / (input_depth * kernel_size * kernel_size)));
        break;
    case ConvKernelInitialization::XAVIER:
        dis = std::normal_distribution<>(0, std::sqrt(1.0 / (input_depth * kernel_size * kernel_size)));
        break;
    case ConvKernelInitialization::RANDOM_NORMAL:
        dis = std::normal_distribution<>(0, 1.0);
        break;
    }

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
            std::cout << "Filter " << f << ", Depth " << d << " kernel (first 3x3 block):\n"
                      << kernel.block(0, 0, std::min(3, static_cast<int>(kernel.rows())), std::min(3, static_cast<int>(kernel.cols()))) << " ..." << std::endl;
        }
    }
}

void ConvolutionLayer::initializeBiases(ConvBiasInitialization bias_init)
{
    biases.resize(filters);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> bias_dis(0, 1.0); // Initialize outside of switch statement

    switch (bias_init)
    {
    case ConvBiasInitialization::ZERO:
        biases = Eigen::VectorXd::Zero(filters);
        break;
    case ConvBiasInitialization::RANDOM_NORMAL:
        for (int i = 0; i < filters; ++i)
        {
            biases(i) = bias_dis(gen);
        }
        break;
    case ConvBiasInitialization::NONE:
        biases = Eigen::VectorXd::Zero(filters);
        break;
    }

    if (biases.size() != filters)
    {
        std::cerr << "Warning: Mismatch in biases size, initializing to zero.\n";
        this->biases = Eigen::VectorXd::Zero(filters);
    }

    std::cout << "Initialized biases (first 5 elements):\n"
              << biases.head(std::min(5, static_cast<int>(biases.size()))) << " ..." << std::endl;
}

// Forward pass with parallel processing
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

    std::vector<std::future<void>> futures;

    for (int b = 0; b < batch_size; ++b)
    {
        futures.emplace_back(forwardThreadPool.enqueue([this, &output_batch, &input_batch, b, output_size, input_size]
                                                       { processForwardBatch(output_batch, input_batch, b); }));
    }

    for (auto &f : futures)
    {
        f.get();
    }

    // Debugging output: print a subset of the forward pass output
    std::cout << "Forward output (first 10 elements of the first image):\n"
              << output_batch.block(0, 0, 1, std::min(10, static_cast<int>(output_batch.cols()))) << " ..." << std::endl;

    // Update the static variables in MaxPoolingLayer
    MaxPoolingLayer::setInputSize(output_size);
    MaxPoolingLayer::setInputDepth(filters);

    return output_batch;
}

// Method to process each batch for forward pass
void ConvolutionLayer::processForwardBatch(Eigen::MatrixXd &output_batch, const Eigen::MatrixXd &input_batch, int batch_index)
{
    int input_size = std::sqrt(input_batch.cols() / input_depth);
    int output_size = (input_size - kernel_size + 2 * padding) / stride + 1;
    Eigen::Map<const Eigen::MatrixXd> input(input_batch.row(batch_index).data(), input_size, input_size * input_depth);

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
        std::lock_guard<std::mutex> lock(mutex);
        output_batch.block(batch_index, f * output_size * output_size, 1, output_size * output_size) = Eigen::Map<Eigen::RowVectorXd>(feature_map.data(), feature_map.size());
    }
}

// Backward pass with parallel processing
Eigen::MatrixXd ConvolutionLayer::backward(const Eigen::MatrixXd &d_output_batch, const Eigen::MatrixXd &input_batch, double learning_rate)
{
    int batch_size = input_batch.rows();
    int input_size = std::sqrt(input_batch.cols() / input_depth);
    int output_size = (input_size - kernel_size + 2 * padding) / stride + 1;

    std::vector<std::vector<Eigen::MatrixXd>> d_kernels(filters, std::vector<Eigen::MatrixXd>(input_depth, Eigen::MatrixXd::Zero(kernel_size, kernel_size)));
    Eigen::VectorXd d_biases = Eigen::VectorXd::Zero(filters);
    Eigen::MatrixXd d_input_batch = Eigen::MatrixXd::Zero(batch_size, input_size * input_size * input_depth);

    std::vector<std::future<void>> futures;

    for (int b = 0; b < batch_size; ++b)
    {
        futures.emplace_back(backwardThreadPool.enqueue([this, &d_output_batch, &input_batch, &d_input_batch, &d_kernels, &d_biases, b, learning_rate]
                                                        { processBackwardBatch(d_output_batch, input_batch, d_input_batch, d_kernels, d_biases, b, learning_rate); }));
    }

    for (auto &f : futures)
    {
        f.get();
    }

    // Print a subset of gradients for debugging
    std::cout << "Gradient weights (first 3x3 block of first filter and depth 0):\n"
              << d_kernels[0][0].block(0, 0, std::min(3, static_cast<int>(d_kernels[0][0].rows())), std::min(3, static_cast<int>(d_kernels[0][0].cols()))) << " ..." << std::endl;

    std::cout << "Gradient biases (first 5 elements):\n"
              << d_biases.head(std::min(20, static_cast<int>(d_biases.size())))
              << " ..." << std::endl;

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

// Method to process each batch for backward pass
void ConvolutionLayer::processBackwardBatch(const Eigen::MatrixXd &d_output_batch, const Eigen::MatrixXd &input_batch, Eigen::MatrixXd &d_input_batch,
                                            std::vector<std::vector<Eigen::MatrixXd>> &d_kernels, Eigen::VectorXd &d_biases, int batch_index, double learning_rate)
{
    int input_size = std::sqrt(input_batch.cols() / input_depth);
    int output_size = (input_size - kernel_size + 2 * padding) / stride + 1;
    Eigen::Map<const Eigen::MatrixXd> input(input_batch.row(batch_index).data(), input_size, input_size * input_depth);
    Eigen::Map<const Eigen::MatrixXd> d_output(d_output_batch.row(batch_index).data(), filters, output_size * output_size);

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
                            int row_index = row_start + k;
                            int col_index = col_start + l;

                            if (row_index < padded_input.rows() && col_index < padded_input.cols())
                            {
                                d_kernels[f][d](k, l) += d_output_reshaped(i, j) * padded_input(row_index, col_index);

                                if (row_index < d_input_slice.rows() && col_index < d_input_slice.cols())
                                {
                                    d_input_slice(row_index, col_index) += d_output_reshaped(i, j) * kernels[f][d](k, l);
                                }
                            }
                        }
                    }
                }
            }

            std::lock_guard<std::mutex> lock(mutex);
            Eigen::Map<Eigen::RowVectorXd>(d_input_batch.row(batch_index).data() + d * input_size * input_size, input_size * input_size) = Eigen::Map<Eigen::RowVectorXd>(d_input_slice.data(), d_input_slice.size());
        }

        d_biases(f) += d_output_reshaped.sum();
    }
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
