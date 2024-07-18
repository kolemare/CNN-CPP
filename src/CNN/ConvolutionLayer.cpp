#include "ConvolutionLayer.hpp"
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

bool ConvolutionLayer::needsOptimizer() const
{
    return true;
}

void ConvolutionLayer::setOptimizer(std::unique_ptr<Optimizer> optimizer)
{
    this->optimizer = std::move(optimizer);
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

    kernels = Eigen::Tensor<double, 4>(filters, input_depth, kernel_size, kernel_size);
    for (int f = 0; f < filters; ++f)
    {
        for (int d = 0; d < input_depth; ++d)
        {
            for (int i = 0; i < kernel_size; ++i)
            {
                for (int j = 0; j < kernel_size; ++j)
                {
                    kernels(f, d, i, j) = dis(gen);
                }
            }
        }
    }
}

void ConvolutionLayer::initializeBiases(ConvBiasInitialization bias_init)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> bias_dis(0, 1.0); // Initialize outside of switch statement

    biases = Eigen::Tensor<double, 1>(filters);
    switch (bias_init)
    {
    case ConvBiasInitialization::ZERO:
        biases.setZero();
        break;
    case ConvBiasInitialization::RANDOM_NORMAL:
        for (int i = 0; i < filters; ++i)
        {
            biases(i) = bias_dis(gen);
        }
        break;
    case ConvBiasInitialization::NONE:
        biases.setZero();
        break;
    }

    if (biases.dimension(0) != filters)
    {
        std::cerr << "Warning: Mismatch in biases size, initializing to zero.\n";
        biases.setZero();
    }
}

// Forward pass with parallel processing
Eigen::Tensor<double, 4> ConvolutionLayer::forward(const Eigen::Tensor<double, 4> &input_batch)
{
    int batch_size = input_batch.dimension(0);
    int input_size = input_batch.dimension(2);
    if (input_size * input_size * input_depth != input_batch.dimension(1) * input_batch.dimension(2) * input_batch.dimension(3))
    {
        throw std::invalid_argument("Input dimensions do not match the expected depth and size.");
    }

    int output_size = (input_size - kernel_size + 2 * padding) / stride + 1;
    Eigen::Tensor<double, 4> output_batch(batch_size, filters, output_size, output_size);

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

    return output_batch;
}

// Method to process each batch for forward pass
void ConvolutionLayer::processForwardBatch(Eigen::Tensor<double, 4> &output_batch, const Eigen::Tensor<double, 4> &input_batch, int batch_index)
{
    int input_size = input_batch.dimension(2);
    int output_size = (input_size - kernel_size + 2 * padding) / stride + 1;

    for (int f = 0; f < filters; ++f)
    {
        Eigen::Tensor<double, 2> feature_map(output_size, output_size);
        feature_map.setZero();

        for (int d = 0; d < input_depth; ++d)
        {
            Eigen::Tensor<double, 3> input_slice = input_batch.chip(batch_index, 0).chip(d, 0);
            Eigen::Tensor<double, 3> padded_input = padInput(input_slice, padding);

            for (int i = 0; i < output_size; ++i)
            {
                for (int j = 0; j < output_size; ++j)
                {
                    int row_start = i * stride;
                    int col_start = j * stride;
                    double conv_sum = convolve(padded_input, kernels.chip(f, 0).chip(d, 0), row_start, col_start);
                    feature_map(i, j) += conv_sum;
                }
            }
        }

        // Apply biases
        feature_map = feature_map + biases(f);

        // Copy feature_map to output_batch
        std::lock_guard<std::mutex> lock(mutex);
        output_batch.chip(batch_index, 0).chip(f, 0) = feature_map;
    }
}

// Backward pass with parallel processing
Eigen::Tensor<double, 4> ConvolutionLayer::backward(const Eigen::Tensor<double, 4> &d_output_batch, const Eigen::Tensor<double, 4> &input_batch, double learning_rate)
{
    int batch_size = input_batch.dimension(0);
    int input_size = input_batch.dimension(2);
    int output_size = (input_size - kernel_size + 2 * padding) / stride + 1;

    Eigen::Tensor<double, 4> d_kernels(filters, input_depth, kernel_size, kernel_size);
    d_kernels.setZero();
    Eigen::Tensor<double, 1> d_biases(filters);
    d_biases.setZero();
    Eigen::Tensor<double, 4> d_input_batch(batch_size, input_depth, input_size, input_size);
    d_input_batch.setZero();

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

    // Update weights and biases using the optimizer
    optimizer->update(kernels, biases, d_kernels, d_biases, learning_rate);

    return d_input_batch;
}

// Method to process each batch for backward pass
void ConvolutionLayer::processBackwardBatch(const Eigen::Tensor<double, 4> &d_output_batch, const Eigen::Tensor<double, 4> &input_batch, Eigen::Tensor<double, 4> &d_input_batch,
                                            Eigen::Tensor<double, 4> &d_kernels, Eigen::Tensor<double, 1> &d_biases, int batch_index, double learning_rate)
{
    int input_size = input_batch.dimension(2);
    int output_size = (input_size - kernel_size + 2 * padding) / stride + 1;

    for (int f = 0; f < filters; ++f)
    {
        Eigen::Tensor<double, 2> d_output_reshaped = d_output_batch.chip(batch_index, 0).chip(f, 0);

        for (int d = 0; d < input_depth; ++d)
        {
            Eigen::Tensor<double, 3> input_slice = input_batch.chip(batch_index, 0).chip(d, 0);
            Eigen::Tensor<double, 3> padded_input = padInput(input_slice, padding);
            Eigen::Tensor<double, 3> d_input_slice(input_size, input_size, 1);
            d_input_slice.setZero();

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

                            if (row_index < padded_input.dimension(0) && col_index < padded_input.dimension(1))
                            {
                                d_kernels(f, d, k, l) += d_output_reshaped(i, j) * padded_input(row_index, col_index);

                                if (row_index < d_input_slice.dimension(0) && col_index < d_input_slice.dimension(1))
                                {
                                    d_input_slice(row_index, col_index) += d_output_reshaped(i, j) * kernels(f, d, k, l);
                                }
                            }
                        }
                    }
                }
            }

            std::lock_guard<std::mutex> lock(mutex);
            d_input_batch.chip(batch_index, 0).chip(d, 0) = d_input_slice;
        }

        // Apply bias updates per output feature map
        for (int i = 0; i < output_size; ++i)
        {
            for (int j = 0; j < output_size; ++j)
            {
                d_biases(f) += d_output_reshaped(i, j);
            }
        }
    }
}

void ConvolutionLayer::setBiases(const Eigen::Tensor<double, 1> &new_biases)
{
    if (new_biases.dimension(0) == filters)
    {
        biases = new_biases;
    }
    else
    {
        std::cerr << "Error: The size of new_biases must match the number of filters.\n";
    }
}

Eigen::Tensor<double, 1> ConvolutionLayer::getBiases() const
{
    return biases;
}

Eigen::Tensor<double, 3> ConvolutionLayer::padInput(const Eigen::Tensor<double, 3> &input, int pad)
{
    if (pad == 0)
        return input;

    int padded_size = input.dimension(0) + 2 * pad;
    Eigen::Tensor<double, 3> padded_input(padded_size, padded_size, input.dimension(2));
    padded_input.setZero();
    padded_input.slice(Eigen::array<int, 3>{pad, pad, 0}, input.dimensions()) = input;
    return padded_input;
}

double ConvolutionLayer::convolve(const Eigen::Tensor<double, 3> &input, const Eigen::Tensor<double, 2> &kernel, int start_row, int start_col)
{
    double sum = 0.0;
    for (int i = 0; i < kernel.dimension(0); ++i)
    {
        for (int j = 0; j < kernel.dimension(1); ++j)
        {
            sum += input(start_row + i, start_col + j, 0) * kernel(i, j);
        }
    }
    return sum;
}

void ConvolutionLayer::setKernels(const Eigen::Tensor<double, 4> &new_kernels)
{
    if (new_kernels.dimension(0) == filters && new_kernels.dimension(1) == input_depth)
    {
        kernels = new_kernels;
    }
    else
    {
        std::cerr << "Error: The size of new_kernels must match the number of filters and input depth." << std::endl;
    }
}
