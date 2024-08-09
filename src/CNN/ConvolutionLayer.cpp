#include "ConvolutionLayer.hpp"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <future>
#include <mutex>

/**
 * @brief Constructor for ConvolutionLayer with specified kernel and bias initializations.
 *
 * @param filters Number of filters.
 * @param kernel_size Size of the convolution kernel.
 * @param stride Stride of the convolution.
 * @param padding Padding added to the input.
 * @param kernel_init Method for initializing the kernels.
 * @param bias_init Method for initializing the biases.
 */
ConvolutionLayer::ConvolutionLayer(int filters,
                                   int kernel_size,
                                   int stride,
                                   int padding,
                                   ConvKernelInitialization kernel_init,
                                   ConvBiasInitialization bias_init)
{
    this->filters = filters;
    this->kernel_size = kernel_size;
    this->input_depth = 0;
    this->stride = stride;
    this->padding = padding;
    this->kernel_init = kernel_init;
    this->bias_init = bias_init;

    new (&forwardThreadPool) ThreadPool(std::thread::hardware_concurrency());
    new (&backwardThreadPool) ThreadPool(std::thread::hardware_concurrency());

    initializeKernels(kernel_init);
    initializeBiases(bias_init);
}

/**
 * @brief Destructor for ConvolutionLayer class.
 *
 * This destructor explicitly calls the destructors of the `forwardThreadPool`
 * and `backwardThreadPool` objects to ensure proper cleanup of resources
 * allocated by these thread pools. This is necessary because placement `new`
 * was used to initialize the thread pools, requiring manual management
 * of their lifetimes.
 *
 * @note Ensure that the `ThreadPool` class has a proper destructor to handle
 * any resources it manages, such as threads or allocated memory.
 */
ConvolutionLayer::~ConvolutionLayer()
{
    // Call the destructors explicitly
    forwardThreadPool.~ThreadPool();
    backwardThreadPool.~ThreadPool();
}

/**
 * @brief Indicates that the ConvolutionLayer needs an optimizer.
 *
 * @return true Always returns true.
 */
bool ConvolutionLayer::needsOptimizer() const
{
    return true;
}

/**
 * @brief Set the optimizer for the convolution layer.
 *
 * @param optimizer A shared pointer to the optimizer to use.
 */
void ConvolutionLayer::setOptimizer(std::shared_ptr<Optimizer> optimizer)
{
    this->optimizer = optimizer;
}

/**
 * @brief Get the optimizer associated with the convolution layer.
 *
 * @return A shared pointer to the optimizer.
 */
std::shared_ptr<Optimizer> ConvolutionLayer::getOptimizer()
{
    return this->optimizer;
}

/**
 * @brief Set the input depth and reinitialize the kernels.
 *
 * @param depth The depth of the input.
 */
void ConvolutionLayer::setInputDepth(int depth)
{
    input_depth = depth;
    initializeKernels(this->kernel_init); // Reinitialize kernels with new input depth
    initializeBiases(this->bias_init);    // Reinitialize biases, no need for this
}

/**
 * @brief Get the stride used by the convolution layer.
 *
 * @return The stride.
 */
int ConvolutionLayer::getStride() const
{
    return stride;
}

/**
 * @brief Get the number of filters in the convolution layer.
 *
 * @return The number of filters.
 */
int ConvolutionLayer::getFilters() const
{
    return filters;
}

/**
 * @brief Get the kernel size used by the convolution layer.
 *
 * @return The kernel size.
 */
int ConvolutionLayer::getKernelSize() const
{
    return kernel_size;
}

/**
 * @brief Get the padding used by the convolution layer.
 *
 * @return The padding.
 */
int ConvolutionLayer::getPadding() const
{
    return padding;
}

/**
 * @brief Initialize the kernels based on the specified initialization method.
 *
 * @param kernel_init The method for initializing the kernels.
 */
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

/**
 * @brief Initialize the biases based on the specified initialization method.
 *
 * @param bias_init The method for initializing the biases.
 */
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

/**
 * @brief Perform the forward pass of the convolution layer.
 *
 * @param input_batch A 4D tensor containing the input data.
 * @return A 4D tensor containing the output of the convolution.
 */
Eigen::Tensor<double, 4> ConvolutionLayer::forward(const Eigen::Tensor<double, 4> &input_batch)
{
    // Batch size (number of images in the batch)
    int batch_size = input_batch.dimension(0); // Number of images

    // Input channels (depth), height, and width
    int input_depth = input_batch.dimension(1);  // Depth of the input, e.g., 3 for RGB
    int input_height = input_batch.dimension(2); // Height of each image
    int input_width = input_batch.dimension(3);  // Width of each image

    // Calculate the output height and width
    int output_height = (input_height - kernel_size + 2 * padding) / stride + 1; // Height of the output feature map
    int output_width = (input_width - kernel_size + 2 * padding) / stride + 1;   // Width of the output feature map

    // Output tensor: (batch_size, filters, output_height, output_width)
    Eigen::Tensor<double, 4> output_batch(batch_size, filters, output_height, output_width); // Output tensor to store the result of the convolution

    // Create a vector to store futures for parallel processing
    std::vector<std::future<void>> futures;

    // Iterate over each image in the batch
    for (int b = 0; b < batch_size; ++b)
    {
        // Enqueue the processing of each image to the thread pool
        futures.emplace_back(forwardThreadPool.enqueue([this, &output_batch, &input_batch, b]
                                                       {
            // Process the current batch
            processForwardBatch(output_batch, input_batch, b); }));
    }

    // Wait for all threads to complete
    for (auto &f : futures)
    {
        f.get();
    }

    // Return the output batch
    return output_batch;
}

/**
 * @brief Process each batch for the forward pass.
 *
 * @param output_batch A 4D tensor to store the result of the convolution.
 * @param input_batch A 4D tensor containing the input data.
 * @param batch_index The index of the batch to process.
 */
void ConvolutionLayer::processForwardBatch(Eigen::Tensor<double, 4> &output_batch,
                                           const Eigen::Tensor<double, 4> &input_batch,
                                           int batch_index)
{
    // Input channels (depth), height, and width
    int input_depth = input_batch.dimension(1);  // Depth of the input, e.g., 3 for RGB
    int input_height = input_batch.dimension(2); // Height of each image
    int input_width = input_batch.dimension(3);  // Width of each image

    // Calculate the output height and width
    int output_height = (input_height - kernel_size + 2 * padding) / stride + 1; // Height of the output feature map
    int output_width = (input_width - kernel_size + 2 * padding) / stride + 1;   // Width of the output feature map

    // Iterate over each filter
    for (int f = 0; f < filters; ++f)
    {
        // Create a feature map for the current filter
        Eigen::Tensor<double, 2> feature_map(output_height, output_width); // 2D feature map for the current filter
        feature_map.setZero();

        // Iterate over each depth channel of the input
        for (int d = 0; d < input_depth; ++d)
        {
            // Extract the d-th channel of the current image from the batch
            Eigen::Tensor<double, 2> input_channel(input_height, input_width); // 2D tensor for a single input channel
            for (int i = 0; i < input_height; ++i)
            {
                for (int j = 0; j < input_width; ++j)
                {
                    input_channel(i, j) = input_batch(batch_index, d, i, j); // Copy data from the input batch to the input channel tensor
                }
            }

            // Pad the input channel
            Eigen::Tensor<double, 2> padded_input = padInput(input_channel, padding); // 2D padded input tensor

            // Extract the current 2D kernel for this filter and input depth
            Eigen::Tensor<double, 2> kernel(kernel_size, kernel_size); // 2D kernel tensor for the current filter and input depth
            for (int i = 0; i < kernel_size; ++i)
            {
                for (int j = 0; j < kernel_size; ++j)
                {
                    kernel(i, j) = kernels(f, d, i, j); // Copy data from the 4D kernels tensor to the 2D kernel tensor
                }
            }

            // Perform the convolution
            for (int i = 0; i < output_height; ++i)
            {
                for (int j = 0; j < output_width; ++j)
                {
                    int row_start = i * stride; // Starting row for the current window
                    int col_start = j * stride; // Starting column for the current window
                    // Convolve the input with the filter kernel
                    feature_map(i, j) += convolve(padded_input, kernel, row_start, col_start); // Add the result of the convolution to the feature map
                }
            }
        }

        // Add the bias for the current filter
        for (int i = 0; i < output_height; ++i)
        {
            for (int j = 0; j < output_width; ++j)
            {
                feature_map(i, j) += biases(f); // Add the bias to each element of the feature map
            }
        }

        // Copy the feature map to the output batch
        for (int i = 0; i < output_height; ++i)
        {
            for (int j = 0; j < output_width; ++j)
            {
                output_batch(batch_index, f, i, j) = feature_map(i, j); // Copy the feature map to the output batch
            }
        }
    }
}

/**
 * @brief Perform the backward pass of the convolution layer with parallel processing.
 *
 * @param d_output_batch A 4D tensor containing the gradient of the loss with respect to the output.
 * @param input_batch A 4D tensor containing the input data.
 * @param learning_rate The learning rate for updating the parameters.
 * @return A 4D tensor containing the gradient of the loss with respect to the input.
 */
Eigen::Tensor<double, 4> ConvolutionLayer::backward(const Eigen::Tensor<double, 4> &d_output_batch,
                                                    const Eigen::Tensor<double, 4> &input_batch,
                                                    double learning_rate)
{
    // Batch size (number of images in the batch)
    int batch_size = input_batch.dimension(0); // Number of images

    // Input channels (depth), height, and width
    int input_depth = input_batch.dimension(1);  // Depth of the input, e.g., 3 for RGB
    int input_height = input_batch.dimension(2); // Height of each image
    int input_width = input_batch.dimension(3);  // Width of each image

    // Calculate the output height and width
    int output_height = (input_height - kernel_size + 2 * padding) / stride + 1; // Height of the output feature map
    int output_width = (input_width - kernel_size + 2 * padding) / stride + 1;   // Width of the output feature map

    // Initialize tensors for gradients
    Eigen::Tensor<double, 4> d_kernels(filters, input_depth, kernel_size, kernel_size); // Gradient w.r.t kernels
    d_kernels.setZero();
    Eigen::Tensor<double, 1> d_biases(filters); // Gradient w.r.t biases
    d_biases.setZero();
    Eigen::Tensor<double, 4> d_input_batch(batch_size, input_depth, input_height, input_width); // Gradient w.r.t input
    d_input_batch.setZero();

    // Create a vector to store futures for parallel processing
    std::vector<std::future<void>> futures;

    // Iterate over each image in the batch
    for (int b = 0; b < batch_size; ++b)
    {
        // Enqueue the processing of each image to the thread pool
        futures.emplace_back(backwardThreadPool.enqueue([this, &d_output_batch, &input_batch, &d_input_batch, &d_kernels, &d_biases, b]
                                                        { processBackwardBatch(d_output_batch, input_batch, d_input_batch, d_kernels, d_biases, b); }));
    }

    // Wait for all threads to complete
    for (auto &f : futures)
    {
        f.get();
    }

    // Update weights and biases using the optimizer
    optimizer->update(kernels, biases, d_kernels, d_biases, learning_rate);

    // Return the gradient with respect to the input batch
    return d_input_batch;
}

/**
 * @brief Process each batch for the backward pass.
 *
 * @param d_output_batch A 4D tensor containing the gradient of the loss with respect to the output.
 * @param input_batch A 4D tensor containing the input data.
 * @param d_input_batch A 4D tensor to store the gradient of the loss with respect to the input.
 * @param d_kernels A 4D tensor to store the gradient with respect to the kernels.
 * @param d_biases A 1D tensor to store the gradient with respect to the biases.
 * @param batch_index The index of the batch to process.
 */
void ConvolutionLayer::processBackwardBatch(const Eigen::Tensor<double, 4> &d_output_batch,
                                            const Eigen::Tensor<double, 4> &input_batch,
                                            Eigen::Tensor<double, 4> &d_input_batch,
                                            Eigen::Tensor<double, 4> &d_kernels,
                                            Eigen::Tensor<double, 1> &d_biases,
                                            int batch_index)
{
    // Input channels (depth), height, and width
    int input_depth = input_batch.dimension(1);  // Depth of the input, e.g., 3 for RGB
    int input_height = input_batch.dimension(2); // Height of each image
    int input_width = input_batch.dimension(3);  // Width of each image

    // Calculate the output height and width
    int output_height = (input_height - kernel_size + 2 * padding) / stride + 1; // Height of the output feature map
    int output_width = (input_width - kernel_size + 2 * padding) / stride + 1;   // Width of the output feature map

    // Iterate over each filter
    for (int f = 0; f < filters; ++f)
    {
        // Create a tensor for the gradient of the current output
        Eigen::Tensor<double, 2> d_output(output_height, output_width); // 2D tensor for the gradient of the output
        for (int i = 0; i < output_height; ++i)
        {
            for (int j = 0; j < output_width; ++j)
            {
                d_output(i, j) = d_output_batch(batch_index, f, i, j); // Copy data from the 4D d_output_batch tensor to the 2D d_output tensor
            }
        }

        // Iterate over each depth channel of the input
        for (int d = 0; d < input_depth; ++d)
        {
            // Extract the d-th channel of the current image from the batch
            Eigen::Tensor<double, 2> input_channel(input_height, input_width); // 2D tensor for a single input channel
            for (int i = 0; i < input_height; ++i)
            {
                for (int j = 0; j < input_width; ++j)
                {
                    input_channel(i, j) = input_batch(batch_index, d, i, j); // Copy data from the input batch to the input channel tensor
                }
            }

            // Pad the input channel
            Eigen::Tensor<double, 2> padded_input = padInput(input_channel, padding); // 2D padded input tensor

            // Create a tensor for the gradient of the current input slice
            Eigen::Tensor<double, 2> d_input_slice(input_height, input_width); // 2D tensor for the gradient of the input slice
            d_input_slice.setZero();

            // Extract the current 2D kernel for this filter and input depth
            Eigen::Tensor<double, 2> kernel(kernel_size, kernel_size); // 2D kernel tensor for the current filter and input depth
            for (int i = 0; i < kernel_size; ++i)
            {
                for (int j = 0; j < kernel_size; ++j)
                {
                    kernel(i, j) = kernels(f, d, i, j); // Copy data from the 4D kernels tensor to the 2D kernel tensor
                }
            }

            // Perform the backward convolution to calculate the gradients
            for (int i = 0; i < output_height; ++i)
            {
                for (int j = 0; j < output_width; ++j)
                {
                    int row_start = i * stride; // Starting row for the current window
                    int col_start = j * stride; // Starting column for the current window
                    for (int ki = 0; ki < kernel_size; ++ki)
                    {
                        for (int kj = 0; kj < kernel_size; ++kj)
                        {
                            int row_index = row_start + ki; // Row index in the padded input
                            int col_index = col_start + kj; // Column index in the padded input

                            // Update gradients for the kernel
                            if (row_index < padded_input.dimension(0) && col_index < padded_input.dimension(1))
                            {
                                d_kernels(f, d, ki, kj) += d_output(i, j) * padded_input(row_index, col_index); // Accumulate the gradient for the kernel

                                // Update gradients for the input
                                if (row_index - padding >= 0 && row_index - padding < input_height && col_index - padding >= 0 && col_index - padding < input_width)
                                {
                                    d_input_slice(row_index - padding, col_index - padding) += d_output(i, j) * kernel(ki, kj); // Accumulate the gradient for the input
                                }
                            }
                        }
                    }
                }
            }

            // Add the gradient of the current input slice to the overall gradient of the input batch
            for (int i = 0; i < input_height; ++i)
            {
                for (int j = 0; j < input_width; ++j)
                {
                    d_input_batch(batch_index, d, i, j) += d_input_slice(i, j); // Accumulate the gradient for the input batch
                }
            }
        }

        // Apply bias updates per output feature map
        for (int i = 0; i < output_height; ++i)
        {
            for (int j = 0; j < output_width; ++j)
            {
                d_biases(f) += d_output(i, j); // Accumulate the gradient for the biases
            }
        }
    }
}

/**
 * @brief Set the biases for the convolution layer.
 *
 * @param new_biases A 1D tensor containing the new biases.
 */
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

/**
 * @brief Get the biases of the convolution layer.
 *
 * @return A 1D tensor containing the biases.
 */
Eigen::Tensor<double, 1> ConvolutionLayer::getBiases() const
{
    return Eigen::Tensor<double, 1>(biases); // Return a copy
}

/**
 * @brief Pad the input tensor with zeros.
 *
 * @param input A 2D tensor representing the input to pad.
 * @param pad The number of pixels to pad on each side.
 * @return A 2D tensor representing the padded input.
 */
Eigen::Tensor<double, 2> ConvolutionLayer::padInput(const Eigen::Tensor<double, 2> &input,
                                                    int pad)
{
    int input_height = input.dimension(0); // Height of the input tensor
    int input_width = input.dimension(1);  // Width of the input tensor

    int padded_height = input_height + 2 * pad; // Height of the padded tensor
    int padded_width = input_width + 2 * pad;   // Width of the padded tensor

    Eigen::Tensor<double, 2> padded_input(padded_height, padded_width); // 2D tensor for the padded input
    padded_input.setZero();                                             // Initialize the padded input with zeros

    for (int i = 0; i < input_height; ++i)
    {
        for (int j = 0; j < input_width; ++j)
        {
            padded_input(i + pad, j + pad) = input(i, j); // Copy the input tensor into the padded tensor
        }
    }

    return padded_input;
}

/**
 * @brief Perform the convolution operation.
 *
 * @param input A 2D tensor representing the input.
 * @param kernel A 2D tensor representing the kernel.
 * @param start_row The starting row index for the convolution.
 * @param start_col The starting column index for the convolution.
 * @return The result of the convolution as a double.
 */
double ConvolutionLayer::convolve(const Eigen::Tensor<double, 2> &input,
                                  const Eigen::Tensor<double, 2> &kernel,
                                  int start_row,
                                  int start_col)
{
    double sum = 0.0;                        // Sum of the element-wise products
    int kernel_height = kernel.dimension(0); // Height of the kernel tensor
    int kernel_width = kernel.dimension(1);  // Width of the kernel tensor

    for (int i = 0; i < kernel_height; ++i)
    {
        for (int j = 0; j < kernel_width; ++j)
        {
            // Sum up the element-wise product of the input and the kernel
            sum += input(start_row + i, start_col + j) * kernel(i, j); // Add the product of the corresponding elements to the sum
        }
    }
    return sum; // Return the result of the convolution
}

/**
 * @brief Set the kernels for the convolution layer.
 *
 * @param new_kernels A 4D tensor containing the new kernels.
 */
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

/**
 * @brief Get the kernels of the convolution layer.
 *
 * @return A 4D tensor containing the kernels.
 */
Eigen::Tensor<double, 4> ConvolutionLayer::getKernels() const
{
    return Eigen::Tensor<double, 4>(kernels); // Return a copy
}
