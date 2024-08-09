#ifndef CONVOLUTIONLAYER_HPP
#define CONVOLUTIONLAYER_HPP

#include <vector>
#include <random>
#include <unsupported/Eigen/CXX11/Tensor>
#include "Layer.hpp"
#include "ThreadPool.hpp"
#include "Optimizer.hpp"

/**
 * @class ConvolutionLayer
 * @brief Represents a convolutional layer in a neural network.
 *
 * This class implements a convolutional layer, supporting forward and backward
 * passes, weight initialization, and bias management. It uses multithreading
 * to process data in parallel for improved performance.
 */
class ConvolutionLayer : public Layer
{
public:
    /**
     * @brief Constructs a ConvolutionLayer with the specified parameters.
     *
     * @param filters Number of filters in the layer.
     * @param kernel_size Size of each filter (assumed square).
     * @param stride Stride of the convolution operation.
     * @param padding Padding size for input data.
     * @param kernel_init Method for kernel initialization (default: Xavier).
     * @param bias_init Method for bias initialization (default: Zero).
     */
    ConvolutionLayer(int filters,
                     int kernel_size,
                     int stride,
                     int padding,
                     ConvKernelInitialization kernel_init = ConvKernelInitialization::XAVIER,
                     ConvBiasInitialization bias_init = ConvBiasInitialization::ZERO);

    /**
     * @brief Destructor to clean up resources used by the ConvolutionLayer.
     */
    ~ConvolutionLayer();

    /**
     * @brief Sets the input depth, reinitializing kernels as needed.
     *
     * @param depth The depth of the input data.
     */
    void setInputDepth(int depth);

    /**
     * @brief Returns the number of filters in the layer.
     *
     * @return Number of filters.
     */
    int getFilters() const;

    /**
     * @brief Returns the kernel size of the layer.
     *
     * @return Kernel size.
     */
    int getKernelSize() const;

    /**
     * @brief Returns the padding size used in the layer.
     *
     * @return Padding size.
     */
    int getPadding() const;

    /**
     * @brief Returns the stride of the convolution operation.
     *
     * @return Stride value.
     */
    int getStride() const;

    /**
     * @brief Performs the forward pass of the convolution operation.
     *
     * @param input_batch The input data batch as a 4D tensor.
     * @return The output data batch after convolution.
     */
    Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4> &input_batch) override;

    /**
     * @brief Performs the backward pass of the convolution operation.
     *
     * @param d_output_batch The gradient of the loss with respect to the output.
     * @param input_batch The original input data batch.
     * @param learning_rate The learning rate for updating parameters.
     * @return The gradient of the loss with respect to the input.
     */
    Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4> &d_output_batch,
                                      const Eigen::Tensor<double, 4> &input_batch,
                                      double learning_rate) override;

    /**
     * @brief Sets the biases for the convolutional layer.
     *
     * @param new_biases A tensor containing the new bias values.
     */
    void setBiases(const Eigen::Tensor<double, 1> &new_biases);

    /**
     * @brief Returns the biases used in the layer.
     *
     * @return A tensor containing the bias values.
     */
    Eigen::Tensor<double, 1> getBiases() const;

    /**
     * @brief Returns the kernels used in the layer.
     *
     * @return A tensor containing the kernel values.
     */
    Eigen::Tensor<double, 4> getKernels() const;

    /**
     * @brief Sets the kernels for the convolutional layer.
     *
     * @param new_kernels A tensor containing the new kernel values.
     */
    void setKernels(const Eigen::Tensor<double, 4> &new_kernels);

    /**
     * @brief Pads the input tensor with zeros.
     *
     * @param input The input tensor to be padded.
     * @param pad The amount of padding to add to each side.
     * @return A padded tensor.
     */
    Eigen::Tensor<double, 2> padInput(const Eigen::Tensor<double, 2> &input,
                                      int pad);

    /**
     * @brief Performs the convolution operation on a given window.
     *
     * @param input The input tensor.
     * @param kernel The kernel tensor.
     * @param start_row Starting row for the convolution window.
     * @param start_col Starting column for the convolution window.
     * @return The result of the convolution.
     */
    double convolve(const Eigen::Tensor<double, 2> &input,
                    const Eigen::Tensor<double, 2> &kernel,
                    int start_row,
                    int start_col);

    /**
     * @brief Initializes the kernels using the specified method.
     *
     * @param kernel_init The initialization method for kernels.
     */
    void initializeKernels(ConvKernelInitialization kernel_init);

    /**
     * @brief Initializes the biases using the specified method.
     *
     * @param bias_init The initialization method for biases.
     */
    void initializeBiases(ConvBiasInitialization bias_init);

    /**
     * @brief Checks if the layer requires an optimizer.
     *
     * @return True if an optimizer is needed, false otherwise.
     */
    bool needsOptimizer() const override;

    /**
     * @brief Sets the optimizer for the convolutional layer.
     *
     * @param optimizer A shared pointer to the optimizer object.
     */
    void setOptimizer(std::shared_ptr<Optimizer> optimizer) override;

    /**
     * @brief Returns the optimizer set for the layer.
     *
     * @return A shared pointer to the optimizer object.
     */
    std::shared_ptr<Optimizer> getOptimizer() override;

private:
    int filters;     ///< Number of filters
    int kernel_size; ///< Size of each kernel
    int input_depth; ///< Depth of the input
    int stride;      ///< Stride of the convolution
    int padding;     ///< Padding size

    ConvKernelInitialization kernel_init; ///< The initialization method for kernels
    ConvBiasInitialization bias_init;     ///< The initialization method for biases

    std::mutex mutex; ///< Mutex for thread safety

    std::shared_ptr<Optimizer> optimizer; ///< Optimizer for the layer

    Eigen::Tensor<double, 4> kernels; ///< Kernels for each filter
    Eigen::Tensor<double, 1> biases;  ///< Biases for each filter

    ThreadPool forwardThreadPool;  ///< Thread pool for forward pass
    ThreadPool backwardThreadPool; ///< Thread pool for backward pass

    /**
     * @brief Processes a batch during the forward pass.
     *
     * @param output_batch The output batch tensor to store results.
     * @param input_batch The input batch tensor.
     * @param batch_index The index of the batch to process.
     */
    void processForwardBatch(Eigen::Tensor<double, 4> &output_batch,
                             const Eigen::Tensor<double, 4> &input_batch,
                             int batch_index);

    /**
     * @brief Processes a batch during the backward pass.
     *
     * @param d_output_batch The gradient of the output batch.
     * @param input_batch The input batch tensor.
     * @param d_input_batch The gradient of the input batch.
     * @param d_kernels The gradient of the kernels.
     * @param d_biases The gradient of the biases.
     * @param batch_index The index of the batch to process.
     */
    void processBackwardBatch(const Eigen::Tensor<double, 4> &d_output_batch,
                              const Eigen::Tensor<double, 4> &input_batch,
                              Eigen::Tensor<double, 4> &d_input_batch,
                              Eigen::Tensor<double, 4> &d_kernels,
                              Eigen::Tensor<double, 1> &d_biases,
                              int batch_index);
};

#endif // CONVOLUTIONLAYER_HPP
