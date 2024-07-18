#ifndef CONVOLUTIONLAYER_HPP
#define CONVOLUTIONLAYER_HPP

#include <vector>
#include <random>
#include <unsupported/Eigen/CXX11/Tensor>
#include "Layer.hpp"
#include "ThreadPool.hpp"
#include "Optimizer.hpp"

enum class ConvKernelInitialization
{
    HE,
    XAVIER,
    RANDOM_NORMAL
};

enum class ConvBiasInitialization
{
    ZERO,
    RANDOM_NORMAL,
    NONE
};

class ConvolutionLayer : public Layer
{
public:
    ConvolutionLayer(int filters, int kernel_size, int stride, int padding, ConvKernelInitialization kernel_init = ConvKernelInitialization::HE, ConvBiasInitialization bias_init = ConvBiasInitialization::ZERO);

    void setInputDepth(int depth);
    int getFilters() const;
    int getKernelSize() const;
    int getPadding() const;
    int getStride() const;

    Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4> &input_batch) override;
    Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4> &d_output_batch, const Eigen::Tensor<double, 4> &input_batch, double learning_rate) override;

    void setBiases(const Eigen::Tensor<double, 1> &new_biases);
    Eigen::Tensor<double, 1> getBiases() const;

    void setKernels(const Eigen::Tensor<double, 4> &new_kernels);

    Eigen::Tensor<double, 3> padInput(const Eigen::Tensor<double, 3> &input, int pad);
    double convolve(const Eigen::Tensor<double, 3> &input, const Eigen::Tensor<double, 2> &kernel, int start_row, int start_col);

    Eigen::Tensor<double, 4> kernels; // Kernels for each filter
    Eigen::Tensor<double, 1> biases;  // Biases for each filter

    static inline bool debugging = false;

    void initializeKernels(ConvKernelInitialization kernel_init);
    void initializeBiases(ConvBiasInitialization bias_init);

    bool needsOptimizer() const override;
    void setOptimizer(std::unique_ptr<Optimizer> optimizer) override;

private:
    int filters;
    int kernel_size;
    int input_depth;
    int stride;
    int padding;
    std::mutex mutex;

    std::unique_ptr<Optimizer> optimizer;

    // Thread pools for parallel processing
    ThreadPool forwardThreadPool;
    ThreadPool backwardThreadPool;

    void processForwardBatch(Eigen::Tensor<double, 4> &output_batch, const Eigen::Tensor<double, 4> &input_batch, int batch_index);
    void processBackwardBatch(const Eigen::Tensor<double, 4> &d_output_batch, const Eigen::Tensor<double, 4> &input_batch, Eigen::Tensor<double, 4> &d_input_batch,
                              Eigen::Tensor<double, 4> &d_kernels, Eigen::Tensor<double, 1> &d_biases, int batch_index, double learning_rate);
};

#endif // CONVOLUTIONLAYER_HPP
