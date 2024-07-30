#ifndef CONVOLUTIONLAYER_HPP
#define CONVOLUTIONLAYER_HPP

#include <vector>
#include <random>
#include <unsupported/Eigen/CXX11/Tensor>
#include "Layer.hpp"
#include "ThreadPool.hpp"
#include "Optimizer.hpp"

class ConvolutionLayer : public Layer
{
public:
    ConvolutionLayer(int filters, int kernel_size, int stride, int padding, ConvKernelInitialization kernel_init = ConvKernelInitialization::XAVIER, ConvBiasInitialization bias_init = ConvBiasInitialization::ZERO);

    void setInputDepth(int depth);
    int getFilters() const;
    int getKernelSize() const;
    int getPadding() const;
    int getStride() const;

    Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4> &input_batch) override;
    Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4> &d_output_batch, const Eigen::Tensor<double, 4> &input_batch, double learning_rate) override;

    void setBiases(const Eigen::Tensor<double, 1> &new_biases);
    Eigen::Tensor<double, 1> getBiases() const;
    Eigen::Tensor<double, 4> getKernels() const;
    void setKernels(const Eigen::Tensor<double, 4> &new_kernels);

    Eigen::Tensor<double, 2> padInput(const Eigen::Tensor<double, 2> &input, int pad);
    double convolve(const Eigen::Tensor<double, 2> &input, const Eigen::Tensor<double, 2> &kernel, int start_row, int start_col);

    void initializeKernels(ConvKernelInitialization kernel_init);
    void initializeBiases(ConvBiasInitialization bias_init);

    bool needsOptimizer() const override;
    void setOptimizer(std::shared_ptr<Optimizer> optimizer) override;
    std::shared_ptr<Optimizer> getOptimizer() override;

private:
    int filters;
    int kernel_size;
    int input_depth;
    int stride;
    int padding;
    std::mutex mutex;

    std::shared_ptr<Optimizer> optimizer;

    Eigen::Tensor<double, 4> kernels; // Kernels for each filter
    Eigen::Tensor<double, 1> biases;  // Biases for each filter

    // Thread pools for parallel processing
    ThreadPool forwardThreadPool;
    ThreadPool backwardThreadPool;

    void processForwardBatch(Eigen::Tensor<double, 4> &output_batch, const Eigen::Tensor<double, 4> &input_batch, int batch_index);
    void processBackwardBatch(const Eigen::Tensor<double, 4> &d_output_batch, const Eigen::Tensor<double, 4> &input_batch, Eigen::Tensor<double, 4> &d_input_batch,
                              Eigen::Tensor<double, 4> &d_kernels, Eigen::Tensor<double, 1> &d_biases, int batch_index);
};

#endif // CONVOLUTIONLAYER_HPP
