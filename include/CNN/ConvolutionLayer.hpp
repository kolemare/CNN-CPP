#ifndef CONVOLUTIONLAYER_HPP
#define CONVOLUTIONLAYER_HPP

#include <vector>
#include <random>
#include <Eigen/Dense>
#include "Layer.hpp"
#include "MaxPoolingLayer.hpp"
#include "ThreadPool.hpp"

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

    Eigen::MatrixXd forward(const Eigen::MatrixXd &input_batch) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd &d_output_batch, const Eigen::MatrixXd &input_batch, double learning_rate) override;

    void setBiases(const Eigen::VectorXd &new_biases);
    Eigen::VectorXd getBiases() const;

    void setKernels(const std::vector<std::vector<Eigen::MatrixXd>> &new_kernels);

    Eigen::MatrixXd padInput(const Eigen::MatrixXd &input, int pad);
    double convolve(const Eigen::MatrixXd &input, const Eigen::MatrixXd &kernel, int start_row, int start_col);

    std::vector<std::vector<Eigen::MatrixXd>> kernels; // Kernels for each filter, each filter has a kernel for each input depth
    Eigen::VectorXd biases;                            // Biases for each filter

    static inline bool debugging = false;

private:
    int filters;
    int kernel_size;
    int input_depth;
    int stride;
    int padding;

    // Thread pool for parallel processing
    ThreadPool forwardThreadPool;

    void processBatch(Eigen::MatrixXd &output_batch, const Eigen::MatrixXd &input_batch, int batch_index);
    void initializeKernels(ConvKernelInitialization kernel_init);
    void initializeBiases(ConvBiasInitialization bias_init);
};

#endif // CONVOLUTIONLAYER_HPP
