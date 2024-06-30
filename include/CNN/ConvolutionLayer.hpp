#ifndef CONVOLUTIONLAYER_HPP
#define CONVOLUTIONLAYER_HPP

#include <vector>
#include <random>
#include <Eigen/Dense>
#include "Layer.hpp"
#include "MaxPoolingLayer.hpp"

class ConvolutionLayer : public Layer
{
public:
    ConvolutionLayer(int filters, int kernel_size, int input_depth, int stride, int padding, const Eigen::VectorXd &biases);

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
};

#endif // CONVOLUTIONLAYER_HPP
