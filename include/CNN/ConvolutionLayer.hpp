#ifndef CONVOLUTION_LAYER_HPP
#define CONVOLUTION_LAYER_HPP

#include <vector>
#include <Eigen/Dense>

class ConvolutionLayer
{
public:
    ConvolutionLayer(int filters, int kernel_size, int input_depth, int stride, int padding, const Eigen::VectorXd &biases = Eigen::VectorXd::Zero(1));

    std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd> &input);

    void setBiases(const Eigen::VectorXd &new_biases);

    // Make these methods public for testing
    Eigen::MatrixXd padInput(const Eigen::MatrixXd &input, int pad);
    double convolve(const Eigen::MatrixXd &input, const Eigen::MatrixXd &kernel, int start_row, int start_col);

    std::vector<std::vector<Eigen::MatrixXd>> kernels; // Kernels for each filter, each filter has a kernel for each input depth
    Eigen::VectorXd biases;                            // Biases for each filter

private:
    int filters;
    int kernel_size;
    int input_depth;
    int stride;
    int padding;
};

#endif // CONVOLUTION_LAYER_HPP
