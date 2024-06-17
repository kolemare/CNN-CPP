#ifndef CONVOLUTION_LAYER_HPP
#define CONVOLUTION_LAYER_HPP

#include <Eigen/Dense>

class ConvolutionLayer
{
public:
    ConvolutionLayer(int filters, int kernel_size);

    int getFilters() const;
    int getKernelSize() const;
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input);

private:
    int filters;
    int kernel_size;
};

#endif // CONVOLUTION_LAYER_HPP
