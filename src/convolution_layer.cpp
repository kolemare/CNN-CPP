#include "convolution_layer.hpp"

ConvolutionLayer::ConvolutionLayer(int filters, int kernel_size)
    : filters(filters), kernel_size(kernel_size) {}

int ConvolutionLayer::getFilters() const
{
    return filters;
}

int ConvolutionLayer::getKernelSize() const
{
    return kernel_size;
}

Eigen::MatrixXd ConvolutionLayer::forward(const Eigen::MatrixXd &input)
{
    // Implement the forward pass here
    return Eigen::MatrixXd(); // Placeholder
}
