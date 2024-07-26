#ifndef GRADIENTCLIPPING_HPP
#define GRADIENTCLIPPING_HPP

#include <unsupported/Eigen/CXX11/Tensor>

class GradientClipping
{
public:
    static constexpr double default_clip_value = 1.0; // Set the default clipping value

    // Function to clip gradients for 4D tensors
    static void clipGradients(Eigen::Tensor<double, 4> &gradients, double clip_value = default_clip_value);

    // Function to clip gradients for 2D tensors
    static void clipGradients(Eigen::Tensor<double, 2> &gradients, double clip_value = default_clip_value);
};

#endif // GRADIENTCLIPPING_HPP
