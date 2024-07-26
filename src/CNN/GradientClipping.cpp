#include "GradientClipping.hpp"
#include <algorithm>
#include <cmath>

void GradientClipping::clipGradients(Eigen::Tensor<double, 4> &gradients, double clip_value)
{
    for (int i = 0; i < gradients.size(); ++i)
    {
        gradients.data()[i] = std::max(std::min(gradients.data()[i], clip_value), -clip_value);
    }
}

void GradientClipping::clipGradients(Eigen::Tensor<double, 2> &gradients, double clip_value)
{
    for (int i = 0; i < gradients.size(); ++i)
    {
        gradients.data()[i] = std::max(std::min(gradients.data()[i], clip_value), -clip_value);
    }
}
