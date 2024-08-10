#ifndef GRADIENTCLIPPING_HPP
#define GRADIENTCLIPPING_HPP

#include <unsupported/Eigen/CXX11/Tensor>
#include "Common.hpp"

/**
 * @class GradientClipping
 * @brief A utility class for clipping gradients during training.
 *
 * This class provides static methods to clip gradients in both 2D and 4D tensors
 * to prevent excessively large gradients that can destabilize training.
 */
class GradientClipping
{
public:
    static constexpr double default_clip_value = 1.0; ///< The default value used for gradient clipping.

    /**
     * @brief Clip gradients in a 4D tensor to a specified range.
     *
     * This method adjusts the values in the provided gradient tensor to be within
     * the range [-clip_value, clip_value].
     *
     * @param gradients A reference to a 4D tensor of gradient values.
     * @param clip_value The maximum absolute value to which gradients should be clipped.
     *                   Defaults to `default_clip_value`.
     */
    static void clipGradients(Eigen::Tensor<double, 4> &gradients,
                              double clip_value = default_clip_value);

    /**
     * @brief Clip gradients in a 2D tensor to a specified range.
     *
     * This method adjusts the values in the provided gradient tensor to be within
     * the range [-clip_value, clip_value].
     *
     * @param gradients A reference to a 2D tensor of gradient values.
     * @param clip_value The maximum absolute value to which gradients should be clipped.
     *                   Defaults to `default_clip_value`.
     */
    static void clipGradients(Eigen::Tensor<double, 2> &gradients,
                              double clip_value = default_clip_value);
};

#endif // GRADIENTCLIPPING_HPP
