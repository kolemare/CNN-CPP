/*
MIT License
Copyright (c) 2024 Marko Kostić

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

This project is the CNN-CPP Framework. Usage of this code is free, and 
uploading and using the code is also free, with a humble request to mention 
the origin of the implementation, the author Marko Kostić, and the repository 
link: https://github.com/kolemare/CNN-CPP.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
*/

#ifndef GRADIENTCLIPPING_HPP
#define GRADIENTCLIPPING_HPP

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
