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

#ifndef TENSOR_OPERATIONS_HPP
#define TENSOR_OPERATIONS_HPP

#include "Common.hpp"

/**
 * @brief A utility class for applying updates to tensors.
 *
 * This class provides static methods to apply scaled updates to tensors,
 * which can be used in optimization algorithms for adjusting weights and biases.
 */
class TensorOperations
{
public:
    /**
     * @brief Apply updates to a 2D tensor of weights.
     *
     * This method subtracts the scaled updates from the weights.
     *
     * @param weights The tensor of weights to be updated.
     * @param updates The tensor containing the updates to be applied.
     * @param scale The scale factor to apply to the updates.
     */
    static void applyUpdates(Eigen::Tensor<double, 2> &weights,
                             const Eigen::Tensor<double, 2> &updates,
                             double scale);

    /**
     * @brief Apply updates to a 4D tensor of weights.
     *
     * This method subtracts the scaled updates from the weights.
     *
     * @param weights The tensor of weights to be updated.
     * @param updates The tensor containing the updates to be applied.
     * @param scale The scale factor to apply to the updates.
     */
    static void applyUpdates(Eigen::Tensor<double, 4> &weights,
                             const Eigen::Tensor<double, 4> &updates,
                             double scale);

    /**
     * @brief Apply updates to a 1D tensor of biases.
     *
     * This method subtracts the scaled updates from the biases.
     *
     * @param biases The tensor of biases to be updated.
     * @param updates The tensor containing the updates to be applied.
     * @param scale The scale factor to apply to the updates.
     */
    static void applyUpdates(Eigen::Tensor<double, 1> &biases,
                             const Eigen::Tensor<double, 1> &updates,
                             double scale);
};

#endif // TENSOR_OPERATIONS_HPP
