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
