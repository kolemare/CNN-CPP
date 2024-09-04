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

#ifndef LOSSFUNCTION_HPP
#define LOSSFUNCTION_HPP


#include "Common.hpp"

/**
 * @brief Abstract base class for loss functions.
 *
 * This class defines the interface for various loss functions used in neural network training.
 */
class LossFunction
{
public:
    /**
     * @brief Factory method to create a specific loss function based on the type.
     *
     * @param type The type of loss function to create.
     * @return std::unique_ptr<LossFunction> A unique pointer to the created loss function.
     * @throw std::invalid_argument If the loss type is unknown.
     */
    static std::unique_ptr<LossFunction> create(LossType type);

    /**
     * @brief Compute the loss given predictions and target values.
     *
     * @param predictions The predicted values.
     * @param targets The true target values.
     * @return double The computed loss.
     */
    virtual double compute(const Eigen::Tensor<double, 4> &predictions,
                           const Eigen::Tensor<int, 2> &targets) const = 0;

    /**
     * @brief Compute the derivative of the loss function with respect to the predictions.
     *
     * @param predictions The predicted values.
     * @param targets The true target values.
     * @return Eigen::Tensor<double, 4> The gradient of the loss with respect to the predictions.
     */
    virtual Eigen::Tensor<double, 4> derivative(const Eigen::Tensor<double, 4> &predictions,
                                                const Eigen::Tensor<int, 2> &targets) const = 0;

    /**
     * @brief Virtual destructor for the LossFunction class.
     */
    virtual ~LossFunction() = default;
};

/**
 * @brief Loss function for binary cross entropy.
 */
class BinaryCrossEntropy : public LossFunction
{
public:
    /**
     * @brief Compute the binary cross entropy loss.
     *
     * @param predictions The predicted values.
     * @param targets The true target values.
     * @return double The computed binary cross entropy loss.
     */
    double compute(const Eigen::Tensor<double, 4> &predictions,
                   const Eigen::Tensor<int, 2> &targets) const override;

    /**
     * @brief Compute the derivative of the binary cross entropy loss.
     *
     * @param predictions The predicted values.
     * @param targets The true target values.
     * @return Eigen::Tensor<double, 4> The gradient of the loss with respect to the predictions.
     */
    Eigen::Tensor<double, 4> derivative(const Eigen::Tensor<double, 4> &predictions,
                                        const Eigen::Tensor<int, 2> &targets) const override;
};

/**
 * @brief Loss function for mean squared error.
 */
class MeanSquaredError : public LossFunction
{
public:
    /**
     * @brief Compute the mean squared error loss.
     *
     * @param predictions The predicted values.
     * @param targets The true target values.
     * @return double The computed mean squared error loss.
     */
    double compute(const Eigen::Tensor<double, 4> &predictions,
                   const Eigen::Tensor<int, 2> &targets) const override;

    /**
     * @brief Compute the derivative of the mean squared error loss.
     *
     * @param predictions The predicted values.
     * @param targets The true target values.
     * @return Eigen::Tensor<double, 4> The gradient of the loss with respect to the predictions.
     */
    Eigen::Tensor<double, 4> derivative(const Eigen::Tensor<double, 4> &predictions,
                                        const Eigen::Tensor<int, 2> &targets) const override;
};

/**
 * @brief Loss function for categorical cross entropy.
 */
class CategoricalCrossEntropy : public LossFunction
{
public:
    /**
     * @brief Compute the categorical cross entropy loss.
     *
     * @param predictions The predicted values.
     * @param targets The true target values.
     * @return double The computed categorical cross entropy loss.
     */
    double compute(const Eigen::Tensor<double, 4> &predictions,
                   const Eigen::Tensor<int, 2> &targets) const override;

    /**
     * @brief Compute the derivative of the categorical cross entropy loss.
     *
     * @param predictions The predicted values.
     * @param targets The true target values.
     * @return Eigen::Tensor<double, 4> The gradient of the loss with respect to the predictions.
     */
    Eigen::Tensor<double, 4> derivative(const Eigen::Tensor<double, 4> &predictions,
                                        const Eigen::Tensor<int, 2> &targets) const override;
};

#endif // LOSSFUNCTION_HPP
