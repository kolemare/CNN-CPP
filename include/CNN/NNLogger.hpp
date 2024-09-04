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

#ifndef NNLOGGER_HPP
#define NNLOGGER_HPP

#include "Common.hpp"

/**
 * @class NNLogger
 * @brief Utility class for logging and tracking progress in neural network training.
 *
 * Provides static methods for logging tensor data, tracking training progress,
 * and managing CSV files for recording training metrics.
 */
class NNLogger
{
public:
    /**
     * @brief Prints the full content of a 4D tensor to the console.
     *
     * @param tensor The 4D tensor to print.
     * @param layerType The type of the layer associated with the tensor.
     * @param propagationType The type of propagation (forward or backward).
     */
    static void printFullTensor(const Eigen::Tensor<double, 4> &tensor,
                                const std::string &layerType,
                                PropagationType propagationType);

    /**
     * @brief Prints the full content of a 2D tensor to the console.
     *
     * @param tensor The 2D tensor to print.
     * @param layerType The type of the layer associated with the tensor.
     * @param propagationType The type of propagation (forward or backward).
     */
    static void printFullTensor(const Eigen::Tensor<double, 2> &tensor,
                                const std::string &layerType,
                                PropagationType propagationType);

    /**
     * @brief Prints a summary of a 4D tensor, including statistics such as mean and standard deviation.
     *
     * @param tensor The 4D tensor to summarize.
     * @param layerType The type of the layer associated with the tensor.
     * @param propagationType The type of propagation (forward or backward).
     */
    static void printTensorSummary(const Eigen::Tensor<double, 4> &tensor,
                                   const std::string &layerType,
                                   PropagationType propagationType);

    /**
     * @brief Prints a summary of a 2D tensor, including statistics such as mean and standard deviation.
     *
     * @param tensor The 2D tensor to summarize.
     * @param layerType The type of the layer associated with the tensor.
     * @param propagationType The type of propagation (forward or backward).
     */
    static void printTensorSummary(const Eigen::Tensor<double, 2> &tensor,
                                   const std::string &layerType,
                                   PropagationType propagationType);

    /**
     * @brief Prints the progress of training, including the current epoch, batch, and estimated time remaining.
     *
     * @param epoch The current epoch number.
     * @param epochs The total number of epochs.
     * @param batch The current batch number.
     * @param totalBatches The total number of batches.
     * @param start The starting time point of the training process.
     * @param currentBatchLoss The loss for the current batch.
     * @param progressLevel The level of progress detail to display.
     * @param cumulative_loss The accumulated loss over batches.
     * @param total_batches_completed The number of batches completed.
     */
    static void printProgress(int epoch,
                              int epochs,
                              int batch,
                              int totalBatches,
                              std::chrono::steady_clock::time_point start,
                              double currentBatchLoss,
                              ProgressLevel progressLevel,
                              double &cumulative_loss,       // Passed as reference
                              int &total_batches_completed); // Passed as reference

    /**
     * @brief Initializes a CSV file for logging training metrics.
     *
     * @param filePath The path to csv file for initialization.
     * @throw std::runtime_error If the file cannot be opened.
     */
    static void initializeCSV(const std::string &filePath);

    /**
     * @brief Appends training metrics for a specific epoch to the CSV file.
     *
     * @param filename The name of the CSV file to append to.
     * @param epoch The current epoch number.
     * @param trainAcc The training accuracy for the current epoch.
     * @param trainLoss The training loss for the current epoch.
     * @param validAcc The validation accuracy for the current epoch.
     * @param validLoss The validation loss for the current epoch.
     * @param elralesState The ELRALES state description for the current epoch.
     * @throw std::runtime_error If the file cannot be opened.
     */
    static void appendToCSV(const std::string &filename,
                            int epoch,
                            double trainAcc,
                            double trainLoss,
                            double validAcc,
                            double validLoss,
                            const std::string &elralesState);
};

#endif // NNLOGGER_HPP
