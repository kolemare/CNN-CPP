#ifndef NNLOGGER_HPP
#define NNLOGGER_HPP

#include <unsupported/Eigen/CXX11/Tensor>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <filesystem>
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
     */
    static void printProgress(int epoch,
                              int epochs,
                              int batch,
                              int totalBatches,
                              std::chrono::steady_clock::time_point start,
                              double currentBatchLoss,
                              ProgressLevel progressLevel);

    /**
     * @brief Initializes a CSV file for logging training metrics.
     *
     * @param filename The name of the CSV file to initialize.
     * @throw std::runtime_error If the file cannot be opened.
     */
    static void initializeCSV(const std::string &filename);

    /**
     * @brief Appends training metrics for a specific epoch to the CSV file.
     *
     * @param filename The name of the CSV file to append to.
     * @param epoch The current epoch number.
     * @param trainAcc The training accuracy for the current epoch.
     * @param trainLoss The training loss for the current epoch.
     * @param testAcc The testing accuracy for the current epoch.
     * @param testLoss The testing loss for the current epoch.
     * @param elralesState The ELRALES state description for the current epoch.
     * @throw std::runtime_error If the file cannot be opened.
     */
    static void appendToCSV(const std::string &filename,
                            int epoch,
                            double trainAcc,
                            double trainLoss,
                            double testAcc,
                            double testLoss,
                            const std::string &elralesState);
};

#endif // NNLOGGER_HPP
