#ifndef NNLOGGER_HPP
#define NNLOGGER_HPP

#include <unsupported/Eigen/CXX11/Tensor>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include "Common.hpp"

class NNLogger
{
public:
    static void printFullTensor(const Eigen::Tensor<double, 4> &tensor, const std::string &layerType, PropagationType propagationType);
    static void printFullTensor(const Eigen::Tensor<double, 2> &tensor, const std::string &layerType, PropagationType propagationType);
    static void printTensorSummary(const Eigen::Tensor<double, 4> &tensor, const std::string &layerType, PropagationType propagationType);
    static void printTensorSummary(const Eigen::Tensor<double, 2> &tensor, const std::string &layerType, PropagationType propagationType);
    static void printProgress(int epoch, int epochs, int batch, int totalBatches, std::chrono::steady_clock::time_point start, double currentBatchLoss, ProgressLevel progressLevel);
    static void initializeCSV(const std::string &filename);
    static void appendToCSV(const std::string &filename, int epoch, double trainAcc, double trainLoss, double testAcc, double testLoss, const std::string &elralesState);
};

#endif // NNLOGGER_HPP
