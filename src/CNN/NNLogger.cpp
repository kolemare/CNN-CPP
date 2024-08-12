#include "NNLogger.hpp"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <sstream>

void NNLogger::printFullTensor(const Eigen::Tensor<double, 4> &tensor,
                               const std::string &layerType,
                               PropagationType propagationType)
{
    std::cout << "-------------------------------------------------------------------" << std::endl;
    switch (propagationType)
    {
    case PropagationType::FORWARD:
        std::cout << layerType << " Forward pass:\n";
        break;
    case PropagationType::BACK:
        std::cout << layerType << " Back pass:\n";
        break;
    default:
        break;
    }
    std::cout << tensor << std::endl;
    std::cout << "-------------------------------------------------------------------" << std::endl;
}

void NNLogger::printFullTensor(const Eigen::Tensor<double, 2> &tensor,
                               const std::string &layerType,
                               PropagationType propagationType)
{
    std::cout << "-------------------------------------------------------------------" << std::endl;
    switch (propagationType)
    {
    case PropagationType::FORWARD:
        std::cout << layerType << " Forward pass:\n";
        break;
    case PropagationType::BACK:
        std::cout << layerType << " Back pass:\n";
        break;
    default:
        break;
    }
    std::cout << tensor << std::endl;
    std::cout << "-------------------------------------------------------------------" << std::endl;
}

void NNLogger::printTensorSummary(const Eigen::Tensor<double, 4> &tensor,
                                  const std::string &layerType,
                                  PropagationType propagationType)
{
    std::cout << "-------------------------------------------------------------------" << std::endl;
    std::vector<double> tensorVec(tensor.size());
    std::copy(tensor.data(), tensor.data() + tensor.size(), tensorVec.begin());

    double mean = std::accumulate(tensorVec.begin(), tensorVec.end(), 0.0) / tensorVec.size();

    std::vector<double> diff(tensorVec.size());
    std::transform(tensorVec.begin(), tensorVec.end(), diff.begin(), [mean](double x)
                   { return x - mean; });

    std::vector<double> squaredDiff(diff.size());
    std::transform(diff.begin(), diff.end(), squaredDiff.begin(), [](double x)
                   { return x * x; });

    double variance = std::accumulate(squaredDiff.begin(), squaredDiff.end(), 0.0) / squaredDiff.size();
    double stddev = std::sqrt(variance);

    double minCoeff = *std::min_element(tensorVec.begin(), tensorVec.end());
    double maxCoeff = *std::max_element(tensorVec.begin(), tensorVec.end());

    double zeroPercentage = std::count(tensorVec.begin(), tensorVec.end(), 0.0) / static_cast<double>(tensorVec.size()) * 100.0;
    double negativeCount = std::count_if(tensorVec.begin(), tensorVec.end(), [](double x)
                                         { return x < 0.0; });
    double positiveCount = std::count_if(tensorVec.begin(), tensorVec.end(), [](double x)
                                         { return x > 0.0; });

    switch (propagationType)
    {
    case PropagationType::FORWARD:
        std::cout << layerType << " Forward pass summary:\n";
        break;
    case PropagationType::BACK:
        std::cout << layerType << " Back pass summary:\n";
        break;
    default:
        break;
    }

    std::cout << "Dimensions: " << tensor.dimension(0) << "x" << tensor.dimension(1) << "x" << tensor.dimension(2) << "x" << tensor.dimension(3) << "\n";
    std::cout << "Mean: " << mean << "\n";
    std::cout << "Standard Deviation: " << stddev << "\n";
    std::cout << "Min: " << minCoeff << "\n";
    std::cout << "Max: " << maxCoeff << "\n";
    std::cout << "Percentage of Zeros: " << zeroPercentage << "%\n";
    std::cout << "Number of Negative Values: " << negativeCount << "\n";
    std::cout << "Number of Positive Values: " << positiveCount << "\n\n";
    std::cout << "-------------------------------------------------------------------" << std::endl;
}

void NNLogger::printTensorSummary(const Eigen::Tensor<double, 2> &tensor,
                                  const std::string &layerType,
                                  PropagationType propagationType)
{
    std::cout << "-------------------------------------------------------------------" << std::endl;
    std::vector<double> tensorVec(tensor.size());
    std::copy(tensor.data(), tensor.data() + tensor.size(), tensorVec.begin());

    double mean = std::accumulate(tensorVec.begin(), tensorVec.end(), 0.0) / tensorVec.size();

    std::vector<double> diff(tensorVec.size());
    std::transform(tensorVec.begin(), tensorVec.end(), diff.begin(), [mean](double x)
                   { return x - mean; });

    std::vector<double> squaredDiff(diff.size());
    std::transform(diff.begin(), diff.end(), squaredDiff.begin(), [](double x)
                   { return x * x; });

    double variance = std::accumulate(squaredDiff.begin(), squaredDiff.end(), 0.0) / squaredDiff.size();
    double stddev = std::sqrt(variance);

    double minCoeff = *std::min_element(tensorVec.begin(), tensorVec.end());
    double maxCoeff = *std::max_element(tensorVec.begin(), tensorVec.end());

    double zeroPercentage = std::count(tensorVec.begin(), tensorVec.end(), 0.0) / static_cast<double>(tensorVec.size()) * 100.0;
    double negativeCount = std::count_if(tensorVec.begin(), tensorVec.end(), [](double x)
                                         { return x < 0.0; });
    double positiveCount = std::count_if(tensorVec.begin(), tensorVec.end(), [](double x)
                                         { return x > 0.0; });

    switch (propagationType)
    {
    case PropagationType::FORWARD:
        std::cout << layerType << " Forward pass summary:\n";
        break;
    case PropagationType::BACK:
        std::cout << layerType << " Back pass summary:\n";
        break;
    default:
        break;
    }

    std::cout << "Dimensions: " << tensor.dimension(0) << "x" << tensor.dimension(1) << "\n";
    std::cout << "Mean: " << mean << "\n";
    std::cout << "Standard Deviation: " << stddev << "\n";
    std::cout << "Min: " << minCoeff << "\n";
    std::cout << "Max: " << maxCoeff << "\n";
    std::cout << "Percentage of Zeros: " << zeroPercentage << "%\n";
    std::cout << "Number of Negative Values: " << negativeCount << "\n";
    std::cout << "Number of Positive Values: " << positiveCount << "\n\n";
    std::cout << "-------------------------------------------------------------------" << std::endl;
}

void NNLogger::printProgress(int epoch,
                             int epochs,
                             int batch,
                             int totalBatches,
                             std::chrono::steady_clock::time_point start,
                             double currentBatchLoss,
                             ProgressLevel progressLevel)
{
    if (progressLevel == ProgressLevel::None)
    {
        return;
    }

    static double cumulative_loss = 0.0;
    static int total_batches_completed = 0;

    cumulative_loss += currentBatchLoss;
    total_batches_completed++;

    double average_loss = cumulative_loss / total_batches_completed;

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
    double progress = static_cast<double>(batch + 1) / totalBatches;
    int barWidth = 50;
    int pos = static_cast<int>(barWidth * progress);

    double timePerBatch = elapsed / static_cast<double>(total_batches_completed);
    double remainingTime = (totalBatches * epochs - total_batches_completed) * timePerBatch;

    double overallProgress = static_cast<double>(epoch * totalBatches + batch + 1) / (totalBatches * epochs);
    int overallPos = static_cast<int>(barWidth * overallProgress);

    std::ostringstream oss;
    oss << "-------------------------------------------------------------------" << std::endl;

    std::ostringstream epochProgress;
    epochProgress << "Epoch " << epoch + 1 << "/" << epochs << " | Batch " << batch + 1 << "/" << totalBatches << " [";
    for (int i = 0; i < barWidth; ++i)
    {
        if (i < pos)
            epochProgress << "=";
        else if (i == pos)
            epochProgress << ">";
        else
            epochProgress << " ";
    }
    epochProgress << "] " << int(progress * 100.0) << "%\n";

    oss << epochProgress.str();

    std::ostringstream overallProgressLabel;
    overallProgressLabel << "Overall Progress: ";
    int labelLength = overallProgressLabel.str().length();

    // Dynamically adjust the overall progress label length to match epoch progress length
    std::string epochPrefix = "Epoch " + std::to_string(epoch + 1) + "/" + std::to_string(epochs) + " | Batch " + std::to_string(batch + 1) + "/" + std::to_string(totalBatches);
    while (overallProgressLabel.str().length() < epochPrefix.length())
    {
        overallProgressLabel << " ";
    }

    overallProgressLabel << " [";
    for (int i = 0; i < barWidth; ++i)
    {
        if (i < overallPos)
            overallProgressLabel << "=";
        else if (i == overallPos)
            overallProgressLabel << ">";
        else
            overallProgressLabel << " ";
    }
    overallProgressLabel << "] " << int(overallProgress * 100.0) << "%\n";

    oss << overallProgressLabel.str();

    if (progressLevel == ProgressLevel::ProgressTime || progressLevel == ProgressLevel::Time)
    {
        auto formatTime = [](double seconds)
        {
            int h = static_cast<int>(seconds) / 3600;
            int m = (static_cast<int>(seconds) % 3600) / 60;
            int s = static_cast<int>(seconds) % 60;
            std::ostringstream oss;
            oss << std::setw(2) << std::setfill('0') << h << "H:"
                << std::setw(2) << std::setfill('0') << m << "M:"
                << std::setw(2) << std::setfill('0') << s << "S";
            return oss.str();
        };

        oss << "Elapsed: " << formatTime(elapsed) << "\n";
        oss << "ETA: " << formatTime(remainingTime) << "\n";
    }
    oss << "Loss: " << average_loss << "\n";
    oss << "-------------------------------------------------------------------" << std::endl;
    std::cout << oss.str() << std::flush;
}

void NNLogger::initializeCSV(const std::string &filename)
{
    // Create the logs directory if it doesn't exist
    std::filesystem::create_directory("logs");

    // Construct the full path to the CSV file
    std::string fullPath = "logs/" + filename;

    // Check if cnn.csv already exists and remove it
    if (std::filesystem::exists(fullPath))
    {
        std::filesystem::remove(fullPath);
    }

    // Open the CSV file for writing
    std::ofstream csvFile(fullPath, std::ios::out);
    if (!csvFile.is_open())
    {
        throw std::runtime_error("Unable to open file for CSV initialization.");
    }

    // Write the headers
    csvFile << "epoch_num;training_accuracy;training_loss;testing_accuracy;testing_loss;elrales\n";
    csvFile.close();
}

void NNLogger::appendToCSV(const std::string &filename,
                           int epoch,
                           double trainAcc,
                           double trainLoss,
                           double testAcc,
                           double testLoss,
                           const std::string &elralesState)
{
    std::ofstream csvFile(filename, std::ios::app);
    if (!csvFile.is_open())
    {
        throw std::runtime_error("Unable to open file for appending CSV data.");
    }

    // Append the epoch data
    csvFile << epoch << ";" << trainAcc << ";" << trainLoss << ";" << testAcc << ";" << testLoss << ";" << elralesState << "\n";
    csvFile.close();
}
