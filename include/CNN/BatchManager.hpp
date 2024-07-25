#ifndef BATCHMANAGER_HPP
#define BATCHMANAGER_HPP

// #define SAVE_BATCHES

#include "ImageContainer.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <string>
#include <unordered_map>

class BatchManager
{
public:
    enum class BatchType
    {
        Training,
        Testing
    };

    BatchManager(const ImageContainer &imageContainer, int batchSize, BatchType batchType);

    void initializeBatches();
    void shuffleDataset();
    void interBatchShuffle(Eigen::Tensor<double, 4> &batchImages, Eigen::Tensor<int, 2> &batchLabels);
    void saveBatchImages(const Eigen::Tensor<double, 4> &batchImages, const Eigen::Tensor<int, 2> &batchLabels, int batchIndex);
    bool getNextBatch(Eigen::Tensor<double, 4> &batchImages, Eigen::Tensor<int, 2> &batchLabels);
    size_t getTotalBatches() const;

private:
    const ImageContainer &imageContainer;
    int batchSize;
    std::vector<std::string> categories;
    std::unordered_map<std::string, std::vector<std::shared_ptr<cv::Mat>>> categoryImages;
    std::unordered_map<std::string, std::vector<std::string>> categoryLabels;
    std::vector<std::shared_ptr<cv::Mat>> allImages;
    std::vector<std::string> allLabels;
    std::vector<std::shared_ptr<cv::Mat>> originalAllImages;
    std::vector<std::string> originalAllLabels;
    size_t currentBatchIndex;
    size_t totalBatches;
    BatchType batchType;
};

#endif // BATCHMANAGER_HPP
