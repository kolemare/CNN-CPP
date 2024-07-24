#ifndef BATCHMANAGER_HPP
#define BATCHMANAGER_HPP

#include "ImageContainer.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <string>

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
    bool getNextBatch(Eigen::Tensor<double, 4> &batchImages, Eigen::Tensor<int, 2> &batchLabels);
    size_t getTotalBatches() const;

private:
    const ImageContainer &imageContainer;
    int batchSize;
    std::vector<std::string> categories;
    std::vector<std::shared_ptr<cv::Mat>> allImages;
    std::vector<std::string> allLabels;
    size_t currentBatchIndex;
    size_t totalBatches;
    BatchType batchType;
};

#endif // BATCHMANAGER_HPP
