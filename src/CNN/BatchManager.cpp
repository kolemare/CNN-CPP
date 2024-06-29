#include "BatchManager.hpp"
#include <algorithm>
#include <random>

BatchManager::BatchManager(const ImageContainer &imageContainer, int batchSize, const std::vector<std::string> &categories)
    : imageContainer(imageContainer), batchSize(batchSize), categories(categories), currentBatchIndex(0)
{
    initializeBatches();
}

void BatchManager::initializeBatches()
{
    for (const auto &category : categories)
    {
        auto trainingImages = imageContainer.getTrainingImagesByCategory(category);
        allImages.insert(allImages.end(), trainingImages.begin(), trainingImages.end());

        auto testImages = imageContainer.getTestImagesByCategory(category);
        allImages.insert(allImages.end(), testImages.begin(), testImages.end());

        allLabels.insert(allLabels.end(), trainingImages.size(), category);
        allLabels.insert(allLabels.end(), testImages.size(), category);
    }

    // Shuffle images and labels
    std::vector<int> indices(allImages.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine{});

    std::vector<std::shared_ptr<cv::Mat>> shuffledImages(allImages.size());
    std::vector<std::string> shuffledLabels(allLabels.size());

    for (size_t i = 0; i < indices.size(); ++i)
    {
        shuffledImages[i] = allImages[indices[i]];
        shuffledLabels[i] = allLabels[indices[i]];
    }

    allImages = shuffledImages;
    allLabels = shuffledLabels;

    totalBatches = (allImages.size() + batchSize - 1) / batchSize;
}

bool BatchManager::getNextBatch(Eigen::MatrixXd &batchImages, Eigen::MatrixXd &batchLabels)
{
    if (currentBatchIndex >= totalBatches)
    {
        currentBatchIndex = 0;
        return false;
    }

    int startIndex = currentBatchIndex * batchSize;
    int endIndex = std::min(startIndex + batchSize, static_cast<int>(allImages.size()));

    int currentBatchSize = endIndex - startIndex;
    int imageSize = allImages[0]->rows * allImages[0]->cols * allImages[0]->channels();

    batchImages.resize(currentBatchSize, imageSize);
    batchLabels.resize(currentBatchSize, 1);

    for (int i = 0; i < currentBatchSize; ++i)
    {
        cv::Mat &image = *allImages[startIndex + i];
        cv::Mat flatImage = image.reshape(1, 1);

        Eigen::Map<Eigen::MatrixXf> eigenImage(flatImage.ptr<float>(), 1, imageSize); // Use float type for Eigen::Matrix mapping
        batchImages.row(i) = eigenImage.cast<double>();

        batchLabels(i, 0) = std::find(categories.begin(), categories.end(), allLabels[startIndex + i]) - categories.begin();
    }

    currentBatchIndex++;
    return true;
}
