#include "BatchManager.hpp"
#include <algorithm>
#include <random>
#include <iostream>
#include <opencv2/core/eigen.hpp>

BatchManager::BatchManager(const ImageContainer &imageContainer, int batchSize, const std::vector<std::string> &categories, BatchType batchType)
    : imageContainer(imageContainer), batchSize(batchSize), categories(categories), currentBatchIndex(0), batchType(batchType)
{
    initializeBatches();
}

void BatchManager::initializeBatches()
{
    for (const auto &category : categories)
    {
        std::vector<std::shared_ptr<cv::Mat>> images;
        std::vector<std::string> labels;

        if (batchType == BatchType::Training)
        {
            images = imageContainer.getTrainingImagesByCategory(category);
        }
        else if (batchType == BatchType::Testing)
        {
            images = imageContainer.getTestImagesByCategory(category);
        }
        labels.insert(labels.end(), images.size(), category);

        allImages.insert(allImages.end(), images.begin(), images.end());
        allLabels.insert(allLabels.end(), labels.begin(), labels.end());
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

bool BatchManager::getNextBatch(Eigen::Tensor<double, 4> &batchImages, Eigen::Tensor<int, 2> &batchLabels)
{
    if (currentBatchIndex >= totalBatches)
    {
        currentBatchIndex = 0;
        return false;
    }

    int startIndex = currentBatchIndex * batchSize;
    int endIndex = std::min(startIndex + batchSize, static_cast<int>(allImages.size()));

    int currentBatchSize = endIndex - startIndex;
    int imageHeight = allImages[0]->rows;
    int imageWidth = allImages[0]->cols;
    int imageChannels = allImages[0]->channels();

    batchImages.resize(currentBatchSize, imageChannels, imageHeight, imageWidth);
    batchLabels.resize(currentBatchSize, categories.size());
    batchLabels.setZero();

    for (int i = 0; i < currentBatchSize; ++i)
    {
        cv::Mat &image = *allImages[startIndex + i];
        cv::Mat reshapedImage = image.reshape(1, imageHeight * imageWidth);
        Eigen::MatrixXd eigenImage;
        cv::cv2eigen(reshapedImage, eigenImage);
        eigenImage = eigenImage.cast<double>(); // Use the image directly without further normalization

        // Copy eigenImage data to batchImages tensor
        for (int c = 0; c < imageChannels; ++c)
        {
            for (int h = 0; h < imageHeight; ++h)
            {
                for (int w = 0; w < imageWidth; ++w)
                {
                    batchImages(i, c, h, w) = eigenImage(h, w * imageChannels + c);
                }
            }
        }

        // One-hot encode the label
        int labelIndex = std::distance(categories.begin(), std::find(categories.begin(), categories.end(), allLabels[startIndex + i]));
        batchLabels(i, labelIndex) = 1;
    }

    currentBatchIndex++;
    return true;
}

size_t BatchManager::getTotalBatches() const
{
    return totalBatches;
}
