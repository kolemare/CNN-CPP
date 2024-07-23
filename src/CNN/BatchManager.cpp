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

    batchImages.resize(batchSize, imageChannels, imageHeight, imageWidth); // Ensure the batch size is constant
    batchLabels.resize(batchSize, categories.size());
    batchLabels.setZero();

    // Fill batch with images from dataset
    for (int i = 0; i < currentBatchSize; ++i)
    {
        cv::Mat &image = *allImages[startIndex + i];
        for (int h = 0; h < imageHeight; ++h)
        {
            for (int w = 0; w < imageWidth; ++w)
            {
                for (int c = 0; c < imageChannels; ++c)
                {
                    // Normalize the pixel value
                    batchImages(i, c, h, w) = static_cast<double>(image.at<cv::Vec3b>(h, w)[c]) / 255.0;
                }
            }
        }

        // One-hot encode the label
        int labelIndex = std::distance(categories.begin(), std::find(categories.begin(), categories.end(), allLabels[startIndex + i]));
        batchLabels(i, labelIndex) = 1;
    }

    // Fill remaining slots with random images from the training dataset
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, allImages.size() - 1);

    for (int i = currentBatchSize; i < batchSize; ++i)
    {
        int randomIndex = dis(gen);
        cv::Mat &image = *allImages[randomIndex];
        for (int h = 0; h < imageHeight; ++h)
        {
            for (int w = 0; w << imageWidth; ++w)
            {
                for (int c = 0; c < imageChannels; ++c)
                {
                    // Normalize the pixel value
                    batchImages(i, c, h, w) = static_cast<double>(image.at<cv::Vec3b>(h, w)[c]) / 255.0;
                }
            }
        }

        // One-hot encode the label
        int labelIndex = std::distance(categories.begin(), std::find(categories.begin(), categories.end(), allLabels[randomIndex]));
        batchLabels(i, labelIndex) = 1;
    }

    currentBatchIndex++;
    return true;
}

size_t BatchManager::getTotalBatches() const
{
    return totalBatches;
}
