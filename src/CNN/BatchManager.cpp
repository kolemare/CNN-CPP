#include "BatchManager.hpp"
#include <algorithm>
#include <random>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <filesystem> // For creating directories
#include <fstream>    // For saving Eigen matrices to files

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
            labels.insert(labels.end(), images.size(), category);
        }
        else if (batchType == BatchType::Testing)
        {
            images = imageContainer.getTestImagesByCategory(category);
            labels.insert(labels.end(), images.size(), category);
        }

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

    // Create directories for saving images
    std::filesystem::create_directories("testing/original");
    std::filesystem::create_directories("testing/matrix");

    for (int i = 0; i < currentBatchSize; ++i)
    {
        cv::Mat &image = *allImages[startIndex + i];

        // Ensure the image is continuous
        if (!image.isContinuous())
        {
            image = image.clone();
        }

        // Save original image (scaled back)
        cv::Mat originalImageScaled;
        image.convertTo(originalImageScaled, CV_8UC3, 255.0);
        std::string originalImagePath = "testing/original/original_image_" + std::to_string(currentBatchIndex) + "_" + std::to_string(i) + ".png";
        cv::imwrite(originalImagePath, originalImageScaled);

        // Convert cv::Mat to Eigen::MatrixXd using OpenCV function
        Eigen::MatrixXf eigenImage;
        cv::cv2eigen(image.reshape(1, 1), eigenImage);

        batchImages.row(i) = eigenImage.cast<double>();

        // Convert Eigen::MatrixXd back to cv::Mat for visual inspection using OpenCV function
        cv::Mat restoredImage;
        cv::eigen2cv(eigenImage, restoredImage);
        restoredImage = restoredImage.reshape(3, 32);           // Reshape back to original size
        restoredImage.convertTo(restoredImage, CV_8UC3, 255.0); // Scale back to 0-255

        // Save the restored image
        std::string matrixImagePath = "testing/matrix/matrix_image_" + std::to_string(currentBatchIndex) + "_" + std::to_string(i) + ".png";
        cv::imwrite(matrixImagePath, restoredImage);

        batchLabels(i, 0) = std::find(categories.begin(), categories.end(), allLabels[startIndex + i]) - categories.begin();
    }

    currentBatchIndex++;
    throw std::runtime_error("Batch saved for inspection.");
    return true;
}

size_t BatchManager::getTotalBatches() const
{
    return totalBatches;
}
