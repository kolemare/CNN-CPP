#include "BatchManager.hpp"
#include <algorithm>
#include <random>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

BatchManager::BatchManager(const ImageContainer &imageContainer, int batchSize, BatchType batchType)
    : imageContainer(imageContainer), batchSize(batchSize), currentBatchIndex(0), batchType(batchType)
{
    categories = imageContainer.getUniqueLabels();
    initializeBatches();
}

void BatchManager::initializeBatches()
{
    categoryImages.clear();
    categoryLabels.clear();
    allImages.clear();
    allLabels.clear();
    originalAllImages.clear();
    originalAllLabels.clear();

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

        categoryImages[category] = images;
        categoryLabels[category] = labels;
        allImages.insert(allImages.end(), images.begin(), images.end());
        allLabels.insert(allLabels.end(), labels.begin(), labels.end());
    }

    // Make a copy of all images and labels for potential reuse in incomplete batches
    originalAllImages = allImages;
    originalAllLabels = allLabels;

    totalBatches = (allImages.size() + batchSize - 1) / batchSize;
    shuffleDataset();
}

void BatchManager::shuffleDataset()
{
    auto seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed); // Use steady_clock to seed the generator
    std::vector<int> indices(allImages.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen); // Use the seeded generator

    std::vector<std::shared_ptr<cv::Mat>> shuffledAllImages(allImages.size());
    std::vector<std::string> shuffledAllLabels(allLabels.size());

    for (size_t i = 0; i < indices.size(); ++i)
    {
        shuffledAllImages[i] = allImages[indices[i]];
        shuffledAllLabels[i] = allLabels[indices[i]];
    }

    allImages = shuffledAllImages;
    allLabels = shuffledAllLabels;
}

void BatchManager::interBatchShuffle(Eigen::Tensor<double, 4> &batchImages, Eigen::Tensor<int, 2> &batchLabels)
{
    auto seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    std::vector<int> indices(batchSize);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);

    Eigen::Tensor<double, 4> shuffledBatchImages(batchSize, batchImages.dimension(1), batchImages.dimension(2), batchImages.dimension(3));
    Eigen::Tensor<int, 2> shuffledBatchLabels(batchSize, batchLabels.dimension(1));

    for (int i = 0; i < batchSize; ++i)
    {
        shuffledBatchImages.chip(i, 0) = batchImages.chip(indices[i], 0);
        shuffledBatchLabels.chip(i, 0) = batchLabels.chip(indices[i], 0);
    }

    batchImages = shuffledBatchImages;
    batchLabels = shuffledBatchLabels;
}

void BatchManager::saveBatchImages(const Eigen::Tensor<double, 4> &batchImages, const Eigen::Tensor<int, 2> &batchLabels, int batchIndex)
{
    std::string batchDir = "batch" + std::to_string(batchIndex);
    fs::create_directory(batchDir);

    for (int i = 0; i < batchSize; ++i)
    {
        int labelIndex = -1;
        for (int j = 0; j < batchLabels.dimension(1); ++j)
        {
            if (batchLabels(i, j) == 1)
            {
                labelIndex = j;
                break;
            }
        }

        std::string category = categories[labelIndex];
        std::string categoryDir = batchDir + "/" + category;
        fs::create_directory(categoryDir);

        cv::Mat image(batchImages.dimension(2), batchImages.dimension(3), CV_32FC3);
        for (int h = 0; h < batchImages.dimension(2); ++h)
        {
            for (int w = 0; w < batchImages.dimension(3); ++w)
            {
                for (int c = 0; c < batchImages.dimension(1); ++c)
                {
                    image.at<cv::Vec3f>(h, w)[c] = static_cast<float>(batchImages(i, c, h, w));
                }
            }
        }

        image.convertTo(image, CV_32F, 255.0);

        std::string imagePath = categoryDir + "/image" + std::to_string(i) + ".jpg";
        cv::imwrite(imagePath, image);
    }
}

bool BatchManager::getNextBatch(Eigen::Tensor<double, 4> &batchImages, Eigen::Tensor<int, 2> &batchLabels)
{
    if (currentBatchIndex >= totalBatches)
    {
        currentBatchIndex = 0;

        // Refill the vectors from the original copies if they are empty
        if (allImages.empty() || allLabels.empty())
        {
            allImages = originalAllImages;
            allLabels = originalAllLabels;
        }

        shuffleDataset(); // Reshuffle the dataset at the start of a new epoch
        return false;
    }

    int batchIndex = 0;
    int imageHeight = allImages[0]->rows;
    int imageWidth = allImages[0]->cols;
    int imageChannels = allImages[0]->channels();

    batchImages.resize(batchSize, imageChannels, imageHeight, imageWidth); // Ensure the batch size is constant
    batchLabels.resize(batchSize, categories.size());
    batchLabels.setZero();

    // Calculate the number of images to add from each category to maintain balance
    int numCategories = static_cast<int>(categories.size());
    int imagesPerCategory = batchSize / numCategories;

    for (const auto &category : categories)
    {
        auto &images = categoryImages[category];
        auto &labels = categoryLabels[category];

        int numImagesToAdd = std::min(static_cast<int>(images.size()), imagesPerCategory);
        for (int i = 0; i < numImagesToAdd; ++i)
        {
            if (batchIndex >= batchSize)
                break;

            cv::Mat &image = *images[i];
            for (int h = 0; h < imageHeight; ++h)
            {
                for (int w = 0; w < imageWidth; ++w)
                {
                    for (int c = 0; c < imageChannels; ++c)
                    {
                        batchImages(batchIndex, c, h, w) = image.at<cv::Vec3f>(h, w)[c];
                    }
                }
            }

            // One-hot encode the label
            int labelIndex = std::distance(categories.begin(), std::find(categories.begin(), categories.end(), labels[i]));
            batchLabels(batchIndex, labelIndex) = 1;
            batchIndex++;
        }

        // Remove used images and labels
        if (numImagesToAdd > 0)
        {
            images.erase(images.begin(), images.begin() + numImagesToAdd);
            labels.erase(labels.begin(), labels.begin() + numImagesToAdd);
        }
    }

    // If the batch is not full, fill remaining slots with random images from the original dataset copies
    while (batchIndex < batchSize)
    {
        if (allImages.empty())
        {
            allImages = originalAllImages;
            allLabels = originalAllLabels;
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, allImages.size() - 1);
        int randomIndex = dis(gen);
        cv::Mat &image = *allImages[randomIndex];
        for (int h = 0; h < imageHeight; ++h)
        {
            for (int w = 0; w < imageWidth; ++w)
            {
                for (int c = 0; c < imageChannels; ++c)
                {
                    batchImages(batchIndex, c, h, w) = image.at<cv::Vec3f>(h, w)[c];
                }
            }
        }

        // One-hot encode the label
        int labelIndex = std::distance(categories.begin(), std::find(categories.begin(), categories.end(), allLabels[randomIndex]));
        batchLabels(batchIndex, labelIndex) = 1;
        batchIndex++;
    }

    // Shuffle within the batch
    interBatchShuffle(batchImages, batchLabels);

#ifdef SAVE_BATCHES

    // Save the batch images to disk
    saveBatchImages(batchImages, batchLabels, currentBatchIndex);

#endif

    currentBatchIndex++;
    return true;
}

size_t BatchManager::getTotalBatches() const
{
    return totalBatches;
}
