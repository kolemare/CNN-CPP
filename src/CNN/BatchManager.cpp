#include "BatchManager.hpp"
#include <algorithm>
#include <random>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

/**
 * @brief Constructs a BatchManager object for managing image batches.
 *
 * Initializes the BatchManager with image data, batch size, and type (training or testing).
 *
 * @param imageContainer The container with images and labels.
 * @param batchSize The size of each batch.
 * @param batchType The type of batch (training or testing).
 */
BatchManager::BatchManager(const ImageContainer &imageContainer,
                           int batchSize,
                           BatchType batchType)
    : imageContainer(imageContainer)
{
    this->batchSize = batchSize;
    this->currentBatchIndex = 0;
    this->batchType = batchType;

    // Get unique categories from the image container
    categories = imageContainer.getUniqueLabels();

    // Initialize batches
    initializeBatches();
}

/**
 * @brief Initializes batches by categorizing images and labels.
 *
 * Clears existing data and populates images and labels based on category and batch type.
 */
void BatchManager::initializeBatches()
{
    // Clear existing data
    categoryImages.clear();
    categoryLabels.clear();
    allImages.clear();
    allLabels.clear();
    originalAllImages.clear();
    originalAllLabels.clear();

    // Populate images and labels based on category and batch type
    for (const auto &category : categories)
    {
        std::vector<std::shared_ptr<cv::Mat>> images;
        std::vector<std::string> labels;

        // Get images based on the batch type (training or testing)
        if (batchType == BatchType::Training)
        {
            images = imageContainer.getTrainingImagesByCategory(category);
        }
        else if (batchType == BatchType::Testing)
        {
            images = imageContainer.getTestImagesByCategory(category);
        }

        // Assign labels to each image in the category
        labels.insert(labels.end(), images.size(), category);

        // Store categorized images and labels
        categoryImages[category] = images;
        categoryLabels[category] = labels;

        // Store all images and labels together
        allImages.insert(allImages.end(), images.begin(), images.end());
        allLabels.insert(allLabels.end(), labels.begin(), labels.end());
    }

    // Create copies of all images and labels for potential reuse in incomplete batches
    originalAllImages = allImages;
    originalAllLabels = allLabels;

    // Calculate total number of batches
    totalBatches = (allImages.size() + batchSize - 1) / batchSize;

    // Shuffle the dataset to randomize the order of batches
    shuffleDataset();
}

/**
 * @brief Shuffles the dataset using a random seed.
 *
 * Randomly shuffles the images and labels in the dataset.
 */
void BatchManager::shuffleDataset()
{
    // Use steady_clock to seed the random generator
    auto seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);

    // Create indices for shuffling
    std::vector<int> indices(allImages.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Shuffle the indices
    std::shuffle(indices.begin(), indices.end(), gen);

    // Shuffle images and labels using the shuffled indices
    std::vector<std::shared_ptr<cv::Mat>> shuffledAllImages(allImages.size());
    std::vector<std::string> shuffledAllLabels(allLabels.size());

    for (size_t i = 0; i < indices.size(); ++i)
    {
        shuffledAllImages[i] = allImages[indices[i]];
        shuffledAllLabels[i] = allLabels[indices[i]];
    }

    // Update images and labels with the shuffled versions
    allImages = shuffledAllImages;
    allLabels = shuffledAllLabels;
}

/**
 * @brief Shuffles images and labels within a batch.
 *
 * @param batchImages A 4D tensor containing the batch images to be shuffled.
 * @param batchLabels A 2D tensor containing the batch labels to be shuffled.
 */
void BatchManager::interBatchShuffle(Eigen::Tensor<double, 4> &batchImages,
                                     Eigen::Tensor<int, 2> &batchLabels)
{
    // Use steady_clock to seed the random generator
    auto seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);

    // Create indices for shuffling within the batch
    std::vector<int> indices(batchSize);
    std::iota(indices.begin(), indices.end(), 0);

    // Shuffle the indices
    std::shuffle(indices.begin(), indices.end(), gen);

    // Create tensors to store shuffled images and labels
    Eigen::Tensor<double, 4> shuffledBatchImages(batchSize, batchImages.dimension(1), batchImages.dimension(2), batchImages.dimension(3));
    Eigen::Tensor<int, 2> shuffledBatchLabels(batchSize, batchLabels.dimension(1));

    // Shuffle images and labels using the shuffled indices
    for (int i = 0; i < batchSize; ++i)
    {
        shuffledBatchImages.chip(i, 0) = batchImages.chip(indices[i], 0);
        shuffledBatchLabels.chip(i, 0) = batchLabels.chip(indices[i], 0);
    }

    // Update images and labels with the shuffled versions
    batchImages = shuffledBatchImages;
    batchLabels = shuffledBatchLabels;
}

/**
 * @brief Saves batch images to disk for debugging or analysis.
 *
 * Saves the images of the current batch to disk, organized by category.
 *
 * @param batchImages A 4D tensor containing the batch images.
 * @param batchLabels A 2D tensor containing the batch labels.
 * @param batchIndex The index of the current batch.
 */
void BatchManager::saveBatchImages(const Eigen::Tensor<double, 4> &batchImages,
                                   const Eigen::Tensor<int, 2> &batchLabels,
                                   int batchIndex)
{
    // Create a directory for the current batch
    std::string batchDir = "batch" + std::to_string(batchIndex);
    fs::create_directory(batchDir);

    // Iterate over each image in the batch
    for (int i = 0; i < batchSize; ++i)
    {
        // Determine the label index for the image
        int labelIndex = -1;
        for (int j = 0; j < batchLabels.dimension(1); ++j)
        {
            if (batchLabels(i, j) == 1)
            {
                labelIndex = j;
                break;
            }
        }

        // Get the category name corresponding to the label index
        std::string category = categories[labelIndex];
        std::string categoryDir = batchDir + "/" + category;
        fs::create_directory(categoryDir);

        // Convert the Eigen tensor to OpenCV Mat format
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

        // Convert the image to 8-bit format and save to disk
        image.convertTo(image, CV_32F, 255.0);
        std::string imagePath = categoryDir + "/image" + std::to_string(i) + ".jpg";
        cv::imwrite(imagePath, image);
    }
}

/**
 * @brief Retrieves the next batch of images and labels.
 *
 * Gets the next batch of images and labels for training or testing. If all batches
 * have been processed, it reshuffles the dataset and starts a new epoch.
 *
 * @param batchImages A 4D tensor to store the next batch of images.
 * @param batchLabels A 2D tensor to store the next batch of labels.
 * @return True if a new batch is available, false if starting a new epoch.
 */
bool BatchManager::getNextBatch(Eigen::Tensor<double, 4> &batchImages,
                                Eigen::Tensor<int, 2> &batchLabels)
{
    // Check if all batches have been processed
    if (currentBatchIndex >= totalBatches)
    {
        currentBatchIndex = 0;

        // Refill vectors from original copies if they are empty
        if (allImages.empty() || allLabels.empty())
        {
            allImages = originalAllImages;
            allLabels = originalAllLabels;
        }

        // Reshuffle the dataset for a new epoch
        shuffleDataset();
        return false;
    }

    // Initialize batch size and image dimensions
    int batchIndex = 0;
    int imageHeight = allImages[0]->rows;
    int imageWidth = allImages[0]->cols;
    int imageChannels = allImages[0]->channels();

    // Resize batch tensors
    batchImages.resize(batchSize, imageChannels, imageHeight, imageWidth);
    batchLabels.resize(batchSize, categories.size());
    batchLabels.setZero(); // Initialize labels to zero for one-hot encoding

    // Calculate the number of images to add from each category to maintain balance
    int numCategories = static_cast<int>(categories.size());
    int imagesPerCategory = batchSize / numCategories;

    // Fill the batch with images from each category
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

    // If the batch is not full, fill remaining slots with random images from original dataset copies
    while (batchIndex < batchSize)
    {
        if (allImages.empty())
        {
            allImages = originalAllImages;
            allLabels = originalAllLabels;
        }

        // Select a random image and its label
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

    // Shuffle the batch to randomize the order
    interBatchShuffle(batchImages, batchLabels);

#ifdef SAVE_BATCHES
    // Save the batch images to disk if the SAVE_BATCHES flag is set
    saveBatchImages(batchImages, batchLabels, currentBatchIndex);
#endif

    // Increment the current batch index
    currentBatchIndex++;
    return true;
}

/**
 * @brief Gets the total number of batches in the dataset.
 *
 * @return The total number of batches.
 */
size_t BatchManager::getTotalBatches() const
{
    return totalBatches;
}
