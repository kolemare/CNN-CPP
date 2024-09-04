/*
MIT License
Copyright (c) 2024 Marko Kostić

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

This project is the CNN-CPP Framework. Usage of this code is free, and 
uploading and using the code is also free, with a humble request to mention 
the origin of the implementation, the author Marko Kostić, and the repository 
link: https://github.com/kolemare/CNN-CPP.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
*/

#include "BatchManager.hpp"

namespace fs = std::filesystem;

BatchManager::BatchManager(const ImageContainer &imageContainer,
                           int batchSize,
                           BatchType batchType,
                           BatchMode batchMode)
    : imageContainer(imageContainer)
{
    this->batchSize = batchSize;
    this->currentBatchIndex = 0;
    this->batchType = batchType;
    this->batchMode = batchMode;

    // Get unique categories from the image container
    categories = imageContainer.getUniqueLabels();

    // Map categories to indices
    for (size_t i = 0; i < categories.size(); ++i)
    {
        categoryToIndex[categories[i]] = i;
    }

    // Initialize batches
    initializeBatches();
}

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

        // Create the image filename with the encoding index
        std::string imageFilename = category + "_" + std::to_string(i) + "_" + std::to_string(labelIndex) + ".jpg";
        std::string imagePath = categoryDir + "/" + imageFilename;

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
        image.convertTo(image, CV_8UC3, 255.0 / imageContainer.getNormalizationScale());
        cv::imwrite(imagePath, image);
    }
}

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

    if (BatchMode::UniformDistribution == this->batchMode)
    {
        // Code for uniform distribution across categories
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

                // One-hot encode the label using categoryToIndex
                int labelIndex = categoryToIndex[labels[i]];
                batchLabels(batchIndex, labelIndex) = 1;
                batchIndex++;
            }

            if (numImagesToAdd > 0)
            {
                images.erase(images.begin(), images.begin() + numImagesToAdd);
                labels.erase(labels.begin(), labels.begin() + numImagesToAdd);
            }
        }
    }
    else if (BatchMode::ShuffleOnly == this->batchMode)
    {
        // Ensure we have enough images and labels left to fill the batch
        int remainingImages = static_cast<int>(allImages.size());
        int numImagesToAdd = std::min(remainingImages, batchSize);

        for (int i = 0; i < numImagesToAdd; ++i)
        {
            cv::Mat &image = *allImages[i];
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

            // One-hot encode the label using categoryToIndex
            int labelIndex = categoryToIndex[allLabels[i]];
            batchLabels(batchIndex, labelIndex) = 1;
            batchIndex++;
        }

        // Remove the used images and labels from the main list
        allImages.erase(allImages.begin(), allImages.begin() + numImagesToAdd);
        allLabels.erase(allLabels.begin(), allLabels.begin() + numImagesToAdd);
    }
    else
    {
        throw std::runtime_error("Unrecognized batch mode.");
    }

    // Fill remaining slots with random images from the original dataset copies if necessary
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

        // One-hot encode the label using categoryToIndex
        int labelIndex = categoryToIndex[allLabels[randomIndex]];
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

void BatchManager::loadSinglePredictionBatch()
{
    this->singlePredictionImages = imageContainer.getSinglePredictionImages(); // Initialize singlePredictionImages
}

std::vector<std::string> BatchManager::getSinglePredictionBatch(Eigen::Tensor<double, 4> &batchImages,
                                                                Eigen::Tensor<int, 2> &batchLabels)
{
    // Check if there are any images to process
    if (singlePredictionImages.empty())
    {
        return {};
    }

    // Determine image dimensions based on the first available image
    int imageHeight = singlePredictionImages.begin()->second->rows;
    int imageWidth = singlePredictionImages.begin()->second->cols;
    int imageChannels = singlePredictionImages.begin()->second->channels();

    // Resize batch tensors
    batchImages.resize(batchSize, imageChannels, imageHeight, imageWidth);
    batchLabels.resize(batchSize, categories.size());
    batchLabels.setZero(); // Initialize labels to zero for one-hot encoding

    int batchIndex = 0;
    std::vector<std::string> imageNames; // Store image names in the batch

    // Fill the batch with images from single prediction set
    for (auto it = singlePredictionImages.begin(); it != singlePredictionImages.end();)
    {
        if (batchIndex >= batchSize)
        {
            break; // Break if batch is filled
        }

        const std::string &imageName = it->first;
        cv::Mat &image = *(it->second);
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

        imageNames.push_back(imageName); // Store the image name
        batchIndex++;

        // Remove the image after processing
        it = singlePredictionImages.erase(it);
    }

    // Fill remaining slots with zeroes if necessary
    while (batchIndex < batchSize)
    {
        batchImages.chip(batchIndex, 0).setZero();
        batchLabels.chip(batchIndex, 0).setZero();
        batchIndex++;
    }

    return imageNames; // Return the image names filled in the batch
}

std::string BatchManager::getCategoryName(int index) const
{
    // Ensure the index is within bounds
    if (index >= 0 && index < categories.size())
    {
        return categories[index];
    }
    else
    {
        throw std::out_of_range("Index out of range for categories.");
    }
}

size_t BatchManager::getTotalBatches() const
{
    return totalBatches; // Return the total number of batches in the dataset
}
