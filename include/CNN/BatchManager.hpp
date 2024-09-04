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

#ifndef BATCHMANAGER_HPP
#define BATCHMANAGER_HPP

#include "Common.hpp"
#include "ImageContainer.hpp"

/**
 * @class BatchManager
 * @brief Manages batches of images and labels for training, testing, and single prediction.
 *
 * The BatchManager class provides functionality to manage and process batches of images and labels
 * for training and testing purposes. It handles shuffling of datasets and retrieval of batches.
 * BatchManager serves as "Input Layer" to the neural network.
 */
class BatchManager
{
public:
    /**
     * @brief Constructs a BatchManager object.
     *
     * Initializes the BatchManager with image data, batch size, batch type (training or testing) and batch mode (UniformDistribution or ShuffleOnly).
     *
     * @param imageContainer The container with images and labels.
     * @param batchSize The size of each batch.
     * @param batchType The type of batch (training or testing).
     * @param batchMode Organization of batch (UniformDistribution or ShuffleOnly).
     */
    BatchManager(const ImageContainer &imageContainer,
                 int batchSize,
                 BatchType batchType,
                 BatchMode batchMode);

    /**
     * @brief Initializes batches by categorizing images and labels.
     *
     * Clears existing data and populates images and labels based on category and batch type.
     */
    void initializeBatches();

    /**
     * @brief Shuffles the dataset using a random seed.
     *
     * Randomly shuffles the images and labels in the dataset.
     */
    void shuffleDataset();

    /**
     * @brief Shuffles images and labels within a batch.
     *
     * @param batchImages A 4D tensor containing the batch images to be shuffled.
     * @param batchLabels A 2D tensor containing the batch labels to be shuffled.
     */
    void interBatchShuffle(Eigen::Tensor<double, 4> &batchImages,
                           Eigen::Tensor<int, 2> &batchLabels);

    /**
     * @brief Saves batch images to disk for debugging or analysis.
     *
     * Saves the images of the current batch to disk, organized by category.
     *
     * @param batchImages A 4D tensor containing the batch images.
     * @param batchLabels A 2D tensor containing the batch labels.
     * @param batchIndex The index of the current batch.
     */
    void saveBatchImages(const Eigen::Tensor<double, 4> &batchImages,
                         const Eigen::Tensor<int, 2> &batchLabels,
                         int batchIndex);

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
    bool getNextBatch(Eigen::Tensor<double, 4> &batchImages,
                      Eigen::Tensor<int, 2> &batchLabels);

    /**
     * @brief Loads the batch for single predictions.
     *
     * Prepares single prediction images for processing.
     */
    void loadSinglePredictionBatch();

    /**
     * @brief Retrieves a batch of single prediction images and their categories.
     *
     * Creates a batch from the single prediction images and fills remaining space with zeroes.
     *
     * @param batchImages A 4D tensor to store the batch of images.
     * @param batchLabels A 2D tensor to store the batch of labels.
     * @return A vector of strings representing the names of images in the batch.
     */
    std::vector<std::string> getSinglePredictionBatch(Eigen::Tensor<double, 4> &batchImages,
                                                      Eigen::Tensor<int, 2> &batchLabels);

    /**
     * @brief Gets the category name for a given index.
     *
     * @param index The index of the category.
     * @return The name of the category.
     * @throws std::out_of_range if the index is invalid.
     */
    std::string getCategoryName(int index) const;

    /**
     * @brief Gets the total number of batches in the dataset.
     *
     * @return The total number of batches.
     */
    size_t getTotalBatches() const;

private:
    const ImageContainer &imageContainer;                                                  ///< The container with images and labels.
    int batchSize;                                                                         ///< The size of each batch.
    std::vector<std::string> categories;                                                   ///< The unique categories in the dataset.
    std::unordered_map<std::string, std::vector<std::shared_ptr<cv::Mat>>> categoryImages; ///< Images categorized by labels.
    std::unordered_map<std::string, std::vector<std::string>> categoryLabels;              ///< Labels categorized by image category.
    std::unordered_map<std::string, std::shared_ptr<cv::Mat>> singlePredictionImages;      ///< Map of image names to single prediction images.
    std::vector<std::shared_ptr<cv::Mat>> allImages;                                       ///< All images in the dataset.
    std::vector<std::string> allLabels;                                                    ///< All labels in the dataset.
    std::vector<std::shared_ptr<cv::Mat>> originalAllImages;                               ///< Original copy of all images for reuse.
    std::vector<std::string> originalAllLabels;                                            ///< Original copy of all labels for reuse.
    std::unordered_map<std::string, int> categoryToIndex;                                  ///< Mapping from category names to indices.
    size_t currentBatchIndex;                                                              ///< The index of the current batch.
    size_t totalBatches;                                                                   ///< The total number of batches.
    BatchType batchType;                                                                   ///< The type of batch (training or testing).
    BatchMode batchMode;                                                                   ///< Organization of batch (UniformDistribution or ShuffleOnly).
};

#endif // BATCHMANAGER_HPP
