#ifndef BATCHMANAGER_HPP
#define BATCHMANAGER_HPP

// #define SAVE_BATCHES

#include "Common.hpp"
#include "ImageContainer.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <string>
#include <unordered_map>

/**
 * @class BatchManager
 * @brief Manages batches of images and labels for training and testing.
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
     * Initializes the BatchManager with image data, batch size, and type (training or testing).
     *
     * @param imageContainer The container with images and labels.
     * @param batchSize The size of each batch.
     * @param batchType The type of batch (training or testing).
     */
    BatchManager(const ImageContainer &imageContainer,
                 int batchSize,
                 BatchType batchType);

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
    std::vector<std::shared_ptr<cv::Mat>> allImages;                                       ///< All images in the dataset.
    std::vector<std::string> allLabels;                                                    ///< All labels in the dataset.
    std::vector<std::shared_ptr<cv::Mat>> originalAllImages;                               ///< Original copy of all images for reuse.
    std::vector<std::string> originalAllLabels;                                            ///< Original copy of all labels for reuse.
    size_t currentBatchIndex;                                                              ///< The index of the current batch.
    size_t totalBatches;                                                                   ///< The total number of batches.
    BatchType batchType;                                                                   ///< The type of batch (training or testing).
};

#endif // BATCHMANAGER_HPP
