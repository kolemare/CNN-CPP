#ifndef IMAGECONTAINER_HPP
#define IMAGECONTAINER_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "Common.hpp"

/**
 * @brief A class to manage and organize images and their associated labels.
 *
 * The ImageContainer class provides methods to store, retrieve, and categorize
 * images and labels for training and testing purposes.
 */
class ImageContainer
{
public:
    /**
     * @brief Add an image to the container with a specific category and label.
     *
     * @param image A shared pointer to the image (cv::Mat) to be added.
     * @param category The category of the image.
     * @param label The label associated with the image.
     */
    void addImage(const std::shared_ptr<cv::Mat> &image,
                  const std::string &category,
                  const std::string &label);

    /**
     * @brief Add a mapping from a label to a mapped label.
     *
     * @param label The original label.
     * @param mappedLabel The mapped label to associate with the original label.
     */
    void addLabelMapping(const std::string &label,
                         const std::string &mappedLabel);

    /**
     * @brief Set the list of unique labels for the dataset.
     *
     * @param uniqueLabels A vector of unique label strings.
     */
    const void setUniqueLabels(std::vector<std::string> uniqueLabels);

    /**
     * @brief Get the list of unique labels.
     *
     * @return A constant reference to a vector of unique label strings.
     */
    const std::vector<std::string> &getUniqueLabels() const;

    /**
     * @brief Get all images stored in the container.
     *
     * @return A constant reference to a vector of shared pointers to images (cv::Mat).
     */
    const std::vector<std::shared_ptr<cv::Mat>> &getImages() const;

    /**
     * @brief Get all labels stored in the container.
     *
     * @return A constant reference to a vector of label strings.
     */
    const std::vector<std::string> &getLabels() const;

    /**
     * @brief Get the label mapping.
     *
     * @return A constant reference to an unordered map of label mappings.
     */
    const std::unordered_map<std::string, std::string> &getLabelMapping() const;

    /**
     * @brief Get the training images.
     *
     * @return A constant reference to a vector of shared pointers to training images (cv::Mat).
     */
    const std::vector<std::shared_ptr<cv::Mat>> &getTrainingImages() const;

    /**
     * @brief Get the test images.
     *
     * @return A constant reference to a vector of shared pointers to test images (cv::Mat).
     */
    const std::vector<std::shared_ptr<cv::Mat>> &getTestImages() const;

    /**
     * @brief Get the training labels.
     *
     * @return A constant reference to a vector of training label strings.
     */
    const std::vector<std::string> &getTrainingLabels() const;

    /**
     * @brief Get the test labels.
     *
     * @return A constant reference to a vector of test label strings.
     */
    const std::vector<std::string> &getTestLabels() const;

    /**
     * @brief Get images used for single prediction.
     *
     * @return A constant reference to a vector of shared pointers to images for single prediction (cv::Mat).
     */
    const std::vector<std::shared_ptr<cv::Mat>> &getSinglePredictionImages() const;

    /**
     * @brief Get training images by category.
     *
     * @param category The category to filter training images by.
     * @return A vector of shared pointers to training images (cv::Mat) of the specified category.
     */
    std::vector<std::shared_ptr<cv::Mat>> getTrainingImagesByCategory(const std::string &category) const;

    /**
     * @brief Get test images by category.
     *
     * @param category The category to filter test images by.
     * @return A vector of shared pointers to test images (cv::Mat) of the specified category.
     */
    std::vector<std::shared_ptr<cv::Mat>> getTestImagesByCategory(const std::string &category) const;

private:
    std::vector<std::string> uniqueLabels;                     ///< Unique labels in the dataset
    std::vector<std::shared_ptr<cv::Mat>> images;              ///< All images
    std::vector<std::string> labels;                           ///< All labels
    std::unordered_map<std::string, std::string> labelMapping; ///< Mapping of labels

    std::vector<std::shared_ptr<cv::Mat>> trainingImages;         ///< Training images
    std::vector<std::shared_ptr<cv::Mat>> testImages;             ///< Test images
    std::vector<std::shared_ptr<cv::Mat>> singlePredictionImages; ///< Images for single prediction
    std::vector<std::string> trainingLabels;                      ///< Training labels
    std::vector<std::string> testLabels;                          ///< Test labels
};

#endif // IMAGECONTAINER_HPP
