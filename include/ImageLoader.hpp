#ifndef IMAGELOADER_HPP
#define IMAGELOADER_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

class ImageLoader
{
public:
    ImageLoader(int width, int height);
    void loadImagesFromDirectory(const std::string &datasetPath);
    const std::vector<std::shared_ptr<cv::Mat>> &getImages() const;
    const std::vector<std::string> &getLabels() const;
    const std::unordered_map<std::string, std::string> &getLabelMapping() const;

    const std::vector<std::shared_ptr<cv::Mat>> &getTrainingImages() const;
    const std::vector<std::shared_ptr<cv::Mat>> &getTestImages() const;
    const std::vector<std::string> &getTrainingLabels() const;
    const std::vector<std::string> &getTestLabels() const;
    const std::vector<std::shared_ptr<cv::Mat>> &getSinglePredictionImages() const;

    std::vector<std::shared_ptr<cv::Mat>> getTrainingImagesByCategory(const std::string &category) const;

private:
    int targetWidth;
    int targetHeight;
    std::vector<std::shared_ptr<cv::Mat>> images;
    std::vector<std::string> labels;
    std::unordered_map<std::string, std::string> labelMapping;

    std::vector<std::shared_ptr<cv::Mat>> trainingImages;
    std::vector<std::shared_ptr<cv::Mat>> testImages;
    std::vector<std::string> trainingLabels;
    std::vector<std::string> testLabels;
    std::vector<std::shared_ptr<cv::Mat>> singlePredictionImages;

    void loadImage(const std::string &imagePath, const std::string &label, int totalImages, int &processedImages);
    std::vector<std::string> getImagesInDirectory(const std::string &directoryPath);
};

#endif // IMAGELOADER_HPP
