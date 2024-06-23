#ifndef IMAGECONTAINER_HPP
#define IMAGECONTAINER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

class ImageContainer
{
public:
    void addImage(const std::shared_ptr<cv::Mat> &image, const std::string &category, const std::string &label);
    void addLabelMapping(const std::string &label, const std::string &mappedLabel);

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
    std::vector<std::shared_ptr<cv::Mat>> images;
    std::vector<std::string> labels;
    std::unordered_map<std::string, std::string> labelMapping;

    std::vector<std::shared_ptr<cv::Mat>> trainingImages;
    std::vector<std::shared_ptr<cv::Mat>> testImages;
    std::vector<std::shared_ptr<cv::Mat>> singlePredictionImages;
    std::vector<std::string> trainingLabels;
    std::vector<std::string> testLabels;
};

#endif // IMAGECONTAINER_HPP
