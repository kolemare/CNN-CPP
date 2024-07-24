#include "ImageContainer.hpp"

void ImageContainer::addImage(const std::shared_ptr<cv::Mat> &image, const std::string &category, const std::string &label)
{
    images.push_back(image);
    labels.push_back(category);

    if (label == "training_set")
    {
        trainingImages.push_back(image);
        trainingLabels.push_back(category);
    }
    else if (label == "test_set")
    {
        testImages.push_back(image);
        testLabels.push_back(category);
    }
    else if (label == "single_prediction")
    {
        singlePredictionImages.push_back(image);
    }
}

void ImageContainer::addLabelMapping(const std::string &label, const std::string &mappedLabel)
{
    labelMapping[label] = mappedLabel;
}

const std::vector<std::shared_ptr<cv::Mat>> &ImageContainer::getImages() const
{
    return images;
}

const std::vector<std::string> &ImageContainer::getLabels() const
{
    return labels;
}

const std::unordered_map<std::string, std::string> &ImageContainer::getLabelMapping() const
{
    return labelMapping;
}

const std::vector<std::shared_ptr<cv::Mat>> &ImageContainer::getTrainingImages() const
{
    return trainingImages;
}

const std::vector<std::shared_ptr<cv::Mat>> &ImageContainer::getTestImages() const
{
    return testImages;
}

const std::vector<std::string> &ImageContainer::getTrainingLabels() const
{
    return trainingLabels;
}

const std::vector<std::string> &ImageContainer::getTestLabels() const
{
    return testLabels;
}

const std::vector<std::shared_ptr<cv::Mat>> &ImageContainer::getSinglePredictionImages() const
{
    return singlePredictionImages;
}

std::vector<std::shared_ptr<cv::Mat>> ImageContainer::getTrainingImagesByCategory(const std::string &category) const
{
    std::vector<std::shared_ptr<cv::Mat>> categoryImages;
    for (size_t i = 0; i < trainingLabels.size(); ++i)
    {
        if (trainingLabels[i] == category)
        {
            categoryImages.push_back(trainingImages[i]);
        }
    }
    return categoryImages;
}

std::vector<std::shared_ptr<cv::Mat>> ImageContainer::getTestImagesByCategory(const std::string &category) const
{
    std::vector<std::shared_ptr<cv::Mat>> categoryImages;
    for (size_t i = 0; i < testLabels.size(); ++i)
    {
        if (testLabels[i] == category)
        {
            categoryImages.push_back(testImages[i]);
        }
    }
    return categoryImages;
}

const void ImageContainer::setUniqueLabels(std::vector<std::string> uniqueLabels)
{
    this->uniqueLabels = uniqueLabels;
}

const std::vector<std::string> &ImageContainer::getUniqueLabels() const
{
    return this->uniqueLabels;
}
