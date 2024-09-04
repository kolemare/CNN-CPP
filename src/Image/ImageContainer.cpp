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

#include "ImageContainer.hpp"

void ImageContainer::addImage(const std::shared_ptr<cv::Mat> &image,
                              const std::string &category,
                              const std::string &label)
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
}

void ImageContainer::addSinglePredictionImage(const std::shared_ptr<cv::Mat> &image, const std::string &imageName)
{
    singlePredictionImages[imageName] = image;
}

void ImageContainer::addLabelMapping(const std::string &label,
                                     const std::string &mappedLabel)
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

const std::vector<std::string> &ImageContainer::getTrainingLabels() const
{
    return trainingLabels;
}

const std::vector<std::shared_ptr<cv::Mat>> &ImageContainer::getTestImages() const
{
    return testImages;
}

const std::vector<std::string> &ImageContainer::getTestLabels() const
{
    return testLabels;
}

const std::unordered_map<std::string, std::shared_ptr<cv::Mat>> &ImageContainer::getSinglePredictionImages() const
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

void ImageContainer::setUniqueLabels(const std::vector<std::string> &uniqueLabels)
{
    this->uniqueLabels = uniqueLabels;
}

const std::vector<std::string> &ImageContainer::getUniqueLabels() const
{
    return this->uniqueLabels;
}

void ImageContainer::setNormalizationScale(float scale)
{
    normalizationScale = scale;
}

float ImageContainer::getNormalizationScale() const
{
    return normalizationScale;
}
