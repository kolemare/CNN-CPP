#ifndef IMAGE_AUGMENTOR_HPP
#define IMAGE_AUGMENTOR_HPP

// #define AUGMENT_PROGRESS

#include <opencv2/opencv.hpp>
#include "ImageContainer.hpp"
#include <random>

class ImageAugmentor
{
public:
    ImageAugmentor(float rescaleFactor, float zoomFactor, bool horizontalFlipFlag, bool verticalFlipFlag, float gaussianNoiseStdDev, int gaussianBlurKernelSize, int targetWidth, int targetHeight);

    void augmentImages(ImageContainer &container);

    void setZoomChance(float chance);
    void setHorizontalFlipChance(float chance);
    void setVerticalFlipChance(float chance);
    void setGaussianNoiseChance(float chance);
    void setGaussianBlurChance(float chance);

private:
    cv::Mat rescale(const cv::Mat &image);
    cv::Mat zoom(const cv::Mat &image);
    cv::Mat horizontalFlip(const cv::Mat &image);
    cv::Mat verticalFlip(const cv::Mat &image);
    cv::Mat addGaussianNoise(const cv::Mat &image);
    cv::Mat applyGaussianBlur(const cv::Mat &image);
    void normalizeImage(cv::Mat &image);

    float rescaleFactor;
    float zoomFactor;
    bool horizontalFlipFlag;
    bool verticalFlipFlag;
    float gaussianNoiseStdDev;
    int gaussianBlurKernelSize;
    int targetWidth;
    int targetHeight;

    float zoomChance;
    float horizontalFlipChance;
    float verticalFlipChance;
    float gaussianNoiseChance;
    float gaussianBlurChance;

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution;
};

#endif // IMAGE_AUGMENTOR_HPP
