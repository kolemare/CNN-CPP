#ifndef IMAGE_AUGMENTOR_HPP
#define IMAGE_AUGMENTOR_HPP

#include <opencv2/opencv.hpp>
#include "ImageContainer.hpp"
#include <random>

class ImageAugmentor
{
public:
    ImageAugmentor(float zoomFactor, bool horizontalFlipFlag, bool verticalFlipFlag, float shearRange, float gaussianNoiseStdDev, int gaussianBlurKernelSize, int targetWidth, int targetHeight);

    void augmentImages(ImageContainer &container, const AugmentTarget &augmentTarget);

    void setZoomChance(float chance);
    void setHorizontalFlipChance(float chance);
    void setVerticalFlipChance(float chance);
    void setGaussianNoiseChance(float chance);
    void setGaussianBlurChance(float chance);
    void setShearChance(float chance);

private:
    cv::Mat rescale(const cv::Mat &image);
    cv::Mat zoom(const cv::Mat &image);
    cv::Mat horizontalFlip(const cv::Mat &image);
    cv::Mat verticalFlip(const cv::Mat &image);
    cv::Mat addGaussianNoise(const cv::Mat &image);
    cv::Mat applyGaussianBlur(const cv::Mat &image);
    cv::Mat shear(const cv::Mat &image);
    void normalizeImage(cv::Mat &image);

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
    float shearChance;
    float shearRange;

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution;
};

#endif // IMAGE_AUGMENTOR_HPP
