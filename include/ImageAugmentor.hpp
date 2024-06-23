#ifndef IMAGEAUGMENTOR_HPP
#define IMAGEAUGMENTOR_HPP

#include <opencv2/opencv.hpp>
#include <memory>

class ImageAugmentor
{
public:
    ImageAugmentor(float rescaleFactor, float shearAngle, float zoomFactor, bool horizontalFlip);
    std::shared_ptr<cv::Mat> rescale(const std::shared_ptr<cv::Mat> &image);
    std::shared_ptr<cv::Mat> shear(const std::shared_ptr<cv::Mat> &image);
    std::shared_ptr<cv::Mat> zoom(const std::shared_ptr<cv::Mat> &image);
    std::shared_ptr<cv::Mat> horizontalFlip(const std::shared_ptr<cv::Mat> &image);

private:
    float rescaleFactor;
    float shearAngle;
    float zoomFactor;
    bool horizontalFlipFlag;
};

#endif // IMAGEAUGMENTOR_HPP
