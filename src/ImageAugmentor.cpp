#include "ImageAugmentor.hpp"
#include <cmath>
#include <iostream>

ImageAugmentor::ImageAugmentor(float rescaleFactor, float shearAngle, float zoomFactor, bool horizontalFlip)
    : rescaleFactor(rescaleFactor), shearAngle(shearAngle), zoomFactor(zoomFactor), horizontalFlipFlag(horizontalFlip) {}

std::shared_ptr<cv::Mat> ImageAugmentor::rescale(const std::shared_ptr<cv::Mat> &image)
{
    if (image->empty())
    {
        std::cerr << "Error: Empty image provided to rescale." << std::endl;
        return image;
    }

    if (rescaleFactor <= 0)
    {
        std::cerr << "Error: Invalid rescale factor. It must be greater than 0." << std::endl;
        return image;
    }

    std::cout << "Rescaling image of size: " << image->cols << "x" << image->rows << " with factor: " << rescaleFactor << std::endl;
    auto rescaledImage = std::make_shared<cv::Mat>();
    cv::resize(*image, *rescaledImage, cv::Size(image->cols * rescaleFactor, image->rows * rescaleFactor));
    return rescaledImage;
}

std::shared_ptr<cv::Mat> ImageAugmentor::shear(const std::shared_ptr<cv::Mat> &image)
{
    if (image->empty())
    {
        std::cerr << "Error: Empty image provided to shear." << std::endl;
        return image;
    }
    int width = image->cols;
    int height = image->rows;
    std::cout << "Shearing image of size: " << width << "x" << height << " with angle: " << shearAngle << std::endl;
    cv::Mat shearMatrix = (cv::Mat_<double>(2, 3) << 1, shearAngle, 0, 0, 1, 0);
    auto shearedImage = std::make_shared<cv::Mat>();
    cv::warpAffine(*image, *shearedImage, shearMatrix, cv::Size(width + height * shearAngle, height));
    return shearedImage;
}

std::shared_ptr<cv::Mat> ImageAugmentor::zoom(const std::shared_ptr<cv::Mat> &image)
{
    if (image->empty())
    {
        std::cerr << "Error: Empty image provided to zoom." << std::endl;
        return image;
    }

    if (zoomFactor <= 0)
    {
        std::cerr << "Error: Invalid zoom factor. It must be greater than 0." << std::endl;
        return image;
    }

    std::cout << "Zooming image of size: " << image->cols << "x" << image->rows << " with factor: " << zoomFactor << std::endl;
    auto zoomedImage = std::make_shared<cv::Mat>();
    cv::resize(*image, *zoomedImage, cv::Size(image->cols * zoomFactor, image->rows * zoomFactor));
    return zoomedImage;
}

std::shared_ptr<cv::Mat> ImageAugmentor::horizontalFlip(const std::shared_ptr<cv::Mat> &image)
{
    if (!horizontalFlipFlag)
        return image;
    if (image->empty())
    {
        std::cerr << "Error: Empty image provided to horizontal flip." << std::endl;
        return image;
    }
    std::cout << "Flipping image horizontally of size: " << image->cols << "x" << image->rows << std::endl;
    auto flippedImage = std::make_shared<cv::Mat>();
    cv::flip(*image, *flippedImage, 1);
    return flippedImage;
}
