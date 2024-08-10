#include "ImageAugmentor.hpp"
#include <iostream>

ImageAugmentor::ImageAugmentor(float zoomFactor,
                               bool horizontalFlipFlag,
                               bool verticalFlipFlag,
                               float shearRange,
                               float gaussianNoiseStdDev,
                               int gaussianBlurKernelSize,
                               int targetWidth,
                               int targetHeight)
{
    this->zoomFactor = zoomFactor;
    this->horizontalFlipFlag = horizontalFlipFlag;
    this->verticalFlipFlag = verticalFlipFlag;
    this->shearRange = shearRange;
    this->gaussianNoiseStdDev = gaussianNoiseStdDev;
    this->gaussianBlurKernelSize = gaussianBlurKernelSize;
    this->targetWidth = targetWidth;
    this->targetHeight = targetHeight;
    this->zoomChance = 0.3f;
    this->horizontalFlipChance = 0.3f;
    this->verticalFlipChance = 0.3f;
    this->gaussianNoiseChance = 0.3f;
    this->gaussianBlurChance = 0.3f;
    this->shearChance = 0.3f;
    this->distribution = std::uniform_real_distribution<float>(0.0f, 1.0f);
}

void ImageAugmentor::augmentImages(ImageContainer &container,
                                   const AugmentTarget &augmentTarget)
{

    auto &trainingImages = container.getTrainingImages();

#ifdef AUGMENT_PROGRESS
    int trainingImagesCount = trainingImages.size();
    int processedTrainingImages = 0;
#endif

    for (auto &image : trainingImages)
    {
        *image = rescale(*image);
        if (AugmentTarget::WHOLE_DATASET == augmentTarget || AugmentTarget::TRAIN_DATASET == augmentTarget)
        {
            if (distribution(generator) < zoomChance)
            {
                *image = zoom(*image);
            }
            if (distribution(generator) < horizontalFlipChance)
            {
                *image = horizontalFlip(*image);
            }
            if (distribution(generator) < verticalFlipChance)
            {
                *image = verticalFlip(*image);
            }
            if (distribution(generator) < gaussianNoiseChance)
            {
                *image = addGaussianNoise(*image);
            }
            if (distribution(generator) < gaussianBlurChance)
            {
                *image = applyGaussianBlur(*image);
            }
            if (distribution(generator) < shearChance)
            {
                *image = shear(*image);
            }

#ifdef AUGMENT_PROGRESS
            processedTrainingImages++;
            int progress = (processedTrainingImages * 100) / trainingImagesCount;
            std::cout << "\rAugmenting training images... " << progress << "%" << std::flush;
#endif
        }
        normalizeImage(*image);
    }

    std::cout << std::endl;

    auto &testImages = container.getTestImages();

#ifdef AUGMENT_PROGRESS
    int testImagesCount = testImages.size();
    int processedTestImages = 0;
#endif

    for (auto &image : testImages)
    {
        *image = rescale(*image);
        if (AugmentTarget::WHOLE_DATASET == augmentTarget || AugmentTarget::TEST_DATASET == augmentTarget)
        {
            if (distribution(generator) < zoomChance)
            {
                *image = zoom(*image);
            }
            if (distribution(generator) < horizontalFlipChance)
            {
                *image = horizontalFlip(*image);
            }
            if (distribution(generator) < verticalFlipChance)
            {
                *image = verticalFlip(*image);
            }
            if (distribution(generator) < gaussianNoiseChance)
            {
                *image = addGaussianNoise(*image);
            }
            if (distribution(generator) < gaussianBlurChance)
            {
                *image = applyGaussianBlur(*image);
            }
            if (distribution(generator) < shearChance)
            {
                *image = shear(*image);
            }

#ifdef AUGMENT_PROGRESS
            processedTestImages++;
            int progress = (processedTestImages * 100) / testImagesCount;
            std::cout << "\rAugmenting test images... " << progress << "%" << std::flush;
#endif
        }
        normalizeImage(*image);
    }

    std::cout << std::endl;
}

void ImageAugmentor::setZoomChance(float chance)
{
    zoomChance = chance;
}

void ImageAugmentor::setHorizontalFlipChance(float chance)
{
    horizontalFlipChance = chance;
}

void ImageAugmentor::setVerticalFlipChance(float chance)
{
    verticalFlipChance = chance;
}

void ImageAugmentor::setGaussianNoiseChance(float chance)
{
    gaussianNoiseChance = chance;
}

void ImageAugmentor::setGaussianBlurChance(float chance)
{
    gaussianBlurChance = chance;
}

void ImageAugmentor::setShearChance(float chance)
{
    shearChance = chance;
}

cv::Mat ImageAugmentor::rescale(const cv::Mat &image)
{
    if (image.empty())
    {
        std::cerr << "Error: Empty image provided to rescale." << std::endl;
        return image;
    }

    cv::Mat rescaledImage;
    cv::resize(image, rescaledImage, cv::Size(targetWidth, targetHeight));
    return rescaledImage;
}

cv::Mat ImageAugmentor::zoom(const cv::Mat &image)
{
    if (image.empty())
    {
        std::cerr << "Error: Empty image provided to zoom." << std::endl;
        return image;
    }

    if (zoomFactor <= 0)
    {
        std::cerr << "Error: Invalid zoom factor. It must be greater than 0." << std::endl;
        return image;
    }

    float randomZoomFactor = 1 + distribution(generator) * (zoomFactor - 1);
    cv::Mat zoomedImage;
    cv::resize(image, zoomedImage, cv::Size(), randomZoomFactor, randomZoomFactor);
    cv::Rect roi((zoomedImage.cols - targetWidth) / 2, (zoomedImage.rows - targetHeight) / 2, targetWidth, targetHeight);
    return zoomedImage(roi).clone();
}

cv::Mat ImageAugmentor::horizontalFlip(const cv::Mat &image)
{
    if (!horizontalFlipFlag)
        return image;
    if (image.empty())
    {
        std::cerr << "Error: Empty image provided to horizontal flip." << std::endl;
        return image;
    }
    cv::Mat flippedImage;
    cv::flip(image, flippedImage, 1);
    return flippedImage;
}

cv::Mat ImageAugmentor::verticalFlip(const cv::Mat &image)
{
    if (!verticalFlipFlag)
        return image;
    if (image.empty())
    {
        std::cerr << "Error: Empty image provided to vertical flip." << std::endl;
        return image;
    }
    cv::Mat flippedImage;
    cv::flip(image, flippedImage, 0);
    return flippedImage;
}

cv::Mat ImageAugmentor::addGaussianNoise(const cv::Mat &image)
{
    if (image.empty())
    {
        std::cerr << "Error: Empty image provided to add Gaussian noise." << std::endl;
        return image;
    }

    cv::Mat noisyImage = image.clone();
    cv::Mat noise(image.size(), image.type());
    cv::randn(noise, 0, gaussianNoiseStdDev);
    noisyImage += noise;
    return noisyImage;
}

cv::Mat ImageAugmentor::applyGaussianBlur(const cv::Mat &image)
{
    if (image.empty())
    {
        std::cerr << "Error: Empty image provided to apply Gaussian blur." << std::endl;
        return image;
    }

    cv::Mat blurredImage;
    cv::GaussianBlur(image, blurredImage, cv::Size(gaussianBlurKernelSize, gaussianBlurKernelSize), 0);
    return blurredImage;
}

cv::Mat ImageAugmentor::shear(const cv::Mat &image)
{
    if (image.empty())
    {
        std::cerr << "Error: Empty image provided to shear." << std::endl;
        return image;
    }

    float shearFactor = distribution(generator) * shearRange * 2 - shearRange;
    cv::Mat shearedImage;
    cv::Mat transform = (cv::Mat_<double>(2, 3) << 1, shearFactor, 0, 0, 1, 0);

    cv::warpAffine(image, shearedImage, transform, image.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
    return shearedImage;
}

void ImageAugmentor::normalizeImage(cv::Mat &image)
{
    // Normalize image to range [0, 1]
    image.convertTo(image, CV_32F, 1.0 / 255.0);
}
