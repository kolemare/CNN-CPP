#include "ImageAugmentor.hpp"
#include <iostream>

ImageAugmentor::ImageAugmentor(float rescaleFactor,
                               float zoomFactor,
                               bool horizontalFlipFlag,
                               bool verticalFlipFlag,
                               float gaussianNoiseStdDev,
                               int gaussianBlurKernelSize,
                               int targetWidth,
                               int targetHeight)
    : rescaleFactor(rescaleFactor),
      zoomFactor(zoomFactor),
      horizontalFlipFlag(horizontalFlipFlag),
      verticalFlipFlag(verticalFlipFlag),
      gaussianNoiseStdDev(gaussianNoiseStdDev),
      gaussianBlurKernelSize(gaussianBlurKernelSize),
      targetWidth(targetWidth),
      targetHeight(targetHeight),
      zoomChance(0.3f),
      horizontalFlipChance(0.3f),
      verticalFlipChance(0.3f),
      gaussianNoiseChance(0.3f),
      gaussianBlurChance(0.3f),
      distribution(0.0f, 1.0f) {}

void ImageAugmentor::augmentImages(ImageContainer &container)
{
    auto &trainingImages = container.getTrainingImages();
    auto &testImages = container.getTestImages();

    std::cout << "Augmenting images..." << std::endl;
    int trainingImagesCount = trainingImages.size();
    int testImagesCount = testImages.size();
    int processedTrainingImages = 0;
    int processedTestImages = 0;

    for (auto &image : trainingImages)
    {
        *image = rescale(*image);
        if (distribution(generator) < zoomChance)
            *image = zoom(*image);
        if (distribution(generator) < horizontalFlipChance)
            *image = horizontalFlip(*image);
        if (distribution(generator) < verticalFlipChance)
            *image = verticalFlip(*image);
        if (distribution(generator) < gaussianNoiseChance)
            *image = addGaussianNoise(*image);
        if (distribution(generator) < gaussianBlurChance)
            *image = applyGaussianBlur(*image);

        normalizeImage(*image);

        processedTrainingImages++;
        int progress = (processedTrainingImages * 100) / trainingImagesCount;
        std::cout << "\rAugmenting training images... " << progress << "%" << std::flush;
    }

    std::cout << std::endl
              << "Augmentation complete for train_set!" << std::endl;

    for (auto &image : testImages)
    {
        *image = rescale(*image);
        if (distribution(generator) < zoomChance)
            *image = zoom(*image);
        if (distribution(generator) < horizontalFlipChance)
            *image = horizontalFlip(*image);
        if (distribution(generator) < verticalFlipChance)
            *image = verticalFlip(*image);
        if (distribution(generator) < gaussianNoiseChance)
            *image = addGaussianNoise(*image);
        if (distribution(generator) < gaussianBlurChance)
            *image = applyGaussianBlur(*image);

        normalizeImage(*image);

        processedTestImages++;
        int progress = (processedTestImages * 100) / testImagesCount;
        std::cout << "\rAugmenting test images... " << progress << "%" << std::flush;
    }

    std::cout << std::endl
              << "Augmentation complete for test_set!" << std::endl;
}

void ImageAugmentor::setZoomChance(float chance) { zoomChance = chance; }
void ImageAugmentor::setHorizontalFlipChance(float chance) { horizontalFlipChance = chance; }
void ImageAugmentor::setVerticalFlipChance(float chance) { verticalFlipChance = chance; }
void ImageAugmentor::setGaussianNoiseChance(float chance) { gaussianNoiseChance = chance; }
void ImageAugmentor::setGaussianBlurChance(float chance) { gaussianBlurChance = chance; }

cv::Mat ImageAugmentor::rescale(const cv::Mat &image)
{
    if (image.empty())
    {
        std::cerr << "Error: Empty image provided to rescale." << std::endl;
        return image;
    }

    if (rescaleFactor <= 0)
    {
        std::cerr << "Error: Invalid rescale factor. It must be greater than 0." << std::endl;
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

    cv::Mat zoomedImage;
    cv::resize(image, zoomedImage, cv::Size(), zoomFactor, zoomFactor);
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

void ImageAugmentor::normalizeImage(cv::Mat &image)
{
    // Normalize image to range [0, 1]
    image.convertTo(image, CV_32F, 1.0 / 255.0);
}
