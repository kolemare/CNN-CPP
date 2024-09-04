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

#include "ImageAugmentor.hpp"

ImageAugmentor::ImageAugmentor(int targetWidth, int targetHeight)
    : targetWidth(targetWidth), targetHeight(targetHeight), distribution(0.0f, 1.0f)
{
    // Set default values
    zoomFactor = 1.2f;
    horizontalFlipFlag = true;
    verticalFlipFlag = true;
    shearRange = 0.2f;
    gaussianNoiseStdDev = 10.0f;
    gaussianBlurKernelSize = 5;
    normalizationScale = 1.0f;
    zoomChance = 0.0f;
    horizontalFlipChance = 0.0f;
    verticalFlipChance = 0.0f;
    gaussianNoiseChance = 0.0f;
    gaussianBlurChance = 0.0f;
    shearChance = 0.0f;
}

void ImageAugmentor::augmentImages(ImageContainer &container,
                                   const AugmentTarget &augmentTarget)
{
    auto &trainingImages = container.getTrainingImages();
    container.setNormalizationScale(normalizationScale);

#ifdef AUGMENT_PROGRESS
    int trainingImagesCount = trainingImages.size();
    int processedTrainingImages = 0;
#endif

    for (auto &image : trainingImages)
    {
        if (markProcessed(*image, processedImages))
        {
            *image = rescale(*image);
            normalizeImage(*image);
        }

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
    }

    auto &testImages = container.getTestImages();

#ifdef AUGMENT_PROGRESS
    int testImagesCount = testImages.size();
    int processedTestImages = 0;
#endif

    for (auto &image : testImages)
    {
        if (markProcessed(*image, processedImages))
        {
            *image = rescale(*image);
            normalizeImage(*image);
        }

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
    }

    auto &singlePredictionImages = container.getSinglePredictionImages();

#ifdef AUGMENT_PROGRESS
    int singlePredictionImagesCount = singlePredictionImages.size();
    int processedSinglePredictionImages = 0;
#endif

    for (auto &[imageName, image] : singlePredictionImages)
    {
        if (markProcessed(*image, processedImages))
        {
            *image = rescale(*image);
            normalizeImage(*image);
        }

        if (AugmentTarget::WHOLE_DATASET == augmentTarget || AugmentTarget::SINGLE_PREDICTION == augmentTarget)
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
            processedSinglePredictionImages++;
            int progress = (processedSinglePredictionImages * 100) / singlePredictionImagesCount;
            std::cout << "\rAugmenting single prediction images... " << progress << "%" << std::flush;
#endif
        }
    }
}

void ImageAugmentor::setZoomFactor(float factor)
{
    zoomFactor = factor;
}

void ImageAugmentor::setHorizontalFlip(bool enable)
{
    horizontalFlipFlag = enable;
}

void ImageAugmentor::setVerticalFlip(bool enable)
{
    verticalFlipFlag = enable;
}

void ImageAugmentor::setShearRange(float range)
{
    shearRange = range;
}

void ImageAugmentor::setGaussianNoiseStdDev(float stddev)
{
    gaussianNoiseStdDev = stddev;
}

void ImageAugmentor::setGaussianBlurKernelSize(int size)
{
    gaussianBlurKernelSize = size;
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

void ImageAugmentor::setNormalizationScale(float scale)
{
    normalizationScale = scale;
}

float ImageAugmentor::getNormalizationScale() const
{
    return normalizationScale;
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
    // Normalize image to range [0, normalizationScale]
    image.convertTo(image, CV_32F, normalizationScale / 255.0);
}

bool ImageAugmentor::markProcessed(cv::Mat &image, std::unordered_set<cv::Mat *> &processedImages)
{
    if (processedImages.find(&image) != processedImages.end())
    {
        return false;
    }

    processedImages.insert(&image);
    return true;
}
