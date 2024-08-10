#ifndef IMAGE_AUGMENTOR_HPP
#define IMAGE_AUGMENTOR_HPP

#include <opencv2/opencv.hpp>
#include "ImageContainer.hpp"
#include <random>

/**
 * @brief A class to perform image augmentation operations.
 *
 * The ImageAugmentor class provides various image transformation operations
 * such as zoom, flip, noise addition, blur, and shear to augment image datasets.
 * ImageAugmentor is also responsible for normalization of image and resizing.
 */
class ImageAugmentor
{
public:
    /**
     * @brief Constructor to initialize the ImageAugmentor with augmentation parameters.
     *
     * @param zoomFactor Factor by which to zoom the image.
     * @param horizontalFlipFlag Flag to enable or disable horizontal flipping.
     * @param verticalFlipFlag Flag to enable or disable vertical flipping.
     * @param shearRange Range for shearing transformation.
     * @param gaussianNoiseStdDev Standard deviation for Gaussian noise.
     * @param gaussianBlurKernelSize Kernel size for Gaussian blur.
     * @param targetWidth Target width for rescaling images.
     * @param targetHeight Target height for rescaling images.
     */
    ImageAugmentor(float zoomFactor,
                   bool horizontalFlipFlag,
                   bool verticalFlipFlag,
                   float shearRange,
                   float gaussianNoiseStdDev,
                   int gaussianBlurKernelSize,
                   int targetWidth,
                   int targetHeight);

    /**
     * @brief Augment images in the specified image container.
     *
     * @param container The ImageContainer holding images to be augmented.
     * @param augmentTarget Specifies the target for augmentation (e.g., training, testing).
     */
    void augmentImages(ImageContainer &container,
                       const AugmentTarget &augmentTarget);

    /**
     * @brief Set the probability of applying zoom augmentation.
     *
     * @param chance Probability (0.0 to 1.0) of applying zoom.
     */
    void setZoomChance(float chance);

    /**
     * @brief Set the probability of applying horizontal flip augmentation.
     *
     * @param chance Probability (0.0 to 1.0) of applying horizontal flip.
     */
    void setHorizontalFlipChance(float chance);

    /**
     * @brief Set the probability of applying vertical flip augmentation.
     *
     * @param chance Probability (0.0 to 1.0) of applying vertical flip.
     */
    void setVerticalFlipChance(float chance);

    /**
     * @brief Set the probability of applying Gaussian noise augmentation.
     *
     * @param chance Probability (0.0 to 1.0) of adding Gaussian noise.
     */
    void setGaussianNoiseChance(float chance);

    /**
     * @brief Set the probability of applying Gaussian blur augmentation.
     *
     * @param chance Probability (0.0 to 1.0) of applying Gaussian blur.
     */
    void setGaussianBlurChance(float chance);

    /**
     * @brief Set the probability of applying shear augmentation.
     *
     * @param chance Probability (0.0 to 1.0) of applying shear transformation.
     */
    void setShearChance(float chance);

private:
    /**
     * @brief Rescale the image to the target width and height.
     *
     * @param image The input image to rescale.
     * @return cv::Mat The rescaled image.
     */
    cv::Mat rescale(const cv::Mat &image);

    /**
     * @brief Apply zoom to the image.
     *
     * @param image The input image to zoom.
     * @return cv::Mat The zoomed image.
     */
    cv::Mat zoom(const cv::Mat &image);

    /**
     * @brief Apply horizontal flip to the image.
     *
     * @param image The input image to flip.
     * @return cv::Mat The horizontally flipped image.
     */
    cv::Mat horizontalFlip(const cv::Mat &image);

    /**
     * @brief Apply vertical flip to the image.
     *
     * @param image The input image to flip.
     * @return cv::Mat The vertically flipped image.
     */
    cv::Mat verticalFlip(const cv::Mat &image);

    /**
     * @brief Add Gaussian noise to the image.
     *
     * @param image The input image to which noise will be added.
     * @return cv::Mat The image with added Gaussian noise.
     */
    cv::Mat addGaussianNoise(const cv::Mat &image);

    /**
     * @brief Apply Gaussian blur to the image.
     *
     * @param image The input image to blur.
     * @return cv::Mat The blurred image.
     */
    cv::Mat applyGaussianBlur(const cv::Mat &image);

    /**
     * @brief Apply shear transformation to the image.
     *
     * @param image The input image to shear.
     * @return cv::Mat The sheared image.
     */
    cv::Mat shear(const cv::Mat &image);

    /**
     * @brief Normalize the image's pixel values.
     *
     * @param image The image to normalize.
     */
    void normalizeImage(cv::Mat &image);

    float zoomFactor;           ///< Zoom factor for augmentation
    bool horizontalFlipFlag;    ///< Flag for horizontal flip
    bool verticalFlipFlag;      ///< Flag for vertical flip
    float gaussianNoiseStdDev;  ///< Standard deviation for Gaussian noise
    int gaussianBlurKernelSize; ///< Kernel size for Gaussian blur
    int targetWidth;            ///< Target width for rescaling
    int targetHeight;           ///< Target height for rescaling

    float zoomChance;           ///< Probability of zooming
    float horizontalFlipChance; ///< Probability of horizontal flipping
    float verticalFlipChance;   ///< Probability of vertical flipping
    float gaussianNoiseChance;  ///< Probability of adding Gaussian noise
    float gaussianBlurChance;   ///< Probability of applying Gaussian blur
    float shearChance;          ///< Probability of shearing
    float shearRange;           ///< Range for shear transformation

    std::default_random_engine generator;               ///< Random number generator
    std::uniform_real_distribution<float> distribution; ///< Uniform distribution for randomness
};

#endif // IMAGE_AUGMENTOR_HPP
