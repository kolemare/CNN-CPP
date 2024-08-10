#ifndef IMAGELOADER_HPP
#define IMAGELOADER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include "ImageContainer.hpp"

/**
 * @brief The ImageLoader class is responsible for loading images from a directory
 *        into an ImageContainer.
 */
class ImageLoader
{
public:
    /**
     * @brief Construct a new ImageLoader object.
     */
    ImageLoader();

    /**
     * @brief Load images from a specified directory into an ImageContainer.
     *
     * This function reads images from the directory specified by `datasetPath`
     * and stores them in the provided `container`. Images are categorized and
     * labeled based on directory structure.
     *
     * @param datasetPath The path to the directory containing the dataset images.
     * @param container The ImageContainer to store the loaded images.
     */
    void loadImagesFromDirectory(const std::string &datasetPath,
                                 ImageContainer &container);

private:
    /**
     * @brief Retrieve the list of image file names in a directory.
     *
     * @param directoryPath The path to the directory.
     * @return A vector of image file names present in the directory.
     */
    std::vector<std::string> getImagesInDirectory(const std::string &directoryPath);

    /**
     * @brief Load an individual image from the file system and store it in the ImageContainer.
     *
     * @param imagePath The file path of the image to be loaded.
     * @param category The category of the image, typically derived from directory structure.
     * @param label The label associated with the image.
     * @param container The ImageContainer to store the loaded image.
     * @param totalImages The total number of images expected to be processed.
     * @param processedImages A reference to an integer tracking the number of processed images.
     */
    void loadImage(const std::string &imagePath,
                   const std::string &category,
                   const std::string &label,
                   ImageContainer &container,
                   int totalImages,
                   int &processedImages);
};

#endif // IMAGELOADER_HPP
