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

#ifndef IMAGELOADER_HPP
#define IMAGELOADER_HPP

#include "Common.hpp"
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

    /**
     * @brief Load single prediction images from a specified directory.
     *
     * This function reads images from the single_prediction directory within `datasetPath`
     * and stores them in the provided `container`.
     *
     * @param datasetPath The path to the directory containing the dataset images.
     * @param container The ImageContainer to store the loaded images.
     */
    void loadSinglePredictionImages(const std::string &datasetPath,
                                    ImageContainer &container);
};

#endif // IMAGELOADER_HPP
