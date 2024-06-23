#ifndef IMAGELOADER_HPP
#define IMAGELOADER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include "ImageContainer.hpp"

class ImageLoader
{
public:
    ImageLoader();
    void loadImagesFromDirectory(const std::string &datasetPath, ImageContainer &container);

private:
    std::vector<std::string> getImagesInDirectory(const std::string &directoryPath);
    void loadImage(const std::string &imagePath, const std::string &category, const std::string &label, ImageContainer &container, int totalImages, int &processedImages);
};

#endif // IMAGELOADER_HPP
