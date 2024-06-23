#include "ImageLoader.hpp"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

ImageLoader::ImageLoader(int width, int height)
    : targetWidth(width), targetHeight(height) {}

void ImageLoader::loadImagesFromDirectory(const std::string &datasetPath)
{
    int totalImages = 0;
    for (const auto &category : fs::directory_iterator(datasetPath))
    {
        if (category.is_directory())
        {
            for (const auto &subCategory : fs::directory_iterator(category.path()))
            {
                if (subCategory.is_directory())
                {
                    std::string subCategoryName = subCategory.path().filename().string();
                    labelMapping[subCategoryName] = subCategoryName;
                    totalImages += std::distance(fs::recursive_directory_iterator(subCategory.path()), fs::recursive_directory_iterator());
                }
            }
        }
    }

    int processedImages = 0;
    for (const auto &category : fs::directory_iterator(datasetPath))
    {
        if (category.is_directory())
        {
            for (const auto &subCategory : fs::directory_iterator(category.path()))
            {
                if (subCategory.is_directory())
                {
                    std::string subCategoryName = subCategory.path().filename().string();
                    std::vector<std::string> imagePaths = getImagesInDirectory(subCategory.path().string());

                    for (const auto &path : imagePaths)
                    {
                        try
                        {
                            loadImage(path, subCategoryName, totalImages, processedImages);
                        }
                        catch (const std::exception &e)
                        {
                            std::cerr << "Error processing image " << path << ": " << e.what() << std::endl;
                        }
                    }
                }
            }
        }
    }
}

std::vector<std::string> ImageLoader::getImagesInDirectory(const std::string &directoryPath)
{
    std::vector<std::string> images;
    for (const auto &entry : fs::recursive_directory_iterator(directoryPath))
    {
        if (entry.is_regular_file() && (entry.path().extension() == ".jpg" || entry.path().extension() == ".png"))
        {
            images.push_back(entry.path().string());
        }
    }
    return images;
}

void ImageLoader::loadImage(const std::string &imagePath, const std::string &label, int totalImages, int &processedImages)
{
    try
    {
        // Load the image with reduced size to detect issues early
        cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR | cv::IMREAD_REDUCED_COLOR_2);
        if (image.empty())
        {
            std::cerr << "Could not read the image: " << imagePath << std::endl;
            return;
        }

        // Validate the image dimensions
        if (image.cols <= 0 || image.rows <= 0 || image.channels() != 3)
        {
            std::cerr << "Invalid image dimensions or channels: " << imagePath << std::endl;
            return;
        }

        // Validate the target dimensions for resizing
        if (targetWidth <= 0 || targetHeight <= 0)
        {
            std::cerr << "Invalid target dimensions for resizing: " << targetWidth << "x" << targetHeight << std::endl;
            return;
        }

        // Resize the image
        try
        {
            cv::resize(image, image, cv::Size(targetWidth, targetHeight));
        }
        catch (const cv::Exception &e)
        {
            std::cerr << "Error resizing image " << imagePath << ": " << e.what() << std::endl;
            return;
        }

        // Add the image to the appropriate collections
        auto sharedImage = std::make_shared<cv::Mat>(image);
        images.push_back(sharedImage);
        labels.push_back(label);

        if (imagePath.find("training_set") != std::string::npos)
        {
            trainingImages.push_back(sharedImage);
            trainingLabels.push_back(label);
        }
        else if (imagePath.find("test_set") != std::string::npos)
        {
            testImages.push_back(sharedImage);
            testLabels.push_back(label);
        }
        else if (imagePath.find("single_prediction") != std::string::npos)
        {
            singlePredictionImages.push_back(sharedImage);
        }

        // Update and display progress
        processedImages++;
        int progress = (processedImages * 100) / totalImages;
        std::cout << "\rImage processing... " << progress << "%";
        std::cout.flush();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error processing image " << imagePath << ": " << e.what() << std::endl;
    }
}

const std::vector<std::shared_ptr<cv::Mat>> &ImageLoader::getImages() const
{
    return images;
}

const std::vector<std::string> &ImageLoader::getLabels() const
{
    return labels;
}

const std::unordered_map<std::string, std::string> &ImageLoader::getLabelMapping() const
{
    return labelMapping;
}

const std::vector<std::shared_ptr<cv::Mat>> &ImageLoader::getTrainingImages() const
{
    return trainingImages;
}

const std::vector<std::shared_ptr<cv::Mat>> &ImageLoader::getTestImages() const
{
    return testImages;
}

const std::vector<std::string> &ImageLoader::getTrainingLabels() const
{
    return trainingLabels;
}

const std::vector<std::string> &ImageLoader::getTestLabels() const
{
    return testLabels;
}

const std::vector<std::shared_ptr<cv::Mat>> &ImageLoader::getSinglePredictionImages() const
{
    return singlePredictionImages;
}

std::vector<std::shared_ptr<cv::Mat>> ImageLoader::getTrainingImagesByCategory(const std::string &category) const
{
    std::vector<std::shared_ptr<cv::Mat>> categoryImages;
    for (size_t i = 0; i < trainingLabels.size(); ++i)
    {
        if (trainingLabels[i] == category)
        {
            categoryImages.push_back(trainingImages[i]);
        }
    }
    return categoryImages;
}
