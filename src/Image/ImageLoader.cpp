#include "ImageLoader.hpp"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

ImageLoader::ImageLoader() {}

void ImageLoader::loadImagesFromDirectory(const std::string &datasetPath, ImageContainer &container)
{
    int totalImages = 0;
    std::vector<std::string> uniqueLabels;
    for (const auto &category : fs::directory_iterator(datasetPath))
    {
        if (category.is_directory())
        {
            for (const auto &subCategory : fs::directory_iterator(category.path()))
            {
                if (subCategory.is_directory())
                {
                    std::string subCategoryName = subCategory.path().filename().string();
                    container.addLabelMapping(subCategoryName, subCategoryName);

                    // Add subCategoryName to uniqueLabels if it's not already present
                    if (std::find(uniqueLabels.begin(), uniqueLabels.end(), subCategoryName) == uniqueLabels.end())
                    {
                        uniqueLabels.push_back(subCategoryName);
                    }

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
                            loadImage(path, subCategoryName, category.path().filename().string(), container, totalImages, processedImages);
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
    container.setUniqueLabels(uniqueLabels);
    std::cout << "\nLoaded " << container.getImages().size() << " images successfully!" << std::endl;
    std::cout << "Training set size: " << container.getTrainingImages().size() << std::endl;
    std::cout << "Test set size: " << container.getTestImages().size() << std::endl;
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

void ImageLoader::loadImage(const std::string &imagePath, const std::string &category, const std::string &label, ImageContainer &container, int totalImages, int &processedImages)
{
    try
    {
        cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (image.empty())
        {
            std::cerr << "Could not read the image: " << imagePath << std::endl;
            return;
        }

        auto sharedImage = std::make_shared<cv::Mat>(image);
        container.addImage(sharedImage, category, label);

#ifdef LOADING_PROGRESS
        processedImages++;
        int progress = (processedImages * 100) / totalImages;
        std::cout << "\rLoading images... " << progress << "%";
        std::cout.flush();
#endif
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error processing image " << imagePath << ": " << e.what() << std::endl;
    }
}
