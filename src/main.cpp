#include <gtest/gtest.h>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <thread>
#include "ImageLoader.hpp"
#include "ImageAugmentor.hpp"
#include "convolution_layer.hpp"

namespace fs = std::filesystem;

void printLabelCombinations(const std::vector<std::shared_ptr<cv::Mat>> &images,
                            const std::vector<std::string> &labels,
                            const std::unordered_map<std::string, std::string> &labelMapping)
{
    std::map<std::string, int> labelCounts;
    for (const std::string &label : labels)
    {
        labelCounts[label]++;
    }

    for (const auto &pair : labelCounts)
    {
        std::cout << "Label " << pair.first << " (" << labelMapping.at(pair.first) << "): "
                  << pair.second << " images" << std::endl;
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    if (argc > 1 && std::string(argv[1]) == "--tests")
    {
        return RUN_ALL_TESTS();
    }
    else
    {
        std::string datasetPath = "datasets/catsdogs";
        if (!fs::exists(datasetPath))
        {
            throw std::runtime_error("Dataset path does not exist: " + datasetPath);
        }

        // Step 1: Load the images
        ImageLoader loader(150, 150);
        loader.loadImagesFromDirectory(datasetPath);

        std::vector<std::shared_ptr<cv::Mat>> images = loader.getImages();
        std::vector<std::string> labels = loader.getLabels();
        std::unordered_map<std::string, std::string> labelMapping = loader.getLabelMapping();

        std::cout << "\nProcessed " << images.size() << " images successfully!" << std::endl;
        printLabelCombinations(images, labels, labelMapping);

        std::vector<std::shared_ptr<cv::Mat>> trainingImages = loader.getTrainingImages();
        std::vector<std::shared_ptr<cv::Mat>> testImages = loader.getTestImages();
        std::vector<std::shared_ptr<cv::Mat>> singlePredictionImages = loader.getSinglePredictionImages();
        std::vector<std::string> trainingLabels = loader.getTrainingLabels();
        std::vector<std::string> testLabels = loader.getTestLabels();

        std::cout << "Training set size: " << trainingImages.size() << std::endl;
        std::cout << "Test set size: " << testImages.size() << std::endl;
        std::cout << "Single prediction set size: " << singlePredictionImages.size() << std::endl;

        // Example usage of Eigen
        Eigen::MatrixXd mat(2, 2);
        mat(0, 0) = 3;
        mat(1, 0) = 2.5;
        mat(0, 1) = -1;
        mat(1, 1) = mat(1, 0) + mat(0, 1);
        std::cout << "Eigen Matrix:\n"
                  << mat << std::endl;

        // Get training images for the "cats" category
        std::string category = "cats";
        std::vector<std::shared_ptr<cv::Mat>> catImages = loader.getTrainingImagesByCategory(category);
        std::cout << "Number of training images for " << category << ": " << catImages.size() << std::endl;

        std::this_thread::sleep_for(std::chrono::seconds(10));

        // Step 2: Augment the images
        float rescaleFactor = 0.5f;
        float shearAngle = 0.2f; // in radians, e.g., 0.2 rad
        float zoomFactor = 1.2f;
        bool horizontalFlipFlag = true;

        ImageAugmentor augmentor(rescaleFactor, shearAngle, zoomFactor, horizontalFlipFlag);

        for (auto &image : trainingImages)
        {
            image = augmentor.rescale(image);
            image = augmentor.shear(image);
            image = augmentor.zoom(image);
            image = augmentor.horizontalFlip(image);
        }

        for (auto &image : testImages)
        {
            image = augmentor.rescale(image);
            image = augmentor.shear(image);
            image = augmentor.zoom(image);
            image = augmentor.horizontalFlip(image);
        }

        std::this_thread::sleep_for(std::chrono::seconds(10));

        return 0;
    }
}
