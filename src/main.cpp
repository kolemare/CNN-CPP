#include <gtest/gtest.h>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <thread>
#include "ImageLoader.hpp"
#include "ImageAugmentor.hpp"
#include "BatchManager.hpp"
#include <Eigen/Dense>

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
        ImageLoader loader;
        ImageContainer container;
        loader.loadImagesFromDirectory(datasetPath, container);

        std::cout << "\nProcessed " << container.getImages().size() << " images successfully!" << std::endl;
        printLabelCombinations(container.getImages(), container.getLabels(), container.getLabelMapping());

        std::cout << "Training set size: " << container.getTrainingImages().size() << std::endl;
        std::cout << "Test set size: " << container.getTestImages().size() << std::endl;
        std::cout << "Single prediction set size: " << container.getSinglePredictionImages().size() << std::endl;

        // Example usage of Eigen
        Eigen::MatrixXd mat(2, 2);
        mat(0, 0) = 3;
        mat(1, 0) = 2.5;
        mat(0, 1) = -1;
        mat(1, 1) = mat(1, 0) + mat(0, 1);
        std::cout << "Eigen Matrix:\n"
                  << mat << std::endl;

        // Get training images for the "cats" category
        std::string cats = "cats";
        std::string dogs = "dogs";
        std::vector<std::shared_ptr<cv::Mat>> catImagesTrain = container.getTrainingImagesByCategory(cats);
        std::vector<std::shared_ptr<cv::Mat>> dogImagesTrain = container.getTrainingImagesByCategory(dogs);

        std::vector<std::shared_ptr<cv::Mat>> catImagesTest = container.getTestImagesByCategory(cats);
        std::vector<std::shared_ptr<cv::Mat>> dogImagesTest = container.getTestImagesByCategory(dogs);

        std::cout << "Number of training images for " << cats << ": " << catImagesTrain.size() << std::endl;
        std::cout << "Number of training images for " << dogs << ": " << dogImagesTrain.size() << std::endl;
        std::cout << "Number of test images for " << cats << ": " << catImagesTest.size() << std::endl;
        std::cout << "Number of test images for " << dogs << ": " << dogImagesTest.size() << std::endl;

        std::this_thread::sleep_for(std::chrono::seconds(10));

        // Step 2: Augment the images
        float rescaleFactor = 1.0f / 255.0f;
        float zoomFactor = 1.2f;
        bool horizontalFlipFlag = true;
        bool verticalFlipFlag = true;
        float gaussianNoiseStdDev = 10.0f;
        int gaussianBlurKernelSize = 5;
        int targetWidth = 64;
        int targetHeight = 64;

        ImageAugmentor augmentor(rescaleFactor, zoomFactor, horizontalFlipFlag, verticalFlipFlag, gaussianNoiseStdDev, gaussianBlurKernelSize, targetWidth, targetHeight);

        // Set augmentation chances
        augmentor.setZoomChance(0.3f);
        augmentor.setHorizontalFlipChance(0.3f);
        augmentor.setVerticalFlipChance(0.3f);
        augmentor.setGaussianNoiseChance(0.3f);
        augmentor.setGaussianBlurChance(0.3f);

        augmentor.augmentImages(container);

        std::vector<std::shared_ptr<cv::Mat>> catImagesCheck = container.getTrainingImagesByCategory(cats);
        if (!catImagesCheck.empty())
        {
            int width = catImagesCheck[0]->cols;
            int height = catImagesCheck[0]->rows;
            bool allSameSize = true;

            for (const auto &image : catImagesCheck)
            {
                if (image->cols != width || image->rows != height)
                {
                    allSameSize = false;
                    break;
                }
            }

            if (allSameSize)
            {
                std::cout << "All images are of the same size: " << width << "x" << height << std::endl;
            }
            else
            {
                std::cout << "Not all images are of the same size." << std::endl;
            }
        }

        fs::create_directory("abcd");
        for (size_t i = 0; i < catImagesCheck.size(); ++i)
        {
            std::string filename = "abcd/image_" + std::to_string(i) + ".jpg";
            cv::imwrite(filename, *catImagesCheck[i]);
        }

        std::this_thread::sleep_for(std::chrono::seconds(10));

        return 0;
    }
}
