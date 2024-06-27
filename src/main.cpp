#include <gtest/gtest.h>
#include <filesystem>
#include <iostream>
#include "ImageLoader.hpp"
#include "ImageAugmentor.hpp"
#include "BatchManager.hpp"
#include "NeuralNetwork.hpp"
#include <Eigen/Dense>

namespace fs = std::filesystem;

void tensorModel(const std::string &datasetPath)
{
    // Step 1: Load the images
    ImageLoader loader;
    ImageContainer container;
    loader.loadImagesFromDirectory(datasetPath, container);

    std::cout << "\nProcessed " << container.getImages().size() << " images successfully!" << std::endl;
    std::cout << "Training set size: " << container.getTrainingImages().size() << std::endl;
    std::cout << "Test set size: " << container.getTestImages().size() << std::endl;

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

    // Step 3: Create the neural network
    NeuralNetwork cnn;

    // Adding layers to the neural network
    cnn.addConvolutionLayer(32, 3, 3, 1, 1, Eigen::VectorXd::Zero(32));
    cnn.addActivationLayer(RELU);
    cnn.addMaxPoolingLayer(2, 2);

    cnn.addConvolutionLayer(32, 3, 32, 1, 1, Eigen::VectorXd::Zero(32));
    cnn.addActivationLayer(RELU);
    cnn.addMaxPoolingLayer(2, 2);

    cnn.addFlattenLayer();
    cnn.addFullyConnectedLayer(32 * 16 * 16, 128, std::make_unique<SGD>());
    cnn.addActivationLayer(RELU);
    cnn.addFullyConnectedLayer(128, 1, std::make_unique<SGD>());
    cnn.addActivationLayer(SIGMOID);

    // Setting loss function
    cnn.setLossFunction(LossType::BINARY_CROSS_ENTROPY);

    // Categories for training and evaluation
    std::vector<std::string> categories = {"cats", "dogs"};

    // Step 4: Train the neural network
    cnn.train(container, 25, 0.001, 32, categories);

    // Step 5: Evaluate the neural network
    cnn.evaluate(container, categories);

    // Step 6: Making a single prediction
    std::string singleImagePath = "datasets/single_prediction/cat_or_dog_1.jpg";
    cv::Mat image = cv::imread(singleImagePath, cv::IMREAD_COLOR);
    if (image.empty())
    {
        throw std::runtime_error("Could not read the image: " + singleImagePath);
    }

    cv::resize(image, image, cv::Size(64, 64));
    Eigen::MatrixXd singleImageBatch(1, image.rows * image.cols * image.channels());
    cv::Mat flatImage = image.reshape(1, 1);
    Eigen::Map<Eigen::MatrixXd> eigenImage(flatImage.ptr<double>(), 1, image.rows * image.cols * image.channels());
    singleImageBatch.row(0) = eigenImage;

    Eigen::MatrixXd prediction = cnn.forward(singleImageBatch);
    std::string result = prediction(0, 0) >= 0.5 ? "dog" : "cat";

    std::cout << "Prediction for " << singleImagePath << ": " << result << " (Score: " << prediction(0, 0) << ")\n";
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
        try
        {
            tensorModel("datasets/catsdogs");
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    }

    return 0;
}
