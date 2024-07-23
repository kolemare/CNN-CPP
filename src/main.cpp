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
    float shearRange = 0.2f;
    float gaussianNoiseStdDev = 10.0f;
    int gaussianBlurKernelSize = 5;
    int targetWidth = 32;
    int targetHeight = 32;

    ImageAugmentor augmentor(rescaleFactor, zoomFactor, horizontalFlipFlag, verticalFlipFlag, shearRange, gaussianNoiseStdDev, gaussianBlurKernelSize, targetWidth, targetHeight);

    // Set augmentation chances
    augmentor.setZoomChance(1.0f);
    augmentor.setHorizontalFlipChance(0.5f);
    augmentor.setVerticalFlipChance(0.0f);
    augmentor.setGaussianNoiseChance(0.0f);
    augmentor.setGaussianBlurChance(0.0f);
    augmentor.setShearChance(1.0f);

    augmentor.augmentImages(container);

    // Step 3: Create the neural network
    NeuralNetwork cnn;
    cnn.setImageSize(targetWidth, targetHeight);

    // Set log level and progress level
    cnn.setLogLevel(LogLevel::None);
    cnn.setProgressLevel(ProgressLevel::ProgressTime);

    // Step 4: Add layers to the neural network

    // Convolution Layer 1
    int filters1 = 32;
    int kernel_size1 = 3;
    int stride1 = 1;
    int padding1 = 1;
    cnn.addConvolutionLayer(filters1, kernel_size1, stride1, padding1, ConvKernelInitialization::XAVIER, ConvBiasInitialization::ZERO);

    // Activation Layer 1
    ActivationType activation1 = ActivationType::RELU;
    cnn.addActivationLayer(activation1);

    // Max Pooling Layer 1
    int pool_size1 = 2;
    int stride_pool1 = 2;
    cnn.addMaxPoolingLayer(pool_size1, stride_pool1);

    // Convolution Layer 2
    int filters2 = 32;
    int kernel_size2 = 3;
    int stride2 = 1;
    int padding2 = 1;
    cnn.addConvolutionLayer(filters2, kernel_size2, stride2, padding2, ConvKernelInitialization::XAVIER, ConvBiasInitialization::ZERO);

    // Activation Layer 2
    ActivationType activation2 = ActivationType::RELU;
    cnn.addActivationLayer(activation2);

    // Max Pooling Layer 2
    int pool_size2 = 2;
    int stride_pool2 = 2;
    cnn.addMaxPoolingLayer(pool_size2, stride_pool2);

    // Flatten Layer
    cnn.addFlattenLayer();

    // Fully Connected Layer 1
    int fc_output_size1 = 128;
    DenseWeightInitialization fc_weight_init1 = DenseWeightInitialization::XAVIER;
    DenseBiasInitialization fc_bias_init1 = DenseBiasInitialization::ZERO;
    cnn.addFullyConnectedLayer(fc_output_size1, fc_weight_init1, fc_bias_init1);

    // Activation Layer 3
    ActivationType activation3 = ActivationType::RELU;
    cnn.addActivationLayer(activation3);

    // Fully Connected Layer 2
    int fc_output_size2 = 1;
    DenseWeightInitialization fc_weight_init2 = DenseWeightInitialization::XAVIER;
    DenseBiasInitialization fc_bias_init2 = DenseBiasInitialization::ZERO;
    cnn.addFullyConnectedLayer(fc_output_size2, fc_weight_init2, fc_bias_init2);

    // Activation Layer 4
    ActivationType activation4 = ActivationType::SIGMOID;
    cnn.addActivationLayer(activation4);

    // Setting loss function
    LossType loss_type = LossType::BINARY_CROSS_ENTROPY;
    cnn.setLossFunction(loss_type);

    // Adam parameters
    std::unordered_map<std::string, double> adam_params = {
        {"beta1", 0.9},
        {"beta2", 0.999},
        {"epsilon", 1e-7}};

    // Compile the network with an optimizer
    cnn.compile(Optimizer::Type::Adam, adam_params);

    // Categories for training and evaluation
    std::vector<std::string> categories = {"dogs", "cats"};

    // Step 5: Train the neural network
    int epochs = 25;
    double learning_rate = 0.001;
    int batch_size = 32;
    cnn.train(container, epochs, learning_rate, batch_size, categories);

    // Step 6: Evaluate the neural network
    cnn.evaluate(container, categories);

    // Step 7: Making a single prediction
    std::string singleImagePath = "datasets/single_prediction/cat_or_dog_1.jpg";
    cv::Mat image = cv::imread(singleImagePath, cv::IMREAD_COLOR);
    if (image.empty())
    {
        throw std::runtime_error("Could not read the image: " + singleImagePath);
    }

    cv::resize(image, image, cv::Size(32, 32));

    // Convert image to Eigen tensor
    Eigen::Tensor<double, 4> singleImageBatch(1, image.channels(), image.rows, image.cols);
    for (int h = 0; h < image.rows; ++h)
    {
        for (int w = 0; w < image.cols; ++w)
        {
            for (int c = 0; c < image.channels(); ++c)
            {
                singleImageBatch(0, c, h, w) = static_cast<double>(image.at<cv::Vec3b>(h, w)[c]);
            }
        }
    }

    std::cout << "Input dimensions for prediction: " << singleImageBatch.dimension(0) << "x" << singleImageBatch.dimension(1) << "x" << singleImageBatch.dimension(2) << "x" << singleImageBatch.dimension(3) << std::endl;

    Eigen::Tensor<double, 4> prediction = cnn.forward(singleImageBatch);
    std::string result = prediction(0, 0, 0, 0) >= 0.5 ? "dog" : "cat";

    std::cout << "Prediction for " << singleImagePath << ": " << result << " (Score: " << prediction(0, 0, 0, 0) << ")\n";
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
