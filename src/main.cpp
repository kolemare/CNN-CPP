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

    // Step 2: Augment the images
    float zoomFactor = 1.2f;
    bool horizontalFlipFlag = true;
    bool verticalFlipFlag = true;
    float shearRange = 0.2f;
    float gaussianNoiseStdDev = 10.0f;
    int gaussianBlurKernelSize = 5;
    int targetWidth = 32;
    int targetHeight = 32;

    ImageAugmentor augmentor(zoomFactor, horizontalFlipFlag, verticalFlipFlag, shearRange, gaussianNoiseStdDev, gaussianBlurKernelSize, targetWidth, targetHeight);

    // Set augmentation chances
    augmentor.setZoomChance(1.0f);
    augmentor.setHorizontalFlipChance(0.5f);
    augmentor.setVerticalFlipChance(0.0f);
    augmentor.setGaussianNoiseChance(0.0f);
    augmentor.setGaussianBlurChance(0.0f);
    augmentor.setShearChance(1.0f);

    augmentor.augmentImages(container, AugmentTarget::TRAIN_DATASET);

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

    // Batch Normalization Layer 1
    cnn.addBatchNormalizationLayer();

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

    // Batch Normalization Layer 2
    cnn.addBatchNormalizationLayer();

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

    // Batch Normalization Layer 3
    cnn.addBatchNormalizationLayer();

    // Activation Layer 3
    ActivationType activation3 = ActivationType::RELU;
    cnn.addActivationLayer(activation3);

    // Fully Connected Layer 2
    int fc_output_size2 = 64;
    DenseWeightInitialization fc_weight_init2 = DenseWeightInitialization::XAVIER;
    DenseBiasInitialization fc_bias_init2 = DenseBiasInitialization::ZERO;
    cnn.addFullyConnectedLayer(fc_output_size2, fc_weight_init2, fc_bias_init2);

    // Batch Normalization Layer 4
    cnn.addBatchNormalizationLayer();

    // Activation Layer 4
    ActivationType activation4 = ActivationType::RELU;
    cnn.addActivationLayer(activation4);

    // Fully Connected Layer 3
    int fc_output_size3 = 32;
    DenseWeightInitialization fc_weight_init3 = DenseWeightInitialization::XAVIER;
    DenseBiasInitialization fc_bias_init3 = DenseBiasInitialization::ZERO;
    cnn.addFullyConnectedLayer(fc_output_size3, fc_weight_init3, fc_bias_init3);

    // Batch Normalization Layer 5
    cnn.addBatchNormalizationLayer();

    // Activation Layer 5
    ActivationType activation5 = ActivationType::RELU;
    cnn.addActivationLayer(activation5);

    // Fully Connected Layer 4
    int fc_output_size4 = 1;
    DenseWeightInitialization fc_weight_init4 = DenseWeightInitialization::XAVIER;
    DenseBiasInitialization fc_bias_init4 = DenseBiasInitialization::ZERO;
    cnn.addFullyConnectedLayer(fc_output_size4, fc_weight_init4, fc_bias_init4);

    // Activation Layer 6
    ActivationType activation6 = ActivationType::SIGMOID;
    cnn.addActivationLayer(activation6);

    // Setting loss function
    LossType loss_type = LossType::BINARY_CROSS_ENTROPY;
    cnn.setLossFunction(loss_type);

    // Enable Default Gradient Clipping
    cnn.enableGradientClipping();

    // Enable ELRALES
    // double learning_rate_coef = 0.5; // 0.5 Learning Coefficient Multiplier
    // int maxSuccessiveFailures = 3;   // 3 Sucessive Epoch Failures
    // int maxFails = 20;               // 20 Total Epoch Failures
    // double tolerance = 0.0;          // 0% Tolerance
    // cnn.enableELRALES(learning_rate_coef, maxSuccessiveFailures, maxFails, tolerance);

    std::unordered_map<std::string, double> polynomial_params = {
        {"end_learning_rate", 0.00001},
        {"decay_steps", 25},
        {"power", 2.0}};

    cnn.enableLearningDecay(LearningDecayType::POLYNOMIAL, polynomial_params);

    // Compile the network with an optimizer
    cnn.compile(OptimizerType::Adam);

    // Step 5: Train the neural network
    int epochs = 25;
    int batch_size = 32;
    double learning_rate = 0.0001;
    cnn.train(container, epochs, batch_size, learning_rate);

    // Step 6: Evaluate the neural network
    std::tuple<double, double> evaluation = cnn.evaluate(container);

    // Step 7: Making a single prediction
    std::vector<std::string> imagePaths = {"datasets/catsdogs/single_prediction/cat_or_dog_1.jpg", "datasets/catsdogs/single_prediction/cat_or_dog_2.jpg"};
    Eigen::Tensor<double, 4> singleImageBatch(batch_size, 3, targetHeight, targetWidth);
    singleImageBatch.setZero(); // Initialize with zeros

    for (size_t i = 0; i < imagePaths.size(); ++i)
    {
        cv::Mat image = cv::imread(imagePaths[i], cv::IMREAD_COLOR);
        if (image.empty())
        {
            throw std::runtime_error("Could not read the image: " + imagePaths[i]);
        }

        cv::resize(image, image, cv::Size(targetWidth, targetHeight));

        for (int h = 0; h < image.rows; ++h)
        {
            for (int w = 0; w < image.cols; ++w)
            {
                for (int c = 0; c < image.channels(); ++c)
                {
                    singleImageBatch(i, c, h, w) = static_cast<double>(image.at<cv::Vec3b>(h, w)[c]) / 255.0;
                }
            }
        }
    }

    Eigen::Tensor<double, 4> predictions = cnn.forward(singleImageBatch);

    for (size_t i = 0; i < imagePaths.size(); ++i)
    {
        double score = predictions(i, 0, 0, 0);
        std::string result = score >= 0.5 ? "cat" : "dog";
        double confidence = score >= 0.5 ? score * 100.0 : (1 - score) * 100.0;
        std::cout << "Prediction for " << imagePaths[i] << ": " << result << " (Score: " << score << ", Confidence: " << confidence << "%)\n";
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
