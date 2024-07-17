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
    int targetWidth = 64;
    int targetHeight = 64;

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
    cnn.addConvolutionLayer(filters1, kernel_size1, stride1, padding1, ConvKernelInitialization::HE, ConvBiasInitialization::RANDOM_NORMAL);

    // Batch Normalization Layer 1
    // cnn.addBatchNormalizationLayer();

    // Activation Layer 1
    ActivationType activation1 = RELU;
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
    cnn.addConvolutionLayer(filters2, kernel_size2, stride2, padding2, ConvKernelInitialization::HE, ConvBiasInitialization::RANDOM_NORMAL);

    // Batch Normalization Layer 2
    // cnn.addBatchNormalizationLayer();

    // Activation Layer 2
    ActivationType activation2 = RELU;
    cnn.addActivationLayer(activation2);

    // Max Pooling Layer 2
    int pool_size2 = 2;
    int stride_pool2 = 2;
    cnn.addMaxPoolingLayer(pool_size2, stride_pool2);

    // Flatten Layer
    cnn.addFlattenLayer();

    // Fully Connected Layer 1
    int fc_output_size1 = 128;
    DenseWeightInitialization fc_weight_init1 = DenseWeightInitialization::HE;
    DenseBiasInitialization fc_bias_init1 = DenseBiasInitialization::RANDOM_NORMAL;
    cnn.addFullyConnectedLayer(fc_output_size1, fc_weight_init1, fc_bias_init1);

    // Batch Normalization Layer 3
    // cnn.addBatchNormalizationLayer();

    // Activation Layer 3
    ActivationType activation3 = RELU;
    cnn.addActivationLayer(activation3);

    // Fully Connected Layer 2
    int fc_output_size2 = 1;
    DenseWeightInitialization fc_weight_init2 = DenseWeightInitialization::XAVIER;
    DenseBiasInitialization fc_bias_init2 = DenseBiasInitialization::RANDOM_NORMAL;
    cnn.addFullyConnectedLayer(fc_output_size2, fc_weight_init2, fc_bias_init2);

    // Activation Layer 4
    ActivationType activation4 = SIGMOID;
    cnn.addActivationLayer(activation4);

    // Setting loss function
    LossType loss_type = LossType::BINARY_CROSS_ENTROPY;
    cnn.setLossFunction(loss_type);

    // Adam parameters
    std::unordered_map<std::string, double> adam_params = {
        {"beta1", 0.9},
        {"beta2", 0.999},
        {"epsilon", 1e-8}};

    // Compile the network with an optimizer
    std::unique_ptr<Optimizer> optimizer = Optimizer::create(Optimizer::Type::Adam, adam_params);
    cnn.compile(std::move(optimizer));

    // Categories for training and evaluation
    std::vector<std::string> categories = {"cats", "dogs"};

    // Step 5: Train the neural network
    int epochs = 25;
    double learning_rate = 0.00025;
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

    cv::resize(image, image, cv::Size(64, 64));

    // Convert image to Eigen matrix
    Eigen::MatrixXd singleImageBatch(1, image.rows * image.cols * image.channels());
    for (int i = 0; i < image.rows; ++i)
    {
        for (int j = 0; j < image.cols; ++j)
        {
            for (int c = 0; c < image.channels(); ++c)
            {
                singleImageBatch(0, i * image.cols * image.channels() + j * image.channels() + c) = image.at<cv::Vec3b>(i, j)[c] / 255.0;
            }
        }
    }

    std::cout << "Input dimensions for prediction: " << singleImageBatch.rows() << "x" << singleImageBatch.cols() << std::endl;

    Eigen::MatrixXd prediction = cnn.forward(singleImageBatch);
    std::string result = prediction(0, 0) >= 0.5 ? "dog" : "cat";

    std::cout << "Prediction for " << singleImagePath << ": " << result << " (Score: " << prediction(0, 0) << ")\n";
}

double testBinaryCrossEntropy()
{
    Eigen::MatrixXd predictions(4, 1);
    predictions << 0.5, 0.5, 0.5, 0.5; // Random guessing probabilities

    Eigen::MatrixXd targets(4, 1);
    targets << 0, 1, 0, 1; // True labels

    BinaryCrossEntropy lossFunc;
    double loss = lossFunc.compute(predictions, targets);
    return loss;
}

void testBatchNormalization()
{
    // Initialize sample data
    Eigen::MatrixXd input(4, 3);
    input << 1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.5, 9.0,
        10.0, 11.0, 14.0;

    Eigen::MatrixXd d_output(4, 3);
    d_output << 0.1, 0.2, 0.3,
        0.4, 0.5, 0.7,
        0.7, 0.9, 1.0,
        1.1, 1.3, 1.5;

    double epsilon = 1e-5;
    double momentum = 0.9;
    double learning_rate = 0.01;

    // Create BatchNormalizationLayer instance
    BatchNormalizationLayer bn_layer(epsilon, momentum);

    // Perform forward pass
    Eigen::MatrixXd output = bn_layer.forward(input);
    std::cout << "Forward pass output:\n"
              << output << std::endl;

    // Perform backward pass
    Eigen::MatrixXd d_input = bn_layer.backward(d_output, input, learning_rate);
    std::cout << "Backward pass d_input:\n"
              << d_input << std::endl;

    return;
}

int main(int argc, char **argv)
{
    testBatchNormalization();
    // throw std::runtime_error("AAA");
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

int mains()
{
    ConvolutionLayer convLayer(10, 3, 1, 1, ConvKernelInitialization::HE, ConvBiasInitialization::RANDOM_NORMAL);

    convLayer.setInputDepth(3);

    convLayer.initializeKernels(ConvKernelInitialization::HE);
    convLayer.initializeBiases(ConvBiasInitialization::RANDOM_NORMAL);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < 1; i++)
    {
        Eigen::MatrixXd input_batch = Eigen::MatrixXd::NullaryExpr(1, 3 * 32 * 32, [&]()
                                                                   { return dis(gen); });
        Eigen::MatrixXd d_output_batch = Eigen::MatrixXd::NullaryExpr(1, 10 * 32 * 32, [&]()
                                                                      { return dis(gen); });

        // Print the first few elements to verify values
        std::cout << "Input Batch (first 10 elements):" << std::endl;
        std::cout << input_batch.block(0, 0, 1, 10) << std::endl;

        std::cout << "d_output Batch (first 10 elements):" << std::endl;
        std::cout << d_output_batch.block(0, 0, 1, 10) << std::endl;

        convLayer.backward(d_output_batch, input_batch, 0.0001);
    }

    return 0;
}
