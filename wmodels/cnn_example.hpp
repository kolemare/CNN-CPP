#include "ImageLoader.hpp"
#include "ImageAugmentor.hpp"
#include "NeuralNetwork.hpp"

void cnn_example()
{
    std::string datasetPath = "datasets/catsdogs";

    // Step 1: Load the images
    ImageLoader loader;
    ImageContainer container;
    loader.loadImagesFromDirectory(datasetPath, container);

    int targetWidth = 64;
    int targetHeight = 64;

    ImageAugmentor augmentor(targetWidth, targetHeight);

    // Configuration for augmentation
    augmentor.setZoomFactor(1.2f);
    augmentor.setHorizontalFlip(true);
    augmentor.setVerticalFlip(true);
    augmentor.setGaussianNoiseStdDev(10.0f);
    augmentor.setGaussianBlurKernelSize(5);
    augmentor.setShearRange(0.2f);

    // Set augmentation chances
    augmentor.setZoomChance(1.0f);
    augmentor.setHorizontalFlipChance(0.5f);
    augmentor.setVerticalFlipChance(0.0f);
    augmentor.setGaussianNoiseChance(0.0f);
    augmentor.setGaussianBlurChance(0.0f);
    augmentor.setShearChance(1.0f);

    // Augment images
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
    cnn.addConvolutionLayer(filters2, kernel_size2, stride2, padding2, ConvKernelInitialization::HE, ConvBiasInitialization::RANDOM_NORMAL);

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
    int fc_output_size2 = 64;
    DenseWeightInitialization fc_weight_init2 = DenseWeightInitialization::HE;
    DenseBiasInitialization fc_bias_init2 = DenseBiasInitialization::RANDOM_NORMAL;
    cnn.addFullyConnectedLayer(fc_output_size2, fc_weight_init2, fc_bias_init2);

    // Activation Layer 4
    ActivationType activation4 = ActivationType::RELU;
    cnn.addActivationLayer(activation4);

    // Fully Connected Layer 3
    int fc_output_size3 = 32;
    DenseWeightInitialization fc_weight_init3 = DenseWeightInitialization::RANDOM_NORMAL;
    DenseBiasInitialization fc_bias_init3 = DenseBiasInitialization::RANDOM_NORMAL;
    cnn.addFullyConnectedLayer(fc_output_size3, fc_weight_init3, fc_bias_init3);

    // Activation Layer 5
    ActivationType activation5 = ActivationType::RELU;
    cnn.addActivationLayer(activation5);

    // Fully Connected Layer 4
    int fc_output_size4 = 1;
    DenseWeightInitialization fc_weight_init4 = DenseWeightInitialization::XAVIER;
    DenseBiasInitialization fc_bias_init4 = DenseBiasInitialization::RANDOM_NORMAL;
    cnn.addFullyConnectedLayer(fc_output_size4, fc_weight_init4, fc_bias_init4);

    // Activation Layer 6
    ActivationType activation6 = ActivationType::SIGMOID;
    cnn.addActivationLayer(activation6);

    // Setting loss function
    LossType loss_type = LossType::BINARY_CROSS_ENTROPY;
    cnn.setLossFunction(loss_type);

    // Enable Default Gradient Clipping
    cnn.enableGradientClipping(1.0);

    // Compile the network with an optimizer
    cnn.compile(OptimizerType::Adam);

    // Step 5: Train the neural network
    int epochs = 50;
    int batch_size = 32;
    double learning_rate = 0.00005;
    cnn.train(container, epochs, batch_size, learning_rate);
    cnn.makeSinglePredictions(container);
}