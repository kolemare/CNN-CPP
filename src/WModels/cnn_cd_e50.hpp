#include "ImageLoader.hpp"
#include "ImageAugmentor.hpp"
#include "NeuralNetwork.hpp"

void cnn_cd_e50()
{
    std::string datasetPath = "datasets/catsdogs";

    ImageLoader loader;
    ImageContainer container;
    loader.loadImagesFromDirectory(datasetPath, container);

    int targetWidth = 32;
    int targetHeight = 32;

    ImageAugmentor augmentor(targetWidth, targetHeight);

    augmentor.setZoomChance(1.0f);
    augmentor.setHorizontalFlipChance(0.5f);
    augmentor.setVerticalFlipChance(0.0f);
    augmentor.setGaussianNoiseChance(0.0f);
    augmentor.setGaussianBlurChance(0.0f);
    augmentor.setShearChance(1.0f);

    augmentor.augmentImages(container, AugmentTarget::TRAIN_DATASET);

    NeuralNetwork cnn;
    cnn.setImageSize(targetWidth, targetHeight);

    cnn.setLogLevel(LogLevel::None);
    cnn.setProgressLevel(ProgressLevel::ProgressTime);

    cnn.addConvolutionLayer(32, 3);
    cnn.addActivationLayer(ActivationType::RELU);
    cnn.addMaxPoolingLayer(2, 2);
    cnn.addConvolutionLayer(32, 3);
    cnn.addActivationLayer(ActivationType::RELU);
    cnn.addMaxPoolingLayer(2, 2);
    cnn.addFlattenLayer();
    cnn.addFullyConnectedLayer(128);
    cnn.addActivationLayer(ActivationType::RELU);
    cnn.addFullyConnectedLayer(64);
    cnn.addActivationLayer(ActivationType::RELU);
    cnn.addFullyConnectedLayer(32);
    cnn.addActivationLayer(ActivationType::RELU);
    cnn.addFullyConnectedLayer(1);
    cnn.addActivationLayer(ActivationType::SIGMOID);

    cnn.setLossFunction(LossType::BINARY_CROSS_ENTROPY);
    cnn.enableGradientClipping();
    cnn.compile(OptimizerType::Adam);

    int epochs = 50;
    int batch_size = 32;
    double learning_rate = 0.000025;
    cnn.train(container, epochs, batch_size, learning_rate);
    cnn.makeSinglePredictions(container);
}