#include "ImageLoader.hpp"
#include "ImageAugmentor.hpp"
#include "NeuralNetwork.hpp"

void cnn_sanity_2_colors()
{
    std::string datasetPath = "datasets/sanity_2_colors";

    ImageLoader loader;
    ImageContainer container;
    loader.loadImagesFromDirectory(datasetPath, container);

    int targetWidth = 16;
    int targetHeight = 16;

    ImageAugmentor augmentor(targetWidth, targetHeight);

    augmentor.augmentImages(container, AugmentTarget::NONE);

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

    cnn.addFullyConnectedLayer(16);
    cnn.addActivationLayer(ActivationType::RELU);

    cnn.addFullyConnectedLayer(8);
    cnn.addActivationLayer(ActivationType::RELU);

    cnn.addFullyConnectedLayer(1);
    cnn.addActivationLayer(ActivationType::SIGMOID);

    cnn.setLossFunction(LossType::BINARY_CROSS_ENTROPY);
    cnn.setBatchMode(BatchMode::UniformDistribution);
    cnn.enableGradientClipping();
    cnn.compile(OptimizerType::Adam);

    int epochs = 5;
    int batch_size = 10;
    cnn.train(container, epochs, batch_size);
    cnn.makeSinglePredictions(container);
}