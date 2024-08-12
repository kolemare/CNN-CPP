#include "ImageLoader.hpp"
#include "ImageAugmentor.hpp"
#include "NeuralNetwork.hpp"

void cnn_cifar10_bn_e5()
{
    std::string datasetPath = "datasets/cifar10";

    ImageLoader loader;
    ImageContainer container;
    loader.loadImagesFromDirectory(datasetPath, container);

    int targetWidth = 32;
    int targetHeight = 32;

    ImageAugmentor augmentor(targetWidth, targetHeight);

    augmentor.augmentImages(container, AugmentTarget::NONE);

    NeuralNetwork cnn;
    cnn.setImageSize(targetWidth, targetHeight);

    cnn.setLogLevel(LogLevel::None);
    cnn.setProgressLevel(ProgressLevel::ProgressTime);

    cnn.addConvolutionLayer(32, 3);
    cnn.addBatchNormalizationLayer();
    cnn.addActivationLayer(ActivationType::RELU);
    cnn.addMaxPoolingLayer(2, 2);
    cnn.addConvolutionLayer(64, 3);
    cnn.addBatchNormalizationLayer();
    cnn.addActivationLayer(ActivationType::RELU);
    cnn.addMaxPoolingLayer(2, 2);
    cnn.addConvolutionLayer(64, 3);
    cnn.addBatchNormalizationLayer();
    cnn.addActivationLayer(ActivationType::RELU);
    cnn.addFlattenLayer();
    cnn.addFullyConnectedLayer(128);
    cnn.addBatchNormalizationLayer();
    cnn.addActivationLayer(ActivationType::RELU);
    cnn.addFullyConnectedLayer(128);
    cnn.addBatchNormalizationLayer();
    cnn.addActivationLayer(ActivationType::RELU);
    cnn.addFullyConnectedLayer(64);
    cnn.addBatchNormalizationLayer();
    cnn.addActivationLayer(ActivationType::RELU);
    cnn.addFullyConnectedLayer(10);
    cnn.addActivationLayer(ActivationType::SOFTMAX);

    cnn.setLossFunction(LossType::CATEGORICAL_CROSS_ENTROPY);
    cnn.enableGradientClipping();
    cnn.compile(OptimizerType::Adam);

    int epochs = 5;
    int batch_size = 50;
    double learning_rate = 0.0001;
    cnn.train(container, epochs, batch_size, learning_rate);
    cnn.makeSinglePredictions(container);
}