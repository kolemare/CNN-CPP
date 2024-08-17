#include "ImageLoader.hpp"
#include "ImageAugmentor.hpp"
#include "NeuralNetwork.hpp"

void cnn_cd_ld_bn_e25()
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
    cnn.addBatchNormalizationLayer();
    cnn.addActivationLayer(ActivationType::RELU);
    cnn.addMaxPoolingLayer(2, 2);
    cnn.addConvolutionLayer(32, 3);
    cnn.addBatchNormalizationLayer();
    cnn.addActivationLayer(ActivationType::RELU);
    cnn.addMaxPoolingLayer(2, 2);
    cnn.addFlattenLayer();
    cnn.addFullyConnectedLayer(128);
    cnn.addBatchNormalizationLayer();
    cnn.addActivationLayer(ActivationType::RELU);
    cnn.addFullyConnectedLayer(64);
    cnn.addBatchNormalizationLayer();
    cnn.addActivationLayer(ActivationType::RELU);
    cnn.addFullyConnectedLayer(32);
    cnn.addBatchNormalizationLayer();
    cnn.addActivationLayer(ActivationType::RELU);
    cnn.addFullyConnectedLayer(16);
    cnn.addBatchNormalizationLayer();
    cnn.addActivationLayer(ActivationType::RELU);
    cnn.addFullyConnectedLayer(8);
    cnn.addBatchNormalizationLayer();
    cnn.addActivationLayer(ActivationType::RELU);
    cnn.addFullyConnectedLayer(1);
    cnn.addActivationLayer(ActivationType::SIGMOID);
    cnn.setLossFunction(LossType::BINARY_CROSS_ENTROPY);
    cnn.enableGradientClipping();

    std::unordered_map<std::string, double> polynomial_params = {
        {"end_learning_rate", 0.00001},
        {"decay_steps", 25},
        {"power", 2.0}};

    cnn.enableLearningDecay(LearningDecayType::POLYNOMIAL, polynomial_params);
    cnn.compile(OptimizerType::Adam);

    int epochs = 25;
    int batch_size = 32;
    double learning_rate = 0.0001;
    cnn.train(container, epochs, batch_size, learning_rate);
    cnn.makeSinglePredictions(container);
}