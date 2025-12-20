#include "ImageLoader.hpp"
#include "ImageAugmentor.hpp"
#include "NeuralNetwork.hpp"

void cnn_load_model()
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
    cnn.setLossFunction(LossType::CATEGORICAL_CROSS_ENTROPY);
    cnn.loadModel("examples/cifar10/model.onnx");

    cnn.setBatchSize(1);
    cnn.makeSinglePredictions(container);
}