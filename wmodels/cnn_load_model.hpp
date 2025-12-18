#include "ImageLoader.hpp"
#include "ImageAugmentor.hpp"
#include "NeuralNetwork.hpp"

void cnn_load_model()
{
    std::string datasetPath = "datasets/sanity_5_shapes";

    ImageLoader loader;
    ImageContainer container;
    loader.loadImagesFromDirectory(datasetPath, container);

    int targetWidth = 16;
    int targetHeight = 16;

    ImageAugmentor augmentor(targetWidth, targetHeight);

    augmentor.augmentImages(container, AugmentTarget::NONE);

    NeuralNetwork cnn;
    cnn.setImageSize(targetWidth, targetHeight);
    cnn.loadModel("torchtest/model.onnx");

    cnn.setBatchSize(1);
    cnn.makeSinglePredictions(container);
}