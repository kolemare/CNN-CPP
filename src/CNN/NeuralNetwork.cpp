#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork()
{
    this->flattenAdded = false;
    this->clippingSet = false;
    this->elralesSet = false;
    this->compiled = false;
    this->trained = false;
    this->batchSize = 0;
    this->currentDepth = 3;
    this->logLevel = LogLevel::None;
    this->progressLevel = ProgressLevel::None;
    outputCSV = "logs/cnn.csv";
}

void NeuralNetwork::setImageSize(const int targetWidth,
                                 const int targetHeight)
{
    inputHeight = targetHeight;
    inputWidth = targetWidth;
    this->hardReset();
}

void NeuralNetwork::setCSVPath(std::string outputCSV)
{
    this->outputCSV = outputCSV;
    std::cout << "Output CSV relative path => " << outputCSV << std::endl;
}

void NeuralNetwork::setLogLevel(LogLevel level)
{
    logLevel = level;
    this->hardReset();
}

void NeuralNetwork::setProgressLevel(ProgressLevel level)
{
    progressLevel = level;
    this->hardReset();
}

void NeuralNetwork::addConvolutionLayer(int filters,
                                        int kernel_size,
                                        int stride,
                                        int padding,
                                        ConvKernelInitialization kernel_init,
                                        ConvBiasInitialization bias_init)
{
    layers.push_back(std::make_shared<ConvolutionLayer>(filters, kernel_size, stride, padding, kernel_init, bias_init));
    if (LogLevel::LayerSummary == logLevel)
    {
        std::cout << "Added Convolution Layer with " << filters << " filters, kernel size " << kernel_size << ", stride " << stride << ", padding " << padding << std::endl;
    }
    this->hardReset();
}

void NeuralNetwork::addMaxPoolingLayer(int pool_size,
                                       int stride)
{
    layers.push_back(std::make_shared<MaxPoolingLayer>(pool_size, stride));
    if (LogLevel::LayerSummary == logLevel)
    {
        std::cout << "Added Max Pooling Layer with pool size " << pool_size << ", stride " << stride << std::endl;
    }
    this->hardReset();
}

void NeuralNetwork::addAveragePoolingLayer(int pool_size,
                                           int stride)
{
    layers.push_back(std::make_shared<AveragePoolingLayer>(pool_size, stride));
    if (LogLevel::LayerSummary == logLevel)
    {
        std::cout << "Added Average Pooling Layer with pool size " << pool_size << ", stride " << stride << std::endl;
    }
    this->hardReset();
}

void NeuralNetwork::addFlattenLayer()
{
    if (!flattenAdded)
    {
        layers.push_back(std::make_shared<FlattenLayer>());
        flattenAdded = true;
        if (LogLevel::LayerSummary == logLevel)
        {
            std::cout << "Added Flatten Layer" << std::endl;
        }
    }
    else
    {
        std::cerr << "Flatten layer already added." << std::endl;
    }
    this->hardReset();
}

void NeuralNetwork::addFullyConnectedLayer(int output_size,
                                           DenseWeightInitialization weight_init,
                                           DenseBiasInitialization bias_init)
{
    layers.push_back(std::make_shared<FullyConnectedLayer>(output_size, weight_init, bias_init));
    if (LogLevel::LayerSummary == logLevel)
    {
        std::cout << "Added Fully Connected Layer with output size " << output_size << std::endl;
    }
    this->hardReset();
}

void NeuralNetwork::addActivationLayer(ActivationType type)
{
    layers.push_back(std::make_shared<ActivationLayer>(type));
    if (LogLevel::LayerSummary == logLevel)
    {
        std::cout << "Added Activation Layer of type " << static_cast<int>(type) << std::endl;
    }
    this->hardReset();
}

void NeuralNetwork::addBatchNormalizationLayer(double epsilon,
                                               double momentum)
{
    layers.push_back(std::make_shared<BatchNormalizationLayer>(epsilon, momentum));
    if (LogLevel::LayerSummary == logLevel)
    {
        std::cout << "Added Batch Normalization Layer" << std::endl;
    }
    this->hardReset();
}

void NeuralNetwork::setLossFunction(LossType type)
{
    lossFunction = LossFunction::create(type);
    if (LogLevel::LayerSummary == logLevel)
    {
        std::cout << "Set Loss Function of type " << static_cast<int>(type) << std::endl;
    }
    this->hardReset();
}

void NeuralNetwork::compile(OptimizerType optimizerType,
                            const std::unordered_map<std::string, double> &optimizer_params)
{
    std::unordered_map<std::string, double> default_params;

    switch (optimizerType)
    {
    case OptimizerType::SGD:
        // No parameters needed for SGD, empty map
        break;
    case OptimizerType::SGDWithMomentum:
        default_params = {{"momentum", 0.9}};
        break;
    case OptimizerType::Adam:
        default_params = {{"beta1", 0.9}, {"beta2", 0.999}, {"epsilon", 1e-7}};
        break;
    case OptimizerType::RMSprop:
        default_params = {{"beta", 0.9}, {"epsilon", 1e-7}};
        break;
    default:
        throw std::invalid_argument("Unknown optimizer type");
    }

    // Combine provided params with defaults, preferring provided params
    for (const auto &param : optimizer_params)
    {
        default_params[param.first] = param.second;
    }

    optimizer = Optimizer::create(optimizerType, default_params);

    int height = inputHeight;
    int width = inputWidth;
    int input_size = -1;

    if (!lossFunction)
    {
        // Loss function must be set before compilation
        throw std::runtime_error("Loss function must be set before compiling.");
    }

    if (!clippingSet)
    {
        // Default => GradientClipping DISABLED
        this->enableGradientClipping(0, GradientClippingMode::DISABLED);
    }

    if (!elralesSet)
    {
        // Default => ELRALES DISABLED
        this->enableELRALES(0.0, 0, 0, 0.0, ELRALES_Mode::DISABLED);
    }

    if (ELRALES_Mode::ENABLED == elralesMode && LearningDecayType::NONE != learningDecayMode)
    {
        throw std::runtime_error("Cannot use both ELRALES and LearningDecay simultaneously.");
    }

    for (size_t i = 0; i < layers.size(); ++i)
    {
        if (auto conv_layer = dynamic_cast<ConvolutionLayer *>(layers[i].get()))
        {
            conv_layer->setInputDepth(currentDepth);
            currentDepth = conv_layer->getFilters();
            height = (height - conv_layer->getKernelSize() + 2 * conv_layer->getPadding()) / conv_layer->getStride() + 1;
            width = (width - conv_layer->getKernelSize() + 2 * conv_layer->getPadding()) / conv_layer->getStride() + 1;
            conv_layer->setOptimizer(optimizer);
        }
        else if (auto pool_layer = dynamic_cast<MaxPoolingLayer *>(layers[i].get()))
        {
            height = (height - pool_layer->getPoolSize()) / pool_layer->getStride() + 1;
            width = (width - pool_layer->getPoolSize()) / pool_layer->getStride() + 1;
        }
        else if (auto pool_layer = dynamic_cast<AveragePoolingLayer *>(layers[i].get()))
        {
            height = (height - pool_layer->getPoolSize()) / pool_layer->getStride() + 1;
            width = (width - pool_layer->getPoolSize()) / pool_layer->getStride() + 1;
        }
        else if (auto fc_layer = dynamic_cast<FullyConnectedLayer *>(layers[i].get()))
        {
            if (input_size == -1)
            {
                throw std::runtime_error("Input size for FullyConnectedLayer cannot be determined.");
            }
            fc_layer->setInputSize(input_size);
            input_size = fc_layer->getOutputSize();
            fc_layer->setOptimizer(optimizer);
        }
        else if (dynamic_cast<FlattenLayer *>(layers[i].get()))
        {
            input_size = height * width * currentDepth;
        }
    }
    compiled = true;
}

Eigen::Tensor<double, 4> NeuralNetwork::forward(const Eigen::Tensor<double, 4> &input)
{
    if (LogLevel::LayerSummary == logLevel)
    {
        NNLogger::printTensorSummary(input, "INPUT", PropagationType::FORWARD);
    }

    Eigen::Tensor<double, 4> output = input;
    layerInputs.clear();

    for (size_t i = 0; i < layers.size(); ++i)
    {
        layerInputs.push_back(output);
        output = layers[i]->forward(output);

        if (LogLevel::LayerSummary == logLevel)
        {
            std::string layerType;
            if (dynamic_cast<ConvolutionLayer *>(layers[i].get()))
            {
                layerType = "Convolution Layer";
            }
            else if (dynamic_cast<MaxPoolingLayer *>(layers[i].get()))
            {
                layerType = "Max Pooling Layer";
            }
            else if (dynamic_cast<AveragePoolingLayer *>(layers[i].get()))
            {
                layerType = "Average Pooling Layer";
            }
            else if (dynamic_cast<FlattenLayer *>(layers[i].get()))
            {
                layerType = "Flatten Layer";
            }
            else if (dynamic_cast<FullyConnectedLayer *>(layers[i].get()))
            {
                layerType = "Fully Connected Layer";
            }
            else if (dynamic_cast<ActivationLayer *>(layers[i].get()))
            {
                layerType = "Activation Layer";
            }
            else if (dynamic_cast<BatchNormalizationLayer *>(layers[i].get()))
            {
                layerType = "Batch Normalization Layer";
            }

            if (LogLevel::FullTensor == logLevel)
            {
                NNLogger::printFullTensor(output, layerType, PropagationType::FORWARD);
            }
            else if (LogLevel::LayerSummary == logLevel)
            {
                NNLogger::printTensorSummary(output, layerType, PropagationType::FORWARD);
            }
        }
    }

    return output;
}

void NeuralNetwork::backward(const Eigen::Tensor<double, 4> &d_output,
                             double learning_rate)
{
    if (LogLevel::LayerSummary == logLevel)
    {
        NNLogger::printTensorSummary(d_output, "OUTPUT", PropagationType::BACK);
    }

    Eigen::Tensor<double, 4> d_input = d_output;

    for (int i = layers.size() - 1; i >= 0; --i)
    {
        std::string layerType;
        if (dynamic_cast<ConvolutionLayer *>(layers[i].get()))
        {
            layerType = "Convolution Layer";
        }
        else if (dynamic_cast<MaxPoolingLayer *>(layers[i].get()))
        {
            layerType = "Max Pooling Layer";
        }
        else if (dynamic_cast<AveragePoolingLayer *>(layers[i].get()))
        {
            layerType = "Average Pooling Layer";
        }
        else if (dynamic_cast<FlattenLayer *>(layers[i].get()))
        {
            layerType = "Flatten Layer";
        }
        else if (dynamic_cast<FullyConnectedLayer *>(layers[i].get()))
        {
            layerType = "Fully Connected Layer";
        }
        else if (dynamic_cast<ActivationLayer *>(layers[i].get()))
        {
            layerType = "Activation Layer";
        }
        else if (dynamic_cast<BatchNormalizationLayer *>(layers[i].get()))
        {
            layerType = "Batch Normalization Layer";
        }

        d_input = layers[i]->backward(d_input, layerInputs[i], learning_rate);

        if (GradientClippingMode::ENABLED == clippingMode)
        {
            GradientClipping::clipGradients(d_input, clipValue);
        }

        if (LogLevel::FullTensor == logLevel)
        {
            NNLogger::printFullTensor(d_input, layerType, PropagationType::BACK);
        }
        else if (LogLevel::LayerSummary == logLevel)
        {
            NNLogger::printTensorSummary(d_input, layerType, PropagationType::BACK);
        }
    }
}

void NeuralNetwork::train(const ImageContainer &imageContainer,
                          int epochs,
                          int batch_size,
                          double learning_rate)
{
    if (!compiled)
    {
        throw std::runtime_error("Network must be compiled before training.");
    }
    if (!lossFunction)
    {
        throw std::runtime_error("Loss function must be set before training.");
    }

    this->batchSize = batch_size;

    NNLogger::initializeCSV(outputCSV);

    BatchManager batchManager(imageContainer, batch_size, BatchType::Training);
    std::cout << "Training started..." << std::endl;
    auto start = std::chrono::steady_clock::now();
    double current_learning_rate = learning_rate;

    double cumulative_loss = 0.0;
    int total_batches_completed = 0;

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        if (LearningDecayType::NONE != learningDecayMode && learningDecay)
        {
            current_learning_rate = learningDecay->computeLearningRate(learning_rate, epoch);
            std::cout << "Learning rate during epoch " << epoch + 1 << ": " << current_learning_rate << std::endl;
        }
        Eigen::Tensor<double, 4> batch_input;
        Eigen::Tensor<int, 2> batch_label;
        int totalBatches = batchManager.getTotalBatches();
        int batchCounter = 0;
        double total_epoch_loss = 0.0;
        int correct_predictions = 0;
        int num_epoch_samples = 0;

        while (batchManager.getNextBatch(batch_input, batch_label))
        {
            // Forward pass
            Eigen::Tensor<double, 4> predictions = forward(batch_input);

            // Compute loss
            double batch_loss = lossFunction->compute(predictions, batch_label);
            total_epoch_loss += batch_loss * batch_input.dimension(0);

            // Count correct predictions
            for (int i = 0; i < predictions.dimension(0); ++i)
            {
                int predicted_label;
                int true_label;

                if (predictions.dimension(3) == 1) // Binary classification
                {
                    predicted_label = (predictions(i, 0, 0, 0) >= 0.5) ? 1 : 0;
                    true_label = batch_label(i, 0);
                }
                else // Multi-class classification
                {
                    predicted_label = 0;
                    double max_value = predictions(i, 0, 0, 0);

                    for (int j = 1; j < predictions.dimension(3); ++j)
                    {
                        if (predictions(i, 0, 0, j) > max_value)
                        {
                            max_value = predictions(i, 0, 0, j);
                            predicted_label = j;
                        }
                    }

                    true_label = 0;
                    for (int j = 0; j < batch_label.dimension(1); ++j)
                    {
                        if (batch_label(i, j) == 1)
                        {
                            true_label = j;
                            break;
                        }
                    }
                }

                if (predicted_label == true_label)
                {
                    correct_predictions++;
                }
                num_epoch_samples++;
            }

            // Backward pass
            Eigen::Tensor<double, 4> d_output = lossFunction->derivative(predictions, batch_label);
            backward(d_output, current_learning_rate);

            if (ProgressLevel::None != progressLevel)
            {
                NNLogger::printProgress(epoch, epochs, batchCounter, totalBatches, start, batch_loss, progressLevel, cumulative_loss, total_batches_completed);
            }

            batchCounter++;
        }

        double average_loss = total_epoch_loss / num_epoch_samples;
        double accuracy = static_cast<double>(correct_predictions) / num_epoch_samples;

        std::cout << "Evaluating..." << std::endl;
        std::tuple<double, double> evaluation = evaluate(imageContainer);

        if (ELRALES_Mode::ENABLED == elralesMode)
        {
            ELRALES_Retval elralesEvaluation = elrales->updateState(average_loss, layers, current_learning_rate, elralesStateMachine);
            std::string elralesState = toString(elralesStateMachine);

            if (ELRALES_Retval::SUCCESSFUL_EPOCH == elralesEvaluation)
            {
                std::cout << "Epoch " << epoch + 1 << " completed." << std::endl;
                std::cout << "Training Accuracy: " << accuracy << std::endl;
                std::cout << "Training Loss: " << average_loss << std::endl;
                std::cout << "Testing Accuracy: " << std::get<0>(evaluation) << std::endl;
                std::cout << "Testing Loss: " << std::get<1>(evaluation) << std::endl;
                std::cout << "ELRALES: " << elralesState << std::endl;
                NNLogger::appendToCSV(outputCSV, epoch + 1, accuracy, average_loss, std::get<0>(evaluation), std::get<1>(evaluation), elralesState);
            }
            else if (ELRALES_Retval::WASTED_EPOCH == elralesEvaluation)
            {
                std::cout << "Wasted Epoch " << epoch + 1 << " completed." << std::endl;
                std::cout << "Wasted Training Accuracy: " << accuracy << std::endl;
                std::cout << "Wasted Training Loss: " << average_loss << std::endl;
                std::cout << "Wasted Testing Accuracy: " << std::get<0>(evaluation) << std::endl;
                std::cout << "Wasted Testing Loss: " << std::get<1>(evaluation) << std::endl;
                std::cout << "ELRALES: " << elralesState << std::endl;
                NNLogger::appendToCSV(outputCSV, epoch + 1, accuracy, average_loss, std::get<0>(evaluation), std::get<1>(evaluation), elralesState);
                ++epochs; // This ensures the number of successful epochs remains constant
            }
            else if (ELRALES_Retval::END_LEARNING == elralesEvaluation)
            {
                std::cout << "EarlyStopping Epoch " << epoch + 1 << " completed." << std::endl;
                std::cout << "EarlyStopping Training Accuracy: " << accuracy << std::endl;
                std::cout << "EarlyStopping Training Loss: " << average_loss << std::endl;
                std::cout << "EarlyStopping Testing Accuracy: " << std::get<0>(evaluation) << std::endl;
                std::cout << "EarlyStopping Testing Loss: " << std::get<1>(evaluation) << std::endl;
                std::cout << "ELRALES: " << elralesState << std::endl;
                NNLogger::appendToCSV(outputCSV, epoch + 1, accuracy, average_loss, std::get<0>(evaluation), std::get<1>(evaluation), elralesState);
                break;
            }
            elralesStateMachineTimeLine.push_back(static_cast<ELRALES_StateMachine>(elralesStateMachine));
        }
        else
        {
            std::cout << "Epoch " << epoch + 1 << " completed." << std::endl;
            std::cout << "Training Accuracy: " << accuracy << std::endl;
            std::cout << "Training Loss: " << average_loss << std::endl;
            std::cout << "Testing Accuracy: " << std::get<0>(evaluation) << std::endl;
            std::cout << "Testing Loss: " << std::get<1>(evaluation) << std::endl;
            std::cout << "ELRALES: OFF" << std::endl;
            NNLogger::appendToCSV(outputCSV, epoch + 1, accuracy, average_loss, std::get<0>(evaluation), std::get<1>(evaluation), "OFF");
        }
    }

    // After the training loop
    std::cout << std::endl;
    std::cout << "Training ended!" << std::endl;
    std::cout << std::endl;
    trained = true;
}

std::tuple<double, double> NeuralNetwork::evaluate(const ImageContainer &imageContainer)
{
    if (!compiled)
    {
        throw std::runtime_error("Network must be compiled before evaluation.");
    }
    if (!lossFunction)
    {
        throw std::runtime_error("Loss function must be set before evaluation.");
    }

    BatchManager batchManager(imageContainer, imageContainer.getTestImages().size(), BatchType::Testing);
    Eigen::Tensor<double, 4> batch_input;
    Eigen::Tensor<int, 2> batch_label;

    double total_loss = 0.0;
    int correct_predictions = 0;
    int num_samples = 0;

    while (batchManager.getNextBatch(batch_input, batch_label))
    {
        Eigen::Tensor<double, 4> predictions = forward(batch_input);

        double batch_loss = lossFunction->compute(predictions, batch_label);
        total_loss += batch_loss * batch_input.dimension(0);

        // Count correct predictions
        for (int i = 0; i < predictions.dimension(0); ++i)
        {
            int predicted_label;
            int true_label;

            if (predictions.dimension(3) == 1) // Binary classification
            {
                predicted_label = (predictions(i, 0, 0, 0) >= 0.5) ? 1 : 0;
                true_label = batch_label(i, 0);
            }
            else // Multi-class classification
            {
                predicted_label = 0;
                double max_value = predictions(i, 0, 0, 0);

                for (int j = 1; j < predictions.dimension(3); ++j)
                {
                    if (predictions(i, 0, 0, j) > max_value)
                    {
                        max_value = predictions(i, 0, 0, j);
                        predicted_label = j;
                    }
                }

                true_label = 0;
                for (int j = 0; j < batch_label.dimension(1); ++j)
                {
                    if (batch_label(i, j) == 1)
                    {
                        true_label = j;
                        break;
                    }
                }
            }

            if (predicted_label == true_label)
            {
                correct_predictions++;
            }
            num_samples++;
        }
    }

    double average_loss = total_loss / num_samples;
    double accuracy = static_cast<double>(correct_predictions) / num_samples;

    return std::make_tuple(accuracy, average_loss);
}

void NeuralNetwork::makeSinglePredictions(const ImageContainer &imageContainer)
{
    if (!compiled)
    {
        throw std::runtime_error("Network must be compiled before making single predictions.");
    }
    if (!trained)
    {
        throw std::runtime_error("Network must be trained before making single predictions.");
    }
    if (0 == batchSize)
    {
        throw std::runtime_error("Bad batch size, unknown error.");
    }

    // Create a batch manager for single prediction
    BatchManager batchManager(imageContainer, batchSize, BatchType::Testing);
    batchManager.loadSinglePredictionBatch();

    // Process each batch of single prediction images
    while (true)
    {
        Eigen::Tensor<double, 4> batchImages;
        Eigen::Tensor<int, 2> batchLabels;

        // Get a batch of single prediction images and their names
        std::vector<std::string> imageNames = batchManager.getSinglePredictionBatch(batchImages, batchLabels);

        if (imageNames.empty())
        {
            break; // No more images to process
        }

        // Perform forward pass to get predictions
        Eigen::Tensor<double, 4> predictions = forward(batchImages);

        for (int i = 0; i < imageNames.size(); ++i)
        {
            const std::string &imageName = imageNames[i];

            if (imageName.empty())
            {
                continue; // Skip empty slots
            }

            std::string predictedCategory;
            double confidence = 0.0;

            if (predictions.dimension(3) == 1) // Binary classification
            {
                double score = predictions(i, 0, 0, 0);

                // Interpret the prediction
                predictedCategory = score >= 0.5 ? batchManager.getCategoryName(0) : batchManager.getCategoryName(1);
                confidence = score >= 0.5 ? score * 100.0 : (1 - score) * 100.0;
            }
            else // Multi-class classification
            {
                int predictedLabel = 0;
                double maxConfidence = predictions(i, 0, 0, 0);

                for (int j = 1; j < predictions.dimension(3); ++j)
                {
                    if (predictions(i, 0, 0, j) > maxConfidence)
                    {
                        maxConfidence = predictions(i, 0, 0, j);
                        predictedLabel = j;
                    }
                }

                predictedCategory = batchManager.getCategoryName(predictedLabel);
                confidence = maxConfidence * 100.0;
            }

            std::cout << "Prediction for \"" << imageName << "\" is \"" << predictedCategory
                      << "\" with confidence " << confidence << "%." << std::endl;
        }
    }
}

void NeuralNetwork::enableGradientClipping(double value,
                                           GradientClippingMode mode)
{
    clippingMode = mode;
    clipValue = value;
    clippingSet = true;
    if (GradientClippingMode::ENABLED == mode)
    {
        std::cout << "|Gradient Clipping: " << value << "|" << std::endl;
    }
    else
    {
        std::cout << "|Gradient Clipping Disabled|" << std::endl;
    }
    this->hardReset();
}

void NeuralNetwork::enableELRALES(double learning_rate_coef,
                                  int maxSuccessiveEpochFailures,
                                  int maxEpochFailures,
                                  double tolerance,
                                  ELRALES_Mode mode)
{
    this->elralesMode = mode;
    this->elralesSet = true;
    this->elralesStateMachine = ELRALES_StateMachine::NORMAL;
    this->learning_rate_coef = learning_rate_coef;
    this->maxSuccessiveEpochFailures = maxSuccessiveEpochFailures;
    this->maxEpochFailures = maxEpochFailures;
    this->tolerance = tolerance;

    if (LearningDecayType::NONE != learningDecayMode && ELRALES_Mode::ENABLED == elralesMode)
    {
        throw std::runtime_error("Cannot use ELRALES when LearningDecay is enabled.");
    }

    if (ELRALES_Mode::ENABLED == mode)
    {
        this->elrales = std::make_unique<ELRALES>(learning_rate_coef, maxSuccessiveEpochFailures, maxEpochFailures, tolerance, layers);
        std::cout << "|ELRALES Enabled with LRC: " << learning_rate_coef
                  << ", MSEF: " << maxSuccessiveEpochFailures
                  << ", MEF: " << maxEpochFailures
                  << ", TOL: " << tolerance
                  << "|" << std::endl;
    }
    if (ELRALES_Mode::ENABLED != mode && ELRALES_Mode::DISABLED != mode)
    {
        throw std::runtime_error("Unknown ELRALES mode.");
    }
    elralesStateMachineTimeLine.push_back(static_cast<ELRALES_StateMachine>(elralesStateMachine));
    this->hardReset();
}

void NeuralNetwork::enableLearningDecay(LearningDecayType decayType,
                                        const std::unordered_map<std::string, double> &params)
{
    if (ELRALES_Mode::ENABLED == elralesMode)
    {
        throw std::runtime_error("Cannot use LearningDecay when ELRALES is enabled.");
    }

    learningDecayMode = decayType;
    learningDecay = std::make_unique<LearningDecay>(decayType, params);
    std::cout << "|Learning Decay Enabled with Type: " << toString(decayType) << "|" << std::endl;
    this->hardReset();
}

void NeuralNetwork::hardReset()
{
    this->compiled = false;
    this->trained = false;
    this->currentDepth = 3;
    this->batchSize = 0;
}
