#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork() : flattenAdded(false) {}

void NeuralNetwork::addConvolutionLayer(int filters, int kernel_size, int input_depth, int stride, int padding, const Eigen::VectorXd &biases)
{
    layers.push_back(std::make_shared<ConvolutionLayer>(filters, kernel_size, input_depth, stride, padding, biases));
}

void NeuralNetwork::addMaxPoolingLayer(int pool_size, int stride)
{
    layers.push_back(std::make_shared<MaxPoolingLayer>(pool_size, stride));
}

void NeuralNetwork::addAveragePoolingLayer(int pool_size, int stride)
{
    layers.push_back(std::make_shared<AveragePoolingLayer>(pool_size, stride));
}

void NeuralNetwork::addFlattenLayer()
{
    if (!flattenAdded)
    {
        layers.push_back(std::make_shared<FlattenLayer>());
        flattenAdded = true;
    }
    else
    {
        std::cerr << "Flatten layer already added." << std::endl;
    }
}

void NeuralNetwork::addFullyConnectedLayer(int input_size, int output_size, std::unique_ptr<Optimizer> optimizer)
{
    layers.push_back(std::make_shared<FullyConnectedLayer>(input_size, output_size, std::move(optimizer)));
}

void NeuralNetwork::addActivationLayer(ActivationType type)
{
    layers.push_back(std::make_shared<ActivationLayer>(type));
}

void NeuralNetwork::setLossFunction(LossType type)
{
    lossFunction = LossFunction::create(type);
}

Eigen::MatrixXd NeuralNetwork::forward(const Eigen::MatrixXd &input)
{
    Eigen::MatrixXd output = input;
    layerInputs.clear();

    for (const auto &layer : layers)
    {
        layerInputs.push_back(output);
        output = layer->forward(output);
    }

    return output;
}

void NeuralNetwork::backward(const Eigen::MatrixXd &d_output, double learning_rate)
{
    Eigen::MatrixXd d_input = d_output;

    for (int i = layers.size() - 1; i >= 0; --i)
    {
        d_input = layers[i]->backward(d_input, layerInputs[i], learning_rate);
    }
}

void NeuralNetwork::train(const std::vector<Eigen::MatrixXd> &inputs, const std::vector<Eigen::MatrixXd> &labels, int epochs, double learning_rate, int batch_size)
{
    if (!lossFunction)
    {
        throw std::runtime_error("Loss function must be set before training.");
    }

    int num_samples = inputs.size();

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        // Shuffle the data
        std::vector<int> indices(num_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::default_random_engine{});

        for (int i = 0; i < num_samples; i += batch_size)
        {
            int batch_end = std::min(i + batch_size, num_samples);
            std::vector<Eigen::MatrixXd> batch_inputs(inputs.begin() + i, inputs.begin() + batch_end);
            std::vector<Eigen::MatrixXd> batch_labels(labels.begin() + i, labels.begin() + batch_end);

            Eigen::MatrixXd batch_input = Eigen::MatrixXd::Zero(batch_inputs.size(), batch_inputs[0].size());
            Eigen::MatrixXd batch_label = Eigen::MatrixXd::Zero(batch_labels.size(), batch_labels[0].size());

            for (size_t j = 0; j < batch_inputs.size(); ++j)
            {
                batch_input.row(j) = batch_inputs[j];
                batch_label.row(j) = batch_labels[j];
            }

            // Forward pass
            Eigen::MatrixXd predictions = forward(batch_input);

            // Compute loss and backward pass
            Eigen::MatrixXd d_output = lossFunction->derivative(predictions, batch_label);
            backward(d_output, learning_rate);
        }

        std::cout << "Epoch " << epoch + 1 << " complete." << std::endl;
    }
}

void NeuralNetwork::evaluate(const std::vector<Eigen::MatrixXd> &inputs, const std::vector<Eigen::MatrixXd> &labels)
{
    if (!lossFunction)
    {
        throw std::runtime_error("Loss function must be set before evaluation.");
    }

    int num_samples = inputs.size();
    double total_loss = 0.0;
    int correct_predictions = 0;

    for (int i = 0; i < num_samples; ++i)
    {
        Eigen::MatrixXd prediction = forward(inputs[i]);
        total_loss += lossFunction->compute(prediction, labels[i]);

        // Assuming binary classification
        if ((prediction(0, 0) >= 0.5 && labels[i](0, 0) == 1.0) || (prediction(0, 0) < 0.5 && labels[i](0, 0) == 0.0))
        {
            correct_predictions++;
        }
    }

    double average_loss = total_loss / num_samples;
    double accuracy = static_cast<double>(correct_predictions) / num_samples;

    std::cout << "Evaluation results - Loss: " << average_loss << ", Accuracy: " << accuracy << std::endl;
}
