// #include "BatchNormalizationLayer.hpp"
// #include <iostream>

// BatchNormalizationLayer::BatchNormalizationLayer(double epsilon, double momentum)
//     : epsilon(epsilon), momentum(momentum), initialized(false) {}

// bool BatchNormalizationLayer::needsOptimizer() const
// {
//     return false;
// }

// void BatchNormalizationLayer::setOptimizer(std::unique_ptr<Optimizer> optimizer)
// {
//     return;
// }

// void BatchNormalizationLayer::initialize(int input_dim)
// {
//     gamma = Eigen::VectorXd::Ones(input_dim);
//     beta = Eigen::VectorXd::Zero(input_dim);
//     moving_mean = Eigen::VectorXd::Zero(input_dim);
//     moving_variance = Eigen::VectorXd::Ones(input_dim);
//     dgamma = Eigen::VectorXd::Zero(input_dim);
//     dbeta = Eigen::VectorXd::Zero(input_dim);
//     initialized = true;
// }

// Eigen::MatrixXd BatchNormalizationLayer::forward(const Eigen::MatrixXd &input)
// {
//     if (!initialized)
//     {
//         initialize(input.cols());
//     }

//     cache_mean = input.colwise().mean();
//     Eigen::MatrixXd input_centered = input.rowwise() - cache_mean.transpose();
//     cache_variance = input_centered.array().square().colwise().mean();

//     Eigen::VectorXd sqrt_variance = (cache_variance.array() + epsilon).sqrt();
//     Eigen::VectorXd inv_sqrt_variance = 1.0 / sqrt_variance.array();
//     cache_normalized = input_centered.array().rowwise() * inv_sqrt_variance.transpose().array();

//     moving_mean = momentum * moving_mean + (1 - momentum) * cache_mean;
//     moving_variance = momentum * moving_variance + (1 - momentum) * cache_variance;

//     Eigen::MatrixXd scaled = cache_normalized.array().rowwise() * gamma.transpose().array();
//     Eigen::MatrixXd shifted = scaled.rowwise() + beta.transpose();

//     return shifted;
// }

// Eigen::MatrixXd BatchNormalizationLayer::backward(const Eigen::MatrixXd &d_output, const Eigen::MatrixXd &input, double learning_rate)
// {
//     int batch_size = input.rows();
//     int input_dim = input.cols();

//     // Compute gradients for gamma and beta
//     Eigen::MatrixXd d_normalized = d_output.array().rowwise() * gamma.transpose().array();
//     dgamma = (d_output.array() * cache_normalized.array()).colwise().sum();
//     dbeta = d_output.colwise().sum();

//     // Compute inverse square root of variance
//     Eigen::ArrayXd inv_sqrt_variance = (cache_variance.array() + epsilon).pow(-0.5);
//     Eigen::MatrixXd input_centered = input.rowwise() - cache_mean.transpose();

//     // Compute gradient with respect to variance
//     Eigen::MatrixXd d_variance_part = d_normalized.array() * input_centered.array();
//     Eigen::ArrayXd d_variance_sum = d_variance_part.colwise().sum();
//     Eigen::ArrayXd d_variance = d_variance_sum * -0.5 * inv_sqrt_variance.pow(3);

//     // Compute gradient with respect to mean
//     Eigen::MatrixXd d_mean_part1 = d_normalized.array().rowwise() * inv_sqrt_variance.transpose().array();
//     Eigen::MatrixXd d_mean_part1_sum = d_mean_part1.colwise().sum();

//     Eigen::MatrixXd d_variance_replicated = d_variance.transpose().replicate(batch_size, 1);
//     Eigen::MatrixXd d_mean_part2 = d_variance_replicated.array() * input_centered.array() * 2.0 / batch_size;
//     Eigen::MatrixXd d_mean_part2_sum = d_mean_part2.colwise().sum();

//     Eigen::MatrixXd d_mean = d_mean_part1_sum + d_mean_part2_sum * -1.0 / batch_size;

//     // Compute final gradient with respect to input
//     Eigen::MatrixXd d_input_part1 = d_normalized.array().rowwise() * inv_sqrt_variance.transpose().array();
//     Eigen::MatrixXd d_input_part2 = d_variance_replicated.array() * input_centered.array() * 2.0 / batch_size;
//     Eigen::MatrixXd d_input = d_input_part1 + d_input_part2;

//     // Correctly subtract the mean gradient
//     Eigen::MatrixXd d_mean_replicated = d_mean.replicate(batch_size, 1);
//     d_input -= d_mean_replicated;

//     // Update gamma and beta parameters
//     updateParameters(learning_rate);

//     return d_input;
// }

// void BatchNormalizationLayer::updateParameters(double learning_rate)
// {
//     gamma -= learning_rate * dgamma;
//     beta -= learning_rate * dbeta;
// }
