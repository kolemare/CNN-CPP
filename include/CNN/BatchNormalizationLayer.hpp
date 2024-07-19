// #ifndef BATCHNORMALIZATIONLAYER_HPP
// #define BATCHNORMALIZATIONLAYER_HPP

// #include "Layer.hpp"
// #include <Eigen/Dense>

// class BatchNormalizationLayer : public Layer
// {
// public:
//     BatchNormalizationLayer(double epsilon = 1e-5, double momentum = 0.9);

//     Eigen::MatrixXd forward(const Eigen::MatrixXd &input_batch) override;
//     Eigen::MatrixXd backward(const Eigen::MatrixXd &d_output_batch, const Eigen::MatrixXd &input_batch, double learning_rate) override;
//     bool needsOptimizer() const override;
//     void setOptimizer(std::unique_ptr<Optimizer> optimizer) override;

// private:
//     void initialize(int input_dim);
//     void updateParameters(double learning_rate);

//     double epsilon;
//     double momentum;
//     bool initialized;
//     Eigen::VectorXd gamma, beta, moving_mean, moving_variance;
//     Eigen::VectorXd dgamma, dbeta;
//     Eigen::MatrixXd cache_normalized;
//     Eigen::VectorXd cache_mean, cache_variance;
// };

// #endif // BATCHNORMALIZATIONLAYER_HPP
