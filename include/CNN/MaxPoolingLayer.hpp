#ifndef MAXPOOLINGLAYER_HPP
#define MAXPOOLINGLAYER_HPP

#include "Layer.hpp"
#include <Eigen/Dense>

class MaxPoolingLayer : public Layer
{
public:
    MaxPoolingLayer(int pool_size, int stride);

    Eigen::MatrixXd forward(const Eigen::MatrixXd &input_batch) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd &d_output_batch, const Eigen::MatrixXd &input_batch, double learning_rate) override;

private:
    int pool_size;
    int stride;

    Eigen::MatrixXd maxPool(const Eigen::MatrixXd &input);
    Eigen::MatrixXd maxPoolBackward(const Eigen::MatrixXd &d_output, const Eigen::MatrixXd &input);

    std::vector<Eigen::MatrixXd> max_indices; // To store indices of max values during forward pass
};

#endif // MAXPOOLINGLAYER_HPP
