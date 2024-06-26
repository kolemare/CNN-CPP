#ifndef AVERAGEPOOLINGLAYER_HPP
#define AVERAGEPOOLINGLAYER_HPP

#include "Layer.hpp"
#include <Eigen/Dense>

class AveragePoolingLayer : public Layer
{
public:
    AveragePoolingLayer(int pool_size, int stride);

    Eigen::MatrixXd forward(const Eigen::MatrixXd &input_batch) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd &d_output_batch, const Eigen::MatrixXd &input_batch, double learning_rate) override;

private:
    int pool_size;
    int stride;

    Eigen::MatrixXd pool(const Eigen::MatrixXd &input);
};

#endif // AVERAGEPOOLINGLAYER_HPP
