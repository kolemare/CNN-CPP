#ifndef FLATTENLAYER_HPP
#define FLATTENLAYER_HPP

#include "Layer.hpp"

class FlattenLayer : public Layer
{
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd &d_output, const Eigen::MatrixXd &input, double learning_rate) override;

private:
    int batch_size;
    int original_size;
};

#endif // FLATTENLAYER_HPP
