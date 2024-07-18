#ifndef MAX_POOLING_LAYER_HPP
#define MAX_POOLING_LAYER_HPP

#include <Eigen/Dense>
#include <vector>
#include "Layer.hpp"

class MaxPoolingLayer : public Layer
{
public:
    MaxPoolingLayer(int pool_size, int stride);

    Eigen::MatrixXd forward(const Eigen::MatrixXd &input_batch);
    Eigen::MatrixXd backward(const Eigen::MatrixXd &d_output_batch, const Eigen::MatrixXd &input_batch, double learning_rate);
    bool needsOptimizer() const override;
    void setOptimizer(std::unique_ptr<Optimizer> optimizer) override;

    // Static variable getters and setters
    static void setInputSize(int size);
    static void setInputDepth(int depth);
    static int getInputSize();
    static int getInputDepth();
    int getPoolSize();
    int getStride();

private:
    int pool_size;
    int stride;

    static int input_size;
    static int input_depth;

    std::vector<Eigen::MatrixXd> max_indices;

    Eigen::MatrixXd maxPool(const Eigen::MatrixXd &input);
    Eigen::MatrixXd maxPoolBackward(const Eigen::MatrixXd &d_output, const Eigen::MatrixXd &input);

    int memorized_input_size;
    int memorized_input_depth;
    int memorized_batch_size;
};

#endif
