#include "AveragePoolingLayer.hpp"
#include <cmath>

AveragePoolingLayer::AveragePoolingLayer(int pool_size, int stride)
    : pool_size(pool_size), stride(stride) {}

Eigen::MatrixXd AveragePoolingLayer::forward(const Eigen::MatrixXd &input_batch)
{
    int batch_size = input_batch.rows();
    int input_size = std::sqrt(input_batch.cols());
    int output_size = (input_size - pool_size) / stride + 1;

    Eigen::MatrixXd output_batch(batch_size, output_size * output_size);

    for (int b = 0; b < batch_size; ++b)
    {
        Eigen::Map<const Eigen::MatrixXd> input(input_batch.row(b).data(), input_size, input_size);
        Eigen::MatrixXd output = pool(input);
        output_batch.row(b) = Eigen::Map<Eigen::RowVectorXd>(output.data(), output.size());
    }

    return output_batch;
}

Eigen::MatrixXd AveragePoolingLayer::backward(const Eigen::MatrixXd &d_output_batch, const Eigen::MatrixXd &input_batch, double learning_rate)
{
    int batch_size = input_batch.rows();
    int input_size = std::sqrt(input_batch.cols());
    int output_size = (input_size - pool_size) / stride + 1;

    Eigen::MatrixXd d_input_batch = Eigen::MatrixXd::Zero(batch_size, input_size * input_size);

    for (int b = 0; b < batch_size; ++b)
    {
        Eigen::Map<const Eigen::MatrixXd> input(input_batch.row(b).data(), input_size, input_size);
        Eigen::Map<const Eigen::MatrixXd> d_output(d_output_batch.row(b).data(), output_size, output_size);
        Eigen::MatrixXd d_input = Eigen::MatrixXd::Zero(input_size, input_size);

        for (int i = 0; i < output_size; ++i)
        {
            for (int j = 0; j < output_size; ++j)
            {
                int row_start = i * stride;
                int col_start = j * stride;

                d_input.block(row_start, col_start, pool_size, pool_size).array() += d_output(i, j) / (pool_size * pool_size);
            }
        }

        d_input_batch.row(b) = Eigen::Map<Eigen::RowVectorXd>(d_input.data(), d_input.size());
    }

    return d_input_batch;
}

Eigen::MatrixXd AveragePoolingLayer::pool(const Eigen::MatrixXd &input)
{
    int input_size = input.rows();
    int output_size = (input_size - pool_size) / stride + 1;
    Eigen::MatrixXd output = Eigen::MatrixXd::Zero(output_size, output_size);

    for (int i = 0; i < output_size; ++i)
    {
        for (int j = 0; j < output_size; ++j)
        {
            int row_start = i * stride;
            int col_start = j * stride;
            output(i, j) = input.block(row_start, col_start, pool_size, pool_size).mean();
        }
    }

    return output;
}
