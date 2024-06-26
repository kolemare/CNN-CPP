#include "MaxPoolingLayer.hpp"
#include <algorithm>
#include <stdexcept>

MaxPoolingLayer::MaxPoolingLayer(int pool_size, int stride)
    : pool_size(pool_size), stride(stride) {}

Eigen::MatrixXd MaxPoolingLayer::forward(const Eigen::MatrixXd &input_batch)
{
    int batch_size = input_batch.rows();
    int input_size = std::sqrt(input_batch.cols());
    if (input_size * input_size != input_batch.cols())
    {
        throw std::invalid_argument("Input dimensions do not match the expected size.");
    }

    int output_size = (input_size - pool_size) / stride + 1;
    Eigen::MatrixXd output_batch(batch_size, output_size * output_size);

    max_indices.clear();

    for (int b = 0; b < batch_size; ++b)
    {
        Eigen::Map<const Eigen::MatrixXd> input(input_batch.row(b).data(), input_size, input_size);
        Eigen::MatrixXd pooled_output = maxPool(input);
        output_batch.row(b) = Eigen::Map<Eigen::RowVectorXd>(pooled_output.data(), pooled_output.size());
    }

    return output_batch;
}

Eigen::MatrixXd MaxPoolingLayer::backward(const Eigen::MatrixXd &d_output_batch, const Eigen::MatrixXd &input_batch, double learning_rate)
{
    int batch_size = input_batch.rows();
    int input_size = std::sqrt(input_batch.cols());
    int output_size = (input_size - pool_size) / stride + 1;

    Eigen::MatrixXd d_input_batch = Eigen::MatrixXd::Zero(batch_size, input_size * input_size);

    for (int b = 0; b < batch_size; ++b)
    {
        Eigen::Map<const Eigen::MatrixXd> input(input_batch.row(b).data(), input_size, input_size);
        Eigen::Map<const Eigen::MatrixXd> d_output(d_output_batch.row(b).data(), output_size, output_size);

        Eigen::MatrixXd d_input = maxPoolBackward(d_output, input);
        d_input_batch.row(b) = Eigen::Map<Eigen::RowVectorXd>(d_input.data(), d_input.size());
    }

    return d_input_batch;
}

Eigen::MatrixXd MaxPoolingLayer::maxPool(const Eigen::MatrixXd &input)
{
    int input_size = input.rows();
    int output_size = (input_size - pool_size) / stride + 1;

    Eigen::MatrixXd output = Eigen::MatrixXd::Zero(output_size, output_size);
    Eigen::MatrixXd indices = Eigen::MatrixXd::Zero(output_size, output_size);

    for (int i = 0; i < output_size; ++i)
    {
        for (int j = 0; j < output_size; ++j)
        {
            int row_start = i * stride;
            int col_start = j * stride;
            Eigen::MatrixXd sub_matrix = input.block(row_start, col_start, pool_size, pool_size);
            double max_val = sub_matrix.maxCoeff();
            output(i, j) = max_val;

            // Store index of max value
            for (int r = 0; r < pool_size; ++r)
            {
                for (int c = 0; c < pool_size; ++c)
                {
                    if (sub_matrix(r, c) == max_val)
                    {
                        indices(i, j) = row_start * input_size + col_start + r * input_size + c;
                        break;
                    }
                }
            }
        }
    }

    max_indices.push_back(indices);
    return output;
}

Eigen::MatrixXd MaxPoolingLayer::maxPoolBackward(const Eigen::MatrixXd &d_output, const Eigen::MatrixXd &input)
{
    int input_size = input.rows();
    int output_size = d_output.rows();
    Eigen::MatrixXd d_input = Eigen::MatrixXd::Zero(input_size, input_size);

    Eigen::MatrixXd indices = max_indices.back();
    max_indices.pop_back();

    for (int i = 0; i < output_size; ++i)
    {
        for (int j = 0; j < output_size; ++j)
        {
            int index = indices(i, j);
            int row = index / input_size;
            int col = index % input_size;
            d_input(row, col) += d_output(i, j);
        }
    }

    return d_input;
}
