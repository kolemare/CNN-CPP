// #include <gtest/gtest.h>
// #include "Optimizer.hpp"
// #include <Eigen/Dense>
// #include <unordered_map>

// class OptimizerTest : public ::testing::Test
// {
// protected:
//     Eigen::MatrixXd weights;
//     Eigen::VectorXd biases;
//     Eigen::MatrixXd d_weights;
//     Eigen::VectorXd d_biases;
//     double learning_rate;

//     virtual void SetUp()
//     {
//         // Initialize weights and biases
//         weights.resize(2, 3);
//         weights << 1, 2, 3,
//             4, 5, 6;

//         biases.resize(2);
//         biases << 1, 2;

//         // Initialize gradients
//         d_weights.resize(2, 3);
//         d_weights << 0.1, 0.2, 0.3,
//             0.4, 0.5, 0.6;

//         d_biases.resize(2);
//         d_biases << 0.1, 0.2;

//         // Set learning rate
//         learning_rate = 0.01;
//     }
// };

// TEST_F(OptimizerTest, SGDUpdate)
// {
//     auto optimizer = Optimizer::create(Optimizer::Type::SGD);

//     optimizer->update(weights, biases, d_weights, d_biases, learning_rate);

//     Eigen::MatrixXd expected_weights(2, 3);
//     expected_weights << 0.999, 1.998, 2.997,
//         3.996, 4.995, 5.994;

//     Eigen::VectorXd expected_biases(2);
//     expected_biases << 0.999, 1.998;

//     ASSERT_TRUE(weights.isApprox(expected_weights, 1e-5)) << "Weights:\n"
//                                                           << weights << "\nExpected:\n"
//                                                           << expected_weights;

//     ASSERT_TRUE(biases.isApprox(expected_biases, 1e-5)) << "Biases:\n"
//                                                         << biases << "\nExpected:\n"
//                                                         << expected_biases;
// }

// TEST_F(OptimizerTest, SGDWithMomentumUpdate)
// {
//     auto optimizer = Optimizer::create(Optimizer::Type::SGDWithMomentum, {{"momentum", 0.9}});

//     optimizer->update(weights, biases, d_weights, d_biases, learning_rate);

//     Eigen::MatrixXd expected_weights(2, 3);
//     expected_weights << 0.999, 1.998, 2.997,
//         3.996, 4.995, 5.994;

//     Eigen::VectorXd expected_biases(2);
//     expected_biases << 0.999, 1.998;

//     ASSERT_TRUE(weights.isApprox(expected_weights, 1e-5)) << "Weights:\n"
//                                                           << weights << "\nExpected:\n"
//                                                           << expected_weights;

//     ASSERT_TRUE(biases.isApprox(expected_biases, 1e-5)) << "Biases:\n"
//                                                         << biases << "\nExpected:\n"
//                                                         << expected_biases;

//     // Update again to check momentum
//     optimizer->update(weights, biases, d_weights, d_biases, learning_rate);

//     expected_weights << 0.9971, 1.9942, 2.9913,
//         3.9884, 4.9855, 5.9826;

//     expected_biases << 0.9971, 1.9942;

//     ASSERT_TRUE(weights.isApprox(expected_weights, 1e-5)) << "Weights:\n"
//                                                           << weights << "\nExpected:\n"
//                                                           << expected_weights;

//     ASSERT_TRUE(biases.isApprox(expected_biases, 1e-5)) << "Biases:\n"
//                                                         << biases << "\nExpected:\n"
//                                                         << expected_biases;
// }

// TEST_F(OptimizerTest, AdamUpdate)
// {
//     auto optimizer = Optimizer::create(Optimizer::Type::Adam, {{"beta1", 0.9}, {"beta2", 0.999}, {"epsilon", 1e-8}});

//     optimizer->update(weights, biases, d_weights, d_biases, learning_rate);

//     Eigen::MatrixXd expected_weights(2, 3);
//     expected_weights << 0.990000002, 1.990000000, 2.9900000003,
//         3.9900000002, 4.9900000002, 5.99000000016;

//     Eigen::VectorXd expected_biases(2);
//     expected_biases << 0.9900000010, 1.9900000005;

//     ASSERT_TRUE(weights.isApprox(expected_weights, 1e-5)) << "Weights:\n"
//                                                           << weights << "\nExpected:\n"
//                                                           << expected_weights;

//     ASSERT_TRUE(biases.isApprox(expected_biases, 1e-5)) << "Biases:\n"
//                                                         << biases << "\nExpected:\n"
//                                                         << expected_biases;

//     // Update again to check Adam optimizer state update
//     optimizer->update(weights, biases, d_weights, d_biases, learning_rate);

//     expected_weights << 0.980000002, 1.980000001, 2.9800000006,
//         3.9800000005, 4.9800000004, 5.9800000003;

//     expected_biases << 0.980000002, 1.980000002;

//     ASSERT_TRUE(weights.isApprox(expected_weights, 1e-5)) << "Weights:\n"
//                                                           << weights << "\nExpected:\n"
//                                                           << expected_weights;

//     ASSERT_TRUE(biases.isApprox(expected_biases, 1e-5)) << "Biases:\n"
//                                                         << biases << "\nExpected:\n"
//                                                         << expected_biases;
// }

// TEST_F(OptimizerTest, RMSpropUpdate)
// {
//     auto optimizer = Optimizer::create(Optimizer::Type::RMSprop, {{"beta", 0.9}, {"epsilon", 1e-8}});

//     optimizer->update(weights, biases, d_weights, d_biases, learning_rate);

//     Eigen::MatrixXd expected_weights(2, 3);
//     expected_weights << 0.9683734645, 1.96837725665, 2.9683745775,
//         3.96837734573, 4.9683772568569, 5.968377256855;

//     Eigen::VectorXd expected_biases(2);
//     expected_biases << 0.968377547453, 1.9683774574345;

//     ASSERT_TRUE(weights.isApprox(expected_weights, 1e-5)) << "Weights:\n"
//                                                           << weights << "\nExpected:\n"
//                                                           << expected_weights;

//     ASSERT_TRUE(biases.isApprox(expected_biases, 1e-5)) << "Biases:\n"
//                                                         << biases << "\nExpected:\n"
//                                                         << expected_biases;

//     // Update again to check RMSprop state update
//     optimizer->update(weights, biases, d_weights, d_biases, learning_rate);

//     expected_weights << 0.945435568545, 1.94543534634, 2.94543534564643,
//         3.94543346343, 4.945435346347, 5.945435457458;

//     expected_biases << 0.945435436, 1.945435457458;

//     ASSERT_TRUE(weights.isApprox(expected_weights, 1e-5)) << "Weights:\n"
//                                                           << weights << "\nExpected:\n"
//                                                           << expected_weights;

//     ASSERT_TRUE(biases.isApprox(expected_biases, 1e-5)) << "Biases:\n"
//                                                         << biases << "\nExpected:\n"
//                                                         << expected_biases;
// }
