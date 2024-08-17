#include <gtest/gtest.h>
#include "cnn_cd_e50.hpp"
#include "cnn_cd_elrales_e25.hpp"
#include "cnn_cd_ld_bn_e25.hpp"
#include "cnn_cifar10_bn_e5.hpp"
#include "cnn_cifar10_e5.hpp"
#include "cnn_sanity_2_colors.hpp"
#include "cnn_sanity_2_shapes.hpp"
#include "cnn_sanity_5_colors.hpp"
#include "cnn_sanity_5_shapes.hpp"
#include "cnn_example.hpp"

#include "ImageLoader.hpp"
#include "ImageAugmentor.hpp"
#include "NeuralNetwork.hpp"

namespace fs = std::filesystem;

/**
 * @file main.cpp
 *
 * @brief Main entry point for CNN-CPP Framework.
 *
 * @param argc Number of command line arguments.
 * @param argv Array of command line arguments.
 * @return int Returns 0 upon successful execution, signaling normal program termination.
 */
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    if (argc > 1 && std::string(argv[1]) == "--tests")
    {
        return RUN_ALL_TESTS();
    }
    else
    {
        try
        {
            // cnn_cd_e50();
            // cnn_cd_elrales_e25();
            // cnn_cd_ld_bn_e25();
            // cnn_cifar10_bn_e5();
            // cnn_cifar10_e5();
            cnn_sanity_2_colors();
            cnn_sanity_2_shapes();
            cnn_sanity_5_colors();
            cnn_sanity_5_shapes();
            // cnn_example();
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    }

    return 0;
}
