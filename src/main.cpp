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
#include "example.hpp"

namespace fs = std::filesystem;

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
            // cnn_sanity_2_colors();
            // cnn_sanity_2_shapes();
            // cnn_sanity_5_colors();
            cnn_sanity_5_shapes();
            // example();
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    }

    return 0;
}
