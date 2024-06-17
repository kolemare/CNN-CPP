cmake_minimum_required(VERSION 3.10)
project(CNN_CPP)

# Set C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include Eigen
add_subdirectory(external/eigen)
include_directories(external/eigen)

# Include OpenCV
add_subdirectory(external/opencv)
include_directories(external/opencv/include)

# Include GoogleTest
add_subdirectory(external/googletest)
enable_testing()

# Include project headers
include_directories(include)

# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Test files
file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")

# Create the executable for the main application
add_executable(CNN_CPP ${SOURCES} ${TEST_SOURCES})

# Link libraries
target_link_libraries(CNN_CPP Eigen3::Eigen ${OpenCV_LIBS} gtest gtest_main)

# Find OpenCV
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(CNN_CPP ${OpenCV_LIBS})
endif()

# Add tests
add_test(NAME runTests COMMAND CNN_CPP)
