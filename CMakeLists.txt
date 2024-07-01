cmake_minimum_required(VERSION 3.10)
project(CNN_CPP)

# Set C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include Eigen
add_subdirectory(external/eigen)
include_directories(external/eigen)

# Include GoogleTest
add_subdirectory(external/googletest)
enable_testing()

# Include project headers
include_directories(include/Image)
include_directories(include/CNN)
include_directories(include/ThreadPool)

# Build OpenCV as a submodule
include(ExternalProject)
ExternalProject_Add(opencv
    PREFIX ${CMAKE_BINARY_DIR}/opencv
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/opencv
    BINARY_DIR ${CMAKE_BINARY_DIR}/opencv-build
    INSTALL_DIR ${CMAKE_BINARY_DIR}/opencv-install
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/opencv-install
               -DCMAKE_BUILD_TYPE=Release
)

# Set OpenCV include and library directories
ExternalProject_Get_Property(opencv INSTALL_DIR)
include_directories(${INSTALL_DIR}/include/opencv4)
link_directories(${INSTALL_DIR}/lib)

# Source files
file(GLOB_RECURSE IMAGE_SOURCES "src/Image/*.cpp")
file(GLOB_RECURSE CNN_SOURCES "src/CNN/*.cpp")
file(GLOB_RECURSE TP_SOURCES "src/ThreadPool/*.cpp")

# Include main.cpp directly
set(MAIN_SOURCE src/main.cpp)

# Test files
file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")

# Create the executable for the main application
add_executable(CNN_CPP ${IMAGE_SOURCES} ${TP_SOURCES} ${CNN_SOURCES} ${MAIN_SOURCE} ${TEST_SOURCES})

# Link Eigen, GoogleTest, and OpenCV libraries
target_link_libraries(CNN_CPP Eigen3::Eigen gtest gtest_main
    ${INSTALL_DIR}/lib/libopencv_core.so
    ${INSTALL_DIR}/lib/libopencv_highgui.so
    ${INSTALL_DIR}/lib/libopencv_imgproc.so
    ${INSTALL_DIR}/lib/libopencv_imgcodecs.so
)

# Ensure OpenCV is built before building the main project
add_dependencies(CNN_CPP opencv)

# Add tests
add_test(NAME runTests COMMAND CNN_CPP)
