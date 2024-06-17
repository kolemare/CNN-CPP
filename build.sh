#!/bin/bash

# Function to clean the build
clean_build() {
    echo "Cleaning build directory and submodules..."
    rm -rf build
    rm -rf external/opencv/build
    rm -rf external/googletest/build
    git submodule foreach --recursive git clean -fdx
    git submodule foreach --recursive git reset --hard
}

# Function to build the application and tests
build() {
    mkdir -p build
    cd build
    cmake -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF ..
    make -j${JOBS}
    echo "Build completed."
}

# Default number of jobs
JOBS=1
CLEAN=false

# Parse command line arguments
for arg in "$@"
do
    case $arg in
        --clean)
        CLEAN=true
        ;;
        -j*)
        JOBS="${arg#-j}"
        ;;
        *)
        ;;
    esac
done

if $CLEAN; then
    clean_build
fi

build
