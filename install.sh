#!/bin/bash

# Function to check if a package is installed
is_installed() {
    dpkg -l | grep -q "$1"
}

# Update package list
apt-get update

# Install git if not installed
if ! is_installed git; then
    echo "Installing git..."
    apt-get install -y git
fi

# Install CMake if not installed
if ! is_installed cmake; then
    echo "Installing CMake..."
    apt-get install -y cmake
fi

# Install build-essential if not installed (includes g++ and make)
if ! is_installed build-essential; then
    echo "Installing build-essential..."
    apt-get install -y build-essential
fi

# Install Eigen dependencies
if ! is_installed libeigen3-dev; then
    echo "Installing Eigen dependencies..."
    apt-get install -y libeigen3-dev
fi

# Install OpenCV dependencies
if ! is_installed libopencv-dev; then
    echo "Installing OpenCV dependencies..."
    apt-get install -y libopencv-dev
fi

# Additional OpenCV dependencies
echo "Installing additional OpenCV dependencies..."
apt-get install -y pkg-config
apt-get install -y libjpeg-dev libtiff-dev libpng-dev
apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
apt-get install -y libv4l-dev
apt-get install -y libxvidcore-dev libx264-dev
apt-get install -y libgtk-3-dev
apt-get install -y libatlas-base-dev gfortran

echo "All dependencies are installed."
