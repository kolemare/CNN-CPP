#!/bin/bash

# Update package list
echo "Updating package list..."
apt-get update -y

# Install necessary packages
echo "Installing required packages..."
apt-get install -y \
    git \
    cmake \
    build-essential \
    libeigen3-dev \
    libopencv-dev \
    pkg-config \
    libjpeg-dev \
    libtiff-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    wget \
    unzip \
    python3 \
    python3-pip \
    doxygen \
    texlive-full

# Upgrade pip and install Python packages
echo "Upgrading pip and installing Python packages..."
pip3 install --upgrade pip
pip3 install requests matplotlib invoke

# Clean up unnecessary packages and cache
echo "Cleaning up unnecessary packages and cache..."
apt-get clean
apt-get autoremove -y
rm -rf /var/lib/apt/lists/*

echo "All dependencies are installed and system cleaned up."
