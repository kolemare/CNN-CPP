#!/bin/bash

# Define the URL of the GitHub release asset
DATASET_URL="https://github.com/kolemare/datasets/releases/download/v1.0.0/catsdogs.zip"

# Define the target directory
TARGET_DIR="datasets"

# Create the target directory if it does not exist
mkdir -p $TARGET_DIR

# Check for the --clean flag
if [ "$1" == "--clean" ]; then
    echo "Cleaning up the datasets directory..."
    find $TARGET_DIR -mindepth 1 ! -name '.gitkeep' -delete
    echo "Datasets directory cleaned."
    exit 0
fi

# Download the dataset file using wget
wget -O $TARGET_DIR/catsdogs.zip $DATASET_URL

# Check if the download was successful
if [ $? -ne 0 ]; then
    echo "Failed to download the dataset."
    exit 1
fi

# Unzip the dataset file
unzip $TARGET_DIR/catsdogs.zip -d $TARGET_DIR

# Check if the unzip was successful
if [ $? -ne 0 ]; then
    echo "Failed to unzip the dataset file."
    exit 1
fi

# Clean up the zip file
rm $TARGET_DIR/catsdogs.zip

echo "Dataset downloaded and extracted to $TARGET_DIR"
