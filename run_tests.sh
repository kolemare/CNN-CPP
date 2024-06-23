#!/bin/bash

# Check if the build directory exists
if [ ! -d "build" ]; then
    echo "Build directory does not exist. Please run the build script first."
    exit 1
fi

# Navigate to the build directory
cd build

# Run the tests using the built executable
if [ -f "./CNN_CPP" ]; then
    echo "Running tests..."
    ./CNN_CPP --tests
else
    echo "Test executable not found. Please run the build script first."
    exit 1
fi

# Return to the root directory
cd ..

echo "Tests run completed."
