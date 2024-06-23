#!/bin/bash

echo "Cleaning build directory and submodules..."
rm -rf .vscode
rm -rf build
rm -rf external/opencv/build
rm -rf external/googletest/build
git submodule foreach --recursive git clean -fdx
git submodule foreach --recursive git reset --hard

echo "Clean completed."
