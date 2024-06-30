#!/bin/bash

clean_build() {
  echo "Cleaning build directory and submodules..."
  rm -rf .vscode
  rm -rf build
  rm -rf external/opencv/build
  rm -rf external/googletest/build
  git submodule foreach --recursive git clean -fdx
  git submodule foreach --recursive git reset --hard
  echo "Build clean completed."
}

clean_dataset() {
  echo "Cleaning dataset directory..."
  find datasets -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} +
  echo "Dataset clean completed."
}

delete_txts() {
  echo "Deleting all .txt files..."
  find . -maxdepth 1 -type f -name '*.txt' ! -name 'CMakeLists.txt' -delete
  echo "TXT files deletion completed."
}

# Flags
CLEAN_BUILD=false
CLEAN_DATASET=false
CLEAN_ALL=false

# Parse arguments
for arg in "$@"; do
  case $arg in
    --build)
      CLEAN_BUILD=true
      ;;
    --dataset)
      CLEAN_DATASET=true
      ;;
    --all)
      CLEAN_ALL=true
      ;;
    *)
      echo "Unknown argument: $arg"
      ;;
  esac
done

# Execute based on flags
if $CLEAN_ALL; then
  clean_build
  clean_dataset
else
  if $CLEAN_BUILD; then
    clean_build
  fi
  if $CLEAN_DATASET; then
    clean_dataset
  fi
fi

# Always delete .txt files
delete_txts
