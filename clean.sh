#!/bin/bash

clean_build() {
  echo "Cleaning build directory and submodules..."
  rm -rf build
  rm -rf external/opencv/build
  rm -rf external/googletest/build
  git submodule foreach --recursive git clean -fdx
  git submodule foreach --recursive git reset --hard
  echo "Build clean completed."
}

clean_datasets() {
  echo "Cleaning datasets directory..."
  if [ -d "datasets" ]; then
    # Use -prune to avoid descending into non-existent directories
    find datasets -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} + 2>/dev/null
    echo "Datasets clean completed."
  else
    echo "Datasets directory does not exist. Skipping."
  fi
}

delete_txts() {
  rm -rf .vscode
  echo "Deleting all .txt files..."
  find . -maxdepth 1 -type f -name '*.txt' ! -name 'CMakeLists.txt' -delete
  echo "TXT files deletion completed."
}

# Flags
CLEAN_BUILD=false
CLEAN_DATASETS=false
CLEAN_ALL=false

# Parse arguments
for arg in "$@"; do
  case $arg in
    --build)
      CLEAN_BUILD=true
      ;;
    --datasets)
      CLEAN_DATASETS=true
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
  clean_datasets
else
  if $CLEAN_BUILD; then
    clean_build
  fi
  if $CLEAN_DATASETS; then
    clean_datasets
  fi
fi

# Always delete .txt files
delete_txts
