#!/usr/bin/env bash
set -e

echo "Creating Python virtual environment..."

python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing required packages..."
pip install requests matplotlib invoke tensorflow keras numpy scipy

echo "Virtual environment ready."
echo "Activate it with: source venv/bin/activate"