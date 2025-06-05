#!/bin/bash

# Script to download and extract Spooky Author Identification dataset
# This script should be run from the spooky directory

set -e  # Exit on any error

echo "Starting download of Spooky Author Identification dataset..."

# Get the current directory (should be spooky)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script directory: $SCRIPT_DIR"

# Download the dataset using Kaggle CLI
echo "Downloading dataset from Kaggle..."
kaggle competitions download -c spooky-author-identification

# Check if download was successful
if [ ! -f "spooky-author-identification.zip" ]; then
    echo "Error: Download failed. spooky-author-identification.zip not found."
    exit 1
fi

echo "Download completed successfully."

# Extract the zip file
echo "Extracting dataset..."
unzip -o spooky-author-identification.zip

# Extract nested zip files
echo "Extracting nested zip files..."
if [ -f "train.zip" ]; then
    echo "Extracting train.zip..."
    unzip -o train.zip
fi

if [ -f "test.zip" ]; then
    echo "Extracting test.zip..."
    unzip -o test.zip
fi

if [ -f "sample_submission.zip" ]; then
    echo "Extracting sample_submission.zip..."
    unzip -o sample_submission.zip
fi

# Clean up zip files
echo "Cleaning up zip files..."
rm -f spooky-author-identification.zip
rm -f train.zip
rm -f test.zip
rm -f sample_submission.zip

# Verify extraction
echo "Verifying extracted files..."
if [ -f "train.csv" ] && [ -f "test.csv" ] && [ -f "sample_submission.csv" ]; then
    echo "✓ All required files extracted successfully:"
    echo "  - train.csv"
    echo "  - test.csv" 
    echo "  - sample_submission.csv"
    
    # List all files in current directory
    echo ""
    echo "All files in spooky directory:"
    ls -la
    
else
    echo "Error: Some required files are missing after extraction."
    echo "Files found:"
    ls -la
    exit 1
fi

echo ""
echo "Dataset download and extraction completed successfully!"
echo "All files are ready in: $SCRIPT_DIR" 