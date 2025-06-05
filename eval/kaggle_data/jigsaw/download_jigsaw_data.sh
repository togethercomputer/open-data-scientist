#!/bin/bash

# Script to download and extract Jigsaw Toxic Comment Classification Challenge dataset
# This script should be run from the jigsaw directory

set -e  # Exit on any error

echo "Starting download of Jigsaw Toxic Comment Classification Challenge dataset..."

# Get the current directory (should be jigsaw)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script directory: $SCRIPT_DIR"

# Download the dataset using Kaggle CLI
echo "Downloading dataset from Kaggle..."
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge

# Check if download was successful
if [ ! -f "jigsaw-toxic-comment-classification-challenge.zip" ]; then
    echo "Error: Download failed. jigsaw-toxic-comment-classification-challenge.zip not found."
    exit 1
fi

echo "Download completed successfully."

# Extract the zip file
echo "Extracting dataset..."
unzip -o jigsaw-toxic-comment-classification-challenge.zip

# Extract nested zip files if they exist
echo "Checking for nested zip files..."
for zipfile in *.zip; do
    if [ "$zipfile" != "jigsaw-toxic-comment-classification-challenge.zip" ] && [ -f "$zipfile" ]; then
        echo "Extracting $zipfile..."
        unzip -o "$zipfile"
    fi
done

# Clean up zip files
echo "Cleaning up zip files..."
rm -f jigsaw-toxic-comment-classification-challenge.zip
rm -f *.zip

# Remove test_labels.csv if it exists (not needed for submission)
if [ -f "test_labels.csv" ]; then
    echo "Removing test_labels.csv..."
    rm -f test_labels.csv
fi

# Verify extraction
echo "Verifying extracted files..."
if [ -f "train.csv" ] && [ -f "test.csv" ]; then
    echo "✓ Required files extracted successfully:"
    echo "  - train.csv"
    echo "  - test.csv"
    
    # Check for other common files
    if [ -f "sample_submission.csv" ]; then
        echo "  - sample_submission.csv"
    fi
    if [ -f "test_labels.csv" ]; then
        echo "  - test_labels.csv"
    fi
    
    # List all files in current directory
    echo ""
    echo "All files in jigsaw directory:"
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