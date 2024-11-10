#!/bin/bash

# Check if the 'data' directory already exists
if [ -d "data" ]; then
  echo "Directory 'data' already exists. Exiting script."
  exit 1
fi

# Create the folder structure
mkdir -p data/sound
mkdir -p data/video/evaluation/cat
mkdir -p data/video/evaluation/not_cat
mkdir -p data/log

echo "Folder structure created successfully."
