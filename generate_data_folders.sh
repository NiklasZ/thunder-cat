#!/bin/bash

# Check if the 'data' directory already exists
if [ -d "data" ]; then
  echo "Directory 'data' already exists. Exiting script."
  exit 1
fi

# Create the folder structure
mkdir -p data/analysis
mkdir -p data/log
mkdir -p data/sound
mkdir -p data/video/evaluation/cat/day_cat
mkdir -p data/video/evaluation/cat/night_cat
mkdir -p data/video/evaluation/other

echo "Folder structure created successfully."
