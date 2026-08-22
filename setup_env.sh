#!/bin/bash

echo "Installing required Python packages..."
pip install matplotlib datasets
pip install -q "torchmetrics[image]"

echo "Navigating to persistent workspace..."
cd /workspace

echo "Downloading Tiny ImageNet..."
wget -c http://cs231n.stanford.edu/tiny-imagenet-200.zip

echo "Extracting directly to fast local NVMe drive (/tmp/data/)..."
mkdir -p /tmp/data/
unzip -q tiny-imagenet-200.zip -d /tmp/data/

echo "Cleaning up the zip file..."
rm tiny-imagenet-200.zip

echo "Setup complete! Your dataset is ready in /tmp/data/tiny-imagenet-200"
