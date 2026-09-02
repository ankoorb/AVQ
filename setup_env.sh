#!/bin/bash

echo "Installing required Python packages..."
pip install matplotlib datasets
pip install -q "torchmetrics[image]"

cd /workspace

echo "Downloading Tiny ImageNet..."
wget -c -q http://cs231n.stanford.edu/tiny-imagenet-200.zip

echo "Extracting Tiny ImageNet files directly to /tmp/data/ ..."
mkdir -p /tmp/data/
unzip -q tiny-imagenet-200.zip -d /tmp/data/

echo "Starting COCO 2017 downloads (this will take a while - ~19GB total)..."
wget -c -q --show-progress http://images.cocodataset.org/zips/train2017.zip
wget -c -q --show-progress http://images.cocodataset.org/zips/val2017.zip

echo "Downloads finished! Extracting COCO 2017 to /tmp/data/coco2017 ..."
mkdir -p /tmp/data/coco2017
unzip -q train2017.zip -d /tmp/data/coco2017/
unzip -q val2017.zip -d /tmp/data/coco2017/

echo "Starting MiniPlaces download from MIT..."
wget -c -q http://6.869.csail.mit.edu/fa17/miniplaces/data.tar.gz

echo "MiniPlaces Download finished! Extracting files to /tmp/data/ ..."
tar -xzf data.tar.gz -C /tmp/data/

echo "Extraction complete! Cloning MiniPlaces devkit to /tmp/data/ ..."
cd /tmp/data/
git clone https://github.com/CSAILVision/miniplaces.git devkit

echo "Cleaning up..."
cd /workspace
rm tiny-imagenet-200.zip
rm data.tar.gz
rm train2017.zip val2017.zip

echo "Setup complete! Your datasets are ready in /tmp/data/"
