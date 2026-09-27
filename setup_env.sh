#!/bin/bash

set -e

echo "Installing required Python packages..."
pip install matplotlib pycocotools
pip install -q "torchmetrics[image]"
pip install iterative-stratification
pip install diveq==0.1.3

cd /workspace

# --------------------------------------------------
# Tiny ImageNet
# --------------------------------------------------

echo "Downloading Tiny ImageNet..."
wget -c -q \
    http://cs231n.stanford.edu/tiny-imagenet-200.zip

echo "Extracting Tiny ImageNet to /tmp/data/ ..."
mkdir -p /tmp/data
unzip -q tiny-imagenet-200.zip -d /tmp/data/


# --------------------------------------------------
# MiniPlaces
# --------------------------------------------------

echo "Downloading MiniPlaces..."
wget -c -q \
    http://6.869.csail.mit.edu/fa17/miniplaces/data.tar.gz

echo "Extracting MiniPlaces to /tmp/data/miniplaces/ ..."
mkdir -p /tmp/data/miniplaces
tar -xzf data.tar.gz -C /tmp/data/miniplaces/

echo "Downloading MiniPlaces metadata..."
mkdir -p /tmp/data/miniplaces/devkit/data

BASE_URL="https://raw.githubusercontent.com/CSAILVision/miniplaces/master/data"

wget -q \
    "$BASE_URL/train.txt" \
    -O /tmp/data/miniplaces/devkit/data/train.txt

wget -q \
    "$BASE_URL/val.txt" \
    -O /tmp/data/miniplaces/devkit/data/val.txt

wget -q \
    "$BASE_URL/categories.txt" \
    -O /tmp/data/miniplaces/devkit/data/categories.txt


# --------------------------------------------------
# COCO 2017
# --------------------------------------------------

echo "Downloading COCO 2017..."
wget -c -q --show-progress http://images.cocodataset.org/zips/train2017.zip
wget -c -q --show-progress http://images.cocodataset.org/zips/val2017.zip
wget -c -q --show-progress http://images.cocodataset.org/annotations/annotations_trainval2017.zip

echo "Extracting COCO 2017..."
mkdir -p /tmp/data/coco2017
unzip -q train2017.zip -d /tmp/data/coco2017/
unzip -q val2017.zip -d /tmp/data/coco2017/
unzip -q annotations_trainval2017.zip -d /tmp/data/coco2017/


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

echo "Cleaning up..."
rm -f /workspace/tiny-imagenet-200.zip
rm -f /workspace/data.tar.gz
rm -f /workspace/train2017.zip /workspace/val2017.zip /workspace/annotations_trainval2017.zip

echo "Setup complete!"
echo "Tiny ImageNet: /tmp/data/tiny-imagenet-200"
echo "MiniPlaces:    /tmp/data/miniplaces"
echo "COCO2017:    /tmp/data/coco2017"
