#!/bin/sh
# https://github.com/chaitanya100100/TailorNet#how-to-run
# https://github.com/zycliao/TailorNet_dataset#data-preparation
set -eu
mkdir -p datasets
mkdir -p checkpoints
wget -c https://datasets.d2.mpi-inf.mpg.de/tailornet/dataset_meta.zip
wget -c https://datasets.d2.mpi-inf.mpg.de/tailornet/old-t-shirt_female.zip
wget -c https://datasets.d2.mpi-inf.mpg.de/tailornet/old-t-shirt_female_weights.zip
mv dataset_meta.zip datasets/
mv old-t-shirt_female.zip datasets/
mv old-t-shirt_female_weights.zip checkpoints/

cd datasets/
unzip dataset_meta.zip
unzip old-t-shirt_female.zip
rm -rf dataset_meta.zip
rm -rf old-t-shirt_female.zip

cd ..
cd checkpoints
unzip old-t-shirt_female_weights.zip
rm -rf old-t-shirt_female_weights.zip
