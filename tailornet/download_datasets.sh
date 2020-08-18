#!/bin/sh
# https://github.com/chaitanya100100/TailorNet#how-to-run
# https://github.com/zycliao/TailorNet_dataset#data-preparation
set -eu
ROOT_DIR=${PWD}

# dataset
cd ${ROOT_DIR}
mkdir -p datasets
mkdir -p datasets/tailornet_dataset
cd datasets/tailornet_dataset
wget -c https://datasets.d2.mpi-inf.mpg.de/tailornet/dataset_meta.zip
wget -c https://datasets.d2.mpi-inf.mpg.de/tailornet/old-t-shirt_female.zip
wget -c https://datasets.d2.mpi-inf.mpg.de/tailornet/t-shirt_female.zip
wget -c https://datasets.d2.mpi-inf.mpg.de/tailornet/t-shirt_male.zip

unzip dataset_meta.zip
unzip old-t-shirt_female.zip
unzip t-shirt_female.zip
unzip t-shirt_male.zip
#rm -rf dataset_meta.zip
#rm -rf old-t-shirt_female.zip
#rm -rf t-shirt_female.zip
#rm -rf t-shirt_male.zip

# checkpoints
cd ${ROOT_DIR}
mkdir -p checkpoints
mkdir -p checkpoints/tailornet
cd checkpoints/tailornet
wget -c https://datasets.d2.mpi-inf.mpg.de/tailornet/old-t-shirt_female_weights.zip
wget -c https://datasets.d2.mpi-inf.mpg.de/tailornet/t-shirt_female_weights.zip
wget -c https://datasets.d2.mpi-inf.mpg.de/tailornet/t-shirt_male_weights.zip

unzip old-t-shirt_female_weights.zip
unzip t-shirt_female_weights.zip
unzip t-shirt_male_weights.zip
#rm -rf old-t-shirt_female_weights.zip
