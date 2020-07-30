#!/bin/sh
set -eu

mkdir -p dataset
mkdir -p dataset/templete_dataset
mkdir -p dataset/templete_dataset/mesh

wget https://dl.fbaipublicfiles.com/pytorch3d/data/dolphin/dolphin.obj
mv dolphin.obj dataset/templete_dataset/mesh
