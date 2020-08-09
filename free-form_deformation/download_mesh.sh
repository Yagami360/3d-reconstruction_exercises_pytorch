#!/bin/sh
set -eu
mkdir -p dataset
mkdir -p dataset/cow_mesh

wget -P dataset/cow_mesh https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.obj
wget -P dataset/cow_mesh https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.mtl
wget -P dataset/cow_mesh https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow_texture.png
