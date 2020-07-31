#!/bin/sh
set -eu

mkdir -p dataset/
wget https://dl.fbaipublicfiles.com/pytorch3d/data/dolphin/dolphin.obj
mv dolphin.obj dataset/
