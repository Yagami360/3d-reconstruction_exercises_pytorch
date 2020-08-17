#!/bin/sh
set -eu
conda activate pytorch15_py36

# mesh のインストール : https://github.com/MPI-IS/mesh
git clone https://github.com/MPI-IS/mesh
cd mesh
#BOOST_INCLUDE_DIRS=/path/to/boost/include make all
CFLAGS="-mmacosx-version-min=10.7 -stdlib=libc++" make all     # for mac
