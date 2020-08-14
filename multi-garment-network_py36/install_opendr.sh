#!/bin/sh
set -eu
conda activate pytorch15_py36

# opendr for python3.6
git clone https://github.com/polmorenoc/opendr.git
cd opendr/opendr
python setup.py build
python setup.py install
#pip install opendr==0.77
