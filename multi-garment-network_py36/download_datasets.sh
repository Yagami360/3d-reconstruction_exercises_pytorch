#!/bin/sh
# https://github.com/bharat-b7/MultiGarmentNetwork#dress-smpl-body-model-with-our-digital-wardrobe
set -eu

wget https://datasets.d2.mpi-inf.mpg.de/MultiGarmentNetwork/Multi-Garmentdataset.zip
wget https://datasets.d2.mpi-inf.mpg.de/MultiGarmentNetwork/Multi-Garmentdataset_02.zip
mv Multi-Garmentdataset.zip datasets/digital_wardrobe
mv Multi-Garmentdataset_02.zip datasets/digital_wardrobe

cd datasets/digital_wardrobe
unzip Multi-Garmentdataset.zip
unzip Multi-Garmentdataset_02.zip
rm -rf Multi-Garmentdataset.zip
rm -rf Multi-Garmentdataset_02.zip

# download smpl registrations files manually from http://smplify.is.tue.mpg.de/downloads, https://smpl.is.tue.mpg.de/downloads
# basicModel_f_lbs_10_207_0_v1.0.0.pkl
# basicmodel_m_lbs_10_207_0_v1.0.0.pkl
# basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
