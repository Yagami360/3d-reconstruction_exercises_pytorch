#!/bin/sh
#conda activate pytorch15_py36
set -eu
mkdir -p _logs

#----------------------
# model
#----------------------
EXPER_NAME=debug
rm -rf tensorboard/${EXPER_NAME}

python render.py \
    --exper_name ${EXPER_NAME} \
    --smpl_registration_path datasets/smpl_registrations/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl \
    --shader soft_silhouette_shader \
    --debug


#    --smpl_registration_path datasets/digital_wardrobe/Multi-Garment_dataset/125611508622317/registration.pkl \

if [ $1 = "poweroff" ] ; then
    sudo poweroff
    sudo shutdown -h now
fi
