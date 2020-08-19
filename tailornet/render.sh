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
    --cloth_type old-t-shirt --gender female \
    --smpl_registration_dir datasets/smpl_registrations \
    --cloth_info_path datasets/tailornet_dataset/garment_class_info.pkl \
    --kernel_sigma 0.01 \
    --shader soft_silhouette_shader \
    --debug

if [ $1 = "poweroff" ] ; then
    sudo poweroff
    sudo shutdown -h now
fi
