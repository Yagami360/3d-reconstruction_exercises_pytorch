#!/bin/sh
#conda activate pytorch15_py36
set -eu
mkdir -p _logs

#----------------------
# model
#----------------------
EXPER_NAME=debug
rm -rf tensorboard/${EXPER_NAME}

python test.py \
    --exper_name ${EXPER_NAME} \
    --registration_path datasets/registrations/basicmodel_m_lbs_10_207_0_v1.0.0.pkl \
    --shader soft_silhouette_shader \
    --batch_size 1 \
    --debug

if [ $1 = "poweroff" ] ; then
    sudo poweroff
    sudo shutdown -h now
fi
