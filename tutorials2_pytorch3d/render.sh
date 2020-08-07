#!/bin/sh
#conda activate pytorch15_py36
set -eu
mkdir -p _logs

#----------------------
# model
#----------------------
EXPER_NAME=render_mesh
rm -rf tensorboard/${EXPER_NAME}

python render.py \
    --exper_name ${EXPER_NAME} \
    --mesh_file dataset/cow_mesh/cow.obj \
    --shader textured_soft_phong_shader \
    --debug

if [ $1 = "poweroff" ] ; then
    sudo poweroff
    sudo shutdown -h now
fi
