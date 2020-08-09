#!/bin/sh
#conda activate pytorch15_py36
set -eu
mkdir -p _logs

#----------------------
# model
#----------------------
EXPER_NAME=render_cloth
rm -rf tensorboard/${EXPER_NAME}

python render.py \
    --exper_name ${EXPER_NAME} \
    --mesh_file dataset/cloth_mesh/TShirtNoCoat.obj \
    --shader soft_silhouette_shader \
    --debug

#    --ffd_param_file dataset/ffd_params/parameters_test_ffd_iges.prm \

if [ $1 = "poweroff" ] ; then
    sudo poweroff
    sudo shutdown -h now
fi
