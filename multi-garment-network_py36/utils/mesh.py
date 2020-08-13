# -*- coding:utf-8 -*-
import os
import numpy as np
import io
from PIL import Image
import cv2
import imageio
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

# mesh
from psbody.mesh import Mesh

# pytorch3d
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

#====================================================
# メッシュの操作関連
#====================================================
def deform_mesh_by_closest_vertices( mesh, src_mesh, tar_mesh, device = torch.device("cpu") ):
    """
    mesh の頂点と最も近い src_mesh の頂点番号を元に、mesh を tar_mesh の形状に変更する。
    [ToDo] 複数バッチ処理に対応
    """    
    if( len(mesh.verts_packed().shape) == 2 ):
        batch_size = 1
    else:
        batch_size = mesh.verts_packed().shape[0]

    # pytorch3d -> psbody.mesh への変換
    mesh_face_pytorch3d = mesh.faces_packed()
    mesh = Mesh( mesh.verts_packed(), mesh.faces_packed() )
    src_mesh = Mesh( src_mesh.verts_packed(), src_mesh.faces_packed() )
    tar_mesh = Mesh( tar_mesh.verts_packed(), tar_mesh.faces_packed() )

    # verts_idx : 最も近い頂点番号
    verts_idx, _ = src_mesh.closest_vertices(mesh.v)
    verts_idx = np.array(verts_idx)
    new_mesh_verts = mesh.v - src_mesh.v[verts_idx] + tar_mesh.v[verts_idx]
    #print( "verts_idx : ", verts_idx )
    #print( "new_mesh_verts.shape : ", new_mesh_verts.shape )    # (7702, 3)
    #print( "mesh.f.shape : ", mesh.f.shape )                    # (15180, 3)

    # psbody.mesh -> pytorch3d への変換
    if( batch_size == 1 ):
        new_mesh = Meshes( torch.from_numpy(new_mesh_verts).requires_grad_(False).float().unsqueeze(0), mesh_face_pytorch3d.unsqueeze(0) ).to(device)
    else:
        NotImplementedError()

    return new_mesh

