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

# opendr
from opendr.topology import loop_subdivider

# pytorch3d
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

#====================================================
# メッシュの操作関連
#====================================================
def upsampling_mesh( verts, faces ):
    """
    メッシュの頂点数を増やし、メッシュを高解像度化する
    [args]
        verts : numpy / shape=[V,3]
        faces : numpy / shape=[F,3]
    """
    (mapping, hres_faces) = loop_subdivider(verts, faces)
    hres_verts = mapping.dot(verts.ravel()).reshape(-1, 3)
    return hres_verts, hres_faces, mapping


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


def repose_mesh( src_mesh, smpl, vert_indices, device = torch.device("cpu") ):
    """
    Multi-Garment Net の SMPL での衣装テンプレートメッシュの変形（論文 (3) 式）
    D^g = G^g - I^g * T(β^g, 0^θ, 0^D) ---- (3)
    衣装テンプレートメッシュの対応する SMPL 体型メッシュへの頂点変形 D を計算

    src_mesh : 変形元の衣装テンプレートメッシュ / G^g
    smpl : 変形先の制御パラメーター β, 0^θ での SMPL 人体メッシュ / T(β^g, 0^θ, 0^D)
    vert_indices :  / I^g
    """
    if( len(src_mesh.verts_packed().shape) == 2 ):
        batch_size = 1
    else:
        batch_size = src_mesh.verts_packed().shape[0]

    # 制御パラメーター θ を 0^θ（ゼロ）にして、メッシュ再生成
    thetas = smpl.thetas
    thetas_zero = torch.zeros( (batch_size, 72), requires_grad=False).float().to(device)
    tar_verts, tar_faces, tar_joints = smpl( thetas = thetas_zero )

    # D^g = offset の計算
    offsets = torch.zeros( tar_verts[0].shape ).float().to(device)
    #print( "offsets.shape : ", offsets.shape )
    #print( "src_mesh.verts_packed().shape : ", src_mesh.verts_packed().shape )
    #print( "tar_verts.shape : ", tar_verts.shape )
    #print( "tar_verts[0].shape : ", tar_verts[0].shape )
    #print( "np.min(vert_indices)={}, np.max(vert_indices)={}".format(np.min(vert_indices), np.max(vert_indices)) )
    #print( "tar_verts[0][vert_indices].shape : ", tar_verts[0][vert_indices].shape )
    if( batch_size == 1 ):
        offsets[vert_indices] = src_mesh.verts_packed() - tar_verts[0][vert_indices]
    else:
        NotImplementedError()

    #print( "offsets : ", offsets )
    #smpl.v_personal = offsets

    # 
    smpl.thetas = thetas
    tar_verts, tar_faces, tar_joints = smpl( thetas = thetas )

    # メッシュの再生成
    new_mesh = Mesh(tar_verts[0], tar_faces[0]).keep_vertices(vert_indices)
    if( batch_size == 1 ):
        #print( "new_mesh.v.shape={}, new_mesh.f.shape={}".format(new_mesh.v.shape, new_mesh.f.shape) )
        #print( "new_mesh.v.dtype={}, new_mesh.f.dtype={}".format(new_mesh.v.dtype, new_mesh.f.dtype) )
        new_mesh = Meshes( torch.from_numpy(new_mesh.v).requires_grad_(False).float().unsqueeze(0), torch.from_numpy(new_mesh.f.astype(np.int32)).requires_grad_(False).int().unsqueeze(0) ).to(device)
    else:
        NotImplementedError()

    return new_mesh
