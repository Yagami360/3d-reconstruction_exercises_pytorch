# -*- coding:utf-8 -*-
import os
import numpy as np
import io
from PIL import Image
import cv2
import imageio
import random
import matplotlib.pyplot as plt

import scipy.sparse as sp
from scipy.sparse import vstack, csr_matrix
from scipy.sparse.linalg import spsolve

from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

# mesh
from psbody.mesh import Mesh
from psbody.mesh.search import AabbTree
from psbody.mesh.geometry.tri_normals import TriNormals
from psbody.mesh.geometry.vert_normals import VertNormals
from psbody.mesh.topology.connectivity import get_vert_connectivity

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
    mesh = Mesh( mesh.verts_packed().detach().cpu().numpy(), mesh.faces_packed().detach().cpu().numpy() )
    src_mesh = Mesh( src_mesh.verts_packed().detach().cpu().numpy(), src_mesh.faces_packed().detach().cpu().numpy() )
    tar_mesh = Mesh( tar_mesh.verts_packed().detach().cpu().numpy(), tar_mesh.faces_packed().detach().cpu().numpy() )

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
    smpl.v_personal = offsets

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


def remove_mesh_interpenetration( mesh, base_mesh, laplacian = None, device = torch.device("cpu") ):
    """
    Laplacian deformation による衣装テンプレートメッシュの変形？
    重なっている領域を除去する？
    [args]
        mesh : 衣装メッシュ
        base_mesh : SMPL人体メッシュ
    """
    def calc_laplacian(mesh):
        """
        メッシュに対して Laplacian deformation での Laplacian を計算する
        """
        # pytorch3d -> psbody.mesh への変換
        mesh = Mesh( mesh.verts_packed().detach().cpu().numpy(), mesh.faces_packed().detach().cpu().numpy() )

        # メッシュを頂点連結関係を取得？
        connectivity = get_vert_connectivity(mesh)

        # connectivity is a sparse matrix, and np.clip can not applied directly on a sparse matrix.
        connectivity.data = np.clip(connectivity.data, 0, 1)
        laplacian = normalize(connectivity, norm='l1', axis=1)
        laplacian = sp.eye(connectivity.shape[0]) - laplacian
        return laplacian

    def get_nearest_points_and_normals(vert, base_verts, base_faces):
        """
        inspired from frankengeist.body.ch.mesh_distance.MeshDistanceSquared
        """
        fn = TriNormals(v=base_verts, f=base_faces).reshape((-1, 3))
        vn = VertNormals(v=base_verts, f=base_faces).reshape((-1, 3))
        tree = AabbTree(Mesh(v=base_verts, f=base_faces))

        nearest_tri, nearest_part, nearest_point = tree.nearest(vert, nearest_part=True)
        nearest_tri = nearest_tri.ravel().astype(np.long)
        nearest_part = nearest_part.ravel().astype(np.long)
        nearest_normals = np.zeros_like(vert)

        # nearest_part tells you whether the closest point in triangle abc is in the interior (0), on an edge (ab:1,bc:2,ca:3), or a vertex (a:4,b:5,c:6)
        cl_tri_idxs = np.nonzero(nearest_part == 0)[0].astype(np.int)
        cl_vrt_idxs = np.nonzero(nearest_part > 3)[0].astype(np.int)
        cl_edg_idxs = np.nonzero((nearest_part <= 3) & (nearest_part > 0))[0].astype(np.int)

        nt = nearest_tri[cl_tri_idxs]
        nearest_normals[cl_tri_idxs] = fn[nt]

        nt = nearest_tri[cl_vrt_idxs]
        npp = nearest_part[cl_vrt_idxs] - 4
        nearest_normals[cl_vrt_idxs] = vn[base_faces[nt, npp]]

        nt = nearest_tri[cl_edg_idxs]
        npp = nearest_part[cl_edg_idxs] - 1
        nearest_normals[cl_edg_idxs] += vn[base_faces[nt, npp]]
        npp = np.mod(nearest_part[cl_edg_idxs], 3)
        nearest_normals[cl_edg_idxs] += vn[base_faces[nt, npp]]
        nearest_normals = nearest_normals / (np.linalg.norm(nearest_normals, axis=-1, keepdims=True) + 1.e-10)
        return nearest_point, nearest_normals

    # laplacian を計算
    if( laplacian is None ):
        laplacian = calc_laplacian(mesh)

    # mesh と base_mesh 間で、最も近い頂点ち法線ベクトルを取得？
    nearest_points, nearest_normals = get_nearest_points_and_normals(
        mesh.verts_packed().detach().cpu().numpy(), 
        base_mesh.verts_packed().detach().cpu().numpy(), 
        base_mesh.faces_packed().detach().cpu().numpy()
    )

    # ?
    direction = np.sign(np.sum((mesh.verts_packed().detach().cpu().numpy() - nearest_points) * nearest_normals, axis=-1))
    indices = np.where(direction < 0)[0]

    eps = 0.001
    ww = 2.0
    n_verts = mesh.num_verts_per_mesh()[0].detach().cpu().numpy()

    pentgt_points = nearest_points[indices] - mesh.verts_packed().detach().cpu().numpy()[indices]
    pentgt_points = nearest_points[indices] + eps * pentgt_points / np.expand_dims(0.0001 + np.linalg.norm(pentgt_points, axis=1), 1)
    tgt_points = mesh.verts_packed().detach().cpu().numpy().copy()
    tgt_points[indices] = ww * pentgt_points

    rc = np.arange(n_verts)
    data = np.ones(n_verts)
    data[indices] *= ww
    I = csr_matrix((data, (rc, rc)), shape=(n_verts, n_verts))

    A = vstack([laplacian, I])
    b = np.vstack(( laplacian.dot(mesh.verts_packed().detach().cpu().numpy()), tgt_points ))
    res = spsolve(A.T.dot(A), A.T.dot(b))

    # 
    new_mesh = Meshes( torch.from_numpy(res).float().unsqueeze(0), mesh.faces_packed().unsqueeze(0) ).to(device)
    return new_mesh


#====================================================
# SMPL 関連
#====================================================
def normalize_y_rotation(raw_theta):
    """Rotate along y axis so that root rotation can always face the camera.
    Theta should be a [3] or [72] numpy array.
    """
    only_global = True
    if raw_theta.shape == (72,):
        theta = raw_theta[:3]
        only_global = False
    else:
        theta = raw_theta[:]
    raw_rot = cv2.Rodrigues(theta)[0]
    rot_z = raw_rot[:, 2]
    # we should rotate along y axis counter-clockwise for t rads to make the object face the camera
    if rot_z[2] == 0:
        t = (rot_z[0] / np.abs(rot_z[0])) * np.pi / 2
    elif rot_z[2] > 0:
        t = np.arctan(rot_z[0]/rot_z[2])
    else:
        t = np.arctan(rot_z[0]/rot_z[2]) + np.pi
    cost, sint = np.cos(t), np.sin(t)
    norm_rot = np.array([[cost, 0, -sint],[0, 1, 0],[sint, 0, cost]])
    final_rot = np.matmul(norm_rot, raw_rot)
    final_theta = cv2.Rodrigues(final_rot)[0][:, 0]
    if not only_global:
        return np.concatenate([final_theta, raw_theta[3:]], 0)
    else:
        return final_theta