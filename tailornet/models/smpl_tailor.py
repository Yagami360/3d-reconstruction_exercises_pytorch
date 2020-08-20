# -*- coding:utf-8 -*-
import os
import numpy as np
import pickle
import scipy.sparse as sp

import torch
import torch.nn as nn

# mesh
from psbody.mesh import Mesh

# pytorch3d
from pytorch3d.structures import Meshes

# 自作モジュール
from models.smpl import SMPLModel
from models.tailor_networks import TailorNet
from utils.mesh import upsampling_mesh, remove_mesh_interpenetration


class SMPLTailorModel(SMPLModel):
    def __init__( 
        self,
        smpl_registration_dir,                  #
        tailornet_dataset_dir,
        load_checkpoints_dir,
        cloth_type = "old-t-shirt",
        gender = "female",
        batch_size = 1, 
        kernel_sigma = 0.01,
        device = torch.device("cpu"),
        debug = False
    ):
        if( gender == "female" ):
            registration_path = os.path.join( smpl_registration_dir, "basicModel_f_lbs_10_207_0_v1.0.0.pkl" )
        elif( gender == "male" ):
            registration_path = os.path.join( smpl_registration_dir, "basicmodel_m_lbs_10_207_0_v1.0.0.pkl" )
        else:
            registration_path = os.path.join( smpl_registration_dir, "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl" )

        super(SMPLTailorModel, self).__init__( 
            registration_path = registration_path, 
            batch_size = batch_size,
            device = device,
            debug = debug
        )

        self.cloth_type = cloth_type
        self.gender = gender
        self.kernel_sigma = kernel_sigma
        self.batch_size = batch_size
        self.device = device
        self.debug = debug

        #----------------------------
        # メッシュを高解像度化
        #----------------------------
        hres_verts, hres_faces, mapping = upsampling_mesh(self.v_template.detach().cpu().numpy(), self.faces)

        #----------------------------------------------------------
        # 高解像度化したメッシュを元に smpl registration param を再生成
        #----------------------------------------------------------
        self.v_template = torch.from_numpy(hres_verts).float().requires_grad_(False).to(device)
        self.v_personal = torch.zeros( (self.v_template.shape), requires_grad=False).float().to(device)
        self.faces = hres_faces
        self.weights = torch.from_numpy(
            np.hstack([
                np.expand_dims(
                    np.mean(
                        mapping.dot(np.repeat(np.expand_dims(self.weights.detach().cpu().numpy()[:, i], -1), 3)).reshape(-1, 3), axis=1),
                    axis=-1)
                for i in range(24)
            ])
        ).float().requires_grad_(False).to(device)

        self.J_regressor = torch.from_numpy(
            np.array(
                sp.csr_matrix(
                    ( self.J_reg_csr.data, self.J_reg_csr.indices, self.J_reg_csr.indptr ), shape=(24, hres_verts.shape[0] )
                ).todense()
            )
        ).float().requires_grad_(False).to(device)

        self.joint_regressor = torch.from_numpy(
            np.array(
                sp.csr_matrix(
                    ( self.J_reg_csr.data, self.J_reg_csr.indices, self.J_reg_csr.indptr ), shape=(24, hres_verts.shape[0] )
                ).todense()
            )
        ).float().requires_grad_(False).to(device)

        self.posedirs = torch.from_numpy(mapping.dot(self.posedirs.detach().cpu().numpy().reshape((-1, 207))).reshape(-1, 3, 207)).float().requires_grad_(False).to(device)
        self.shapedirs = torch.from_numpy(mapping.dot(self.shapedirs.detach().cpu().numpy().reshape((-1, self.shapedirs.shape[-1]))).reshape(-1, 3, self.shapedirs.shape[-1])).float().requires_grad_(False).to(device)

        #----------------------------------------------------------
        # cloth_info_path : SMPL裸体人体メッシュと服メッシュの対応頂点 index 
        #----------------------------------------------------------
        with open( os.path.join(tailornet_dataset_dir, "garment_class_info.pkl" ), 'rb') as f:
            self.cloth_info = pickle.load(f)

        if( debug ):
            print( "self.cloth_info.keys() : ", self.cloth_info.keys() )
            print( "self.cloth_info[{}].keys() : {}".format(cloth_type,self.cloth_info[cloth_type].keys()) )
            print( "self.cloth_info[{}][vert_indices].shape : {}".format(cloth_type,self.cloth_info[cloth_type]["vert_indices"].shape[0]) )

        #----------------------------------------------------------
        # TailorNet
        #----------------------------------------------------------
        self.tailornet = TailorNet( 
            tailornet_dataset_dir = tailornet_dataset_dir, 
            load_checkpoints_dir = load_checkpoints_dir, 
            cloth_type = self.cloth_type, 
            gender = self.gender, 
            kernel_sigma = self.kernel_sigma,
            device = self.device,
            debug = self.debug
        ).to(device)

        return

    def forward(self, betas = None, thetas = None, gammas = None, trans = None, simplify = False  ):
        """
        [args]
            betas : SMPL の人物形状パラメータ beta / shape = [B,10]
            thetas : SMPL の人物姿勢パラメータ theta / shape = [B,72]
            gammas : 服メッシュのスタイル theta / shape = [B,4]
            trans : ワールド座標系への変換行列 / shape = [B,3]
        [return]
            result : メッシュの頂点座標 / shape = [B,6890,3]
            faces : メッシュの面情報 / shape = [B,13776,3]
            joints : 関節点の位置情報 / shape = [B,19,3]
        """
        if( gammas is None ):
            gammas = torch.from_numpy( np.zeros((self.batch_size,4) ) ).float().requires_grad_(False).to(self.device)

        #-------------------------------------
        # TailorNet で服メッシュの頂点変位を算出
        #-------------------------------------
        self.tailornet.eval()
        with torch.no_grad():
            cloth_displacements = self.tailornet( betas, thetas, gammas )
            #print( "cloth_displacements.shape : ", cloth_displacements.shape )
            #print( "cloth_displacements : ", cloth_displacements )
            #print( "[cloth_displacements] min={}, max={}, sum={}".format(torch.min(cloth_displacements), torch.max(cloth_displacements), torch.sum(cloth_displacements)) )

        #-------------------------------------
        # 頂点変位を v_personal に適用
        #-------------------------------------
        verts_indices_cloth = self.cloth_info[self.cloth_type]['vert_indices'].tolist()
        if( cloth_displacements is not None ):
            self.v_personal[verts_indices_cloth] = cloth_displacements

        #-------------------------
        # SMPL 裸体メッシュの生成
        #-------------------------
        verts_body, faces_body, joints_body = super().forward(betas, thetas, trans, simplify)

        #-------------------------
        # 服メッシュの生成
        #-------------------------
        verts_cloth = verts_body[:,verts_indices_cloth,:]
        faces_cloth = torch.from_numpy(self.cloth_info[self.cloth_type]['f']).int().unsqueeze(0).requires_grad_(False).to(self.device)

        #-------------------------
        # remove_mesh_interpenetration
        #-------------------------
        mesh_body = Meshes(verts_body, faces_body).to(self.device)
        mesh_cloth = Meshes(verts_cloth, faces_cloth).to(self.device)
        mesh_cloth = remove_mesh_interpenetration( mesh_cloth, mesh_body, device = self.device )

        return mesh_body, mesh_cloth

