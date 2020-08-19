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

# SMPL lib
from smpl_lib.ch_smpl import Smpl

# 自作モジュール
from models.smpl import SMPLModel
from models.smpl_mgn import SMPLMGNModel
from utils.mesh import upsampling_mesh, remove_mesh_interpenetration

class SMPLTailorModel2(nn.Module):
    def __init__( 
        self,
        smpl_registration_dir,                  #
        cloth_info_path,                        #
        cloth_type = "old-t-shirt",
        gender = "female",
        batch_size = 1, 
        device = torch.device("cpu"), debug = False
    ):
        super(SMPLTailorModel2, self).__init__()
        self.smpl_registration_dir = smpl_registration_dir
        self.cloth_info_path = cloth_info_path
        self.cloth_type = cloth_type
        self.gender = gender
        self.batch_size = batch_size
        self.device = device
        self.debug = debug

        if( gender == "female" ):
            smpl_registration_path = os.path.join( smpl_registration_dir, "basicModel_f_lbs_10_207_0_v1.0.0.pkl" )
        elif( gender == "male" ):
            smpl_registration_path = os.path.join( smpl_registration_dir, "basicmodel_m_lbs_10_207_0_v1.0.0.pkl" )
        else:
            smpl_registration_path = os.path.join( smpl_registration_dir, "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl" )

        #----------------------------
        # registration の読み込み        
        #----------------------------
        with open(smpl_registration_path, 'rb') as f:
            # encoding='latin1' : python2 で書き込まれた pickle を python3 で読み込むときに必要 / 要 chumpy
            params = pickle.load(f, encoding='latin1')
            if( debug ):
                print( "params.keys() :\n", params.keys() ) # dict_keys(['J_regressor_prior', 'f', 'J_regressor', 'kintree_table', 'J', 'weights_prior', 'weights', 'vert_sym_idxs', 'posedirs', 'pose_training_info', 'bs_style', 'v_template', 'shapedirs', 'bs_type'])

        #--------------------------------------
        # smpl registration param を抽出
        #--------------------------------------
        self.weights = np.array(params['weights'])
        self.posedirs = np.array(params['posedirs'])
        self.v_template = np.array(params['v_template'])
        self.shapedirs = np.array(params['shapedirs'])
        self.kintree_table = params['kintree_table']
        self.faces = np.array(params['f'])
        self.J_reg_csr = params['J_regressor'].asformat('csr')
        self.J_regressor = sp.csr_matrix( (self.J_reg_csr.data, self.J_reg_csr.indices, self.J_reg_csr.indptr ), shape=(24, self.v_template.shape[0]) )
        if 'J_regressor_prior' in params.keys():
            self.J_regressor_prior = params['J_regressor_prior']
        if 'bs_type' in params.keys():
            self.bs_type = params['bs_type']
        if 'bs_style' in params.keys():
            self.bs_style = params['bs_style']
        if 'J' in params.keys():
            self.J = params['J']
        if 'v_personal' in params.keys():
            self.v_personal = np.array(params['v_personal'])
        else:
            self.v_personal = np.zeros((self.v_template.shape))

        #----------------------------
        # メッシュを高解像度化
        #----------------------------
        hres_verts, hres_faces, mapping = upsampling_mesh(self.v_template, self.faces)

        #----------------------------------------------------------
        # 高解像度化したメッシュを元に smpl registration param を再生成
        #----------------------------------------------------------
        self.v_template = hres_verts
        self.faces = hres_faces
        self.weights = np.hstack([
                np.expand_dims(
                    np.mean(
                        mapping.dot(np.repeat(np.expand_dims(self.weights[:, i], -1), 3)).reshape(-1, 3), axis=1),
                    axis=-1)
                for i in range(24)
        ])
        self.J_regressor = sp.csr_matrix( (self.J_reg_csr.data, self.J_reg_csr.indices, self.J_reg_csr.indptr), shape=(24, hres_verts.shape[0]) )
        self.posedirs = mapping.dot(self.posedirs.reshape((-1, 207))).reshape(-1, 3, 207)
        self.shapedirs = mapping.dot(self.shapedirs.reshape((-1, self.shapedirs.shape[-1]))).reshape(-1, 3, self.shapedirs.shape[-1])
        self.v_personal = np.zeros_like(self.v_template)

        #----------------------------------------------------------
        # SMPL lib 
        #----------------------------------------------------------
        smpl_model = {
            'v_template': self.v_template,
            'weights': self.weights,
            'posedirs': self.posedirs,
            'shapedirs': self.shapedirs,
            'J_regressor': self.J_regressor,
            'kintree_table': self.kintree_table,
            'bs_type': self.bs_type,
            'bs_style': self.bs_style,
            'J': self.J,
            'f': self.faces,
        }
        self.smpl_base = Smpl(smpl_model)
        
        #
        self.cloth_info_path = cloth_info_path
        with open(self.cloth_info_path, 'rb') as f:
            self.cloth_info = pickle.load(f)

        if( debug ):
            print( "self.cloth_info.keys() : ", self.cloth_info.keys() )
            print( "self.cloth_info[{}].keys() : {}".format(cloth_type,self.cloth_info[cloth_type].keys()) )
            print( "self.cloth_info[{}][vert_indices].shape : {}".format(cloth_type,self.cloth_info[cloth_type]["vert_indices"].shape[0]) )

        return

    def forward(self, betas = None, thetas = None, trans = None, simplify = False, cloth_displacements = None ):
        """
        [args]
            betas : SMPL の人物形状パラメータ beta / shape = [B,10]
            thetas : SMPL の人物姿勢パラメータ theta / shape = [B,72]
            trans : ワールド座標系への変換行列 / shape = [B,3]
        [return]
            result : メッシュの頂点座標 / shape = [B,6890,3]
            faces : メッシュの面情報 / shape = [B,13776,3]
            joints : 関節点の位置情報 / shape = [B,19,3]
        """
        if betas is not None:
            self.smpl_base.betas[:] = betas[0]
        else:
            self.smpl_base.betas[:] = 0
        if thetas is not None:
            self.smpl_base.pose[:] = thetas[0]
        else:
            self.smpl_base.pose[:] = 0

        verts_indices_cloth = self.cloth_info[self.cloth_type]['vert_indices'].tolist()
        if( cloth_displacements is not None ):
            self.smpl_base.v_personal[verts_indices_cloth] = cloth_displacements

        # SMPL 裸体メッシュの生成
        verts_body = torch.from_numpy(self.smpl_base.r).float().unsqueeze(0).to(self.device)
        faces_body = torch.from_numpy(self.smpl_base.f).int().unsqueeze(0).to(self.device)

        # 服メッシュの生成
        verts_cloth = verts_body[:,verts_indices_cloth,:]
        faces_cloth = torch.from_numpy(self.cloth_info[self.cloth_type]['f']).int().unsqueeze(0).to(self.device)

        # remove_mesh_interpenetration
        mesh_body = Meshes(verts_body, faces_body).to(self.device)
        mesh_cloth = Meshes(verts_cloth, faces_cloth).to(self.device)
        mesh_cloth = remove_mesh_interpenetration( mesh_cloth, mesh_body )

        return mesh_body, mesh_cloth

