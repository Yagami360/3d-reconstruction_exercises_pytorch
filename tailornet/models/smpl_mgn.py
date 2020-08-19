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
from utils.mesh import upsampling_mesh

class SMPLMGNModel(SMPLModel):
    def __init__( 
        self,
        registration_path,                      # SMPL 人体メッシュの registration file
        digital_wardrobe_registration_path,     # digital_wardrobe にある betas, thetas を保管した registration file
        cloth_smpl_fts_path,                    # SMPL 人体メッシュと服メッシュの一致する頂点番号を保管した registration file
        cloth_type = "TShirtNoCoat",
        batch_size = 1, 
        device = torch.device("cpu"), debug = False
    ):
        super(SMPLMGNModel, self).__init__(registration_path, batch_size, device, debug = debug)
        self.digital_wardrobe_registration_path = digital_wardrobe_registration_path
        self.cloth_smpl_fts_path = cloth_smpl_fts_path
        self.cloth_type = cloth_type

        #----------------------------
        # メッシュを高解像度化
        #----------------------------
        hres_verts, hres_faces, mapping = upsampling_mesh(self.v_template.detach().cpu().numpy(), self.faces)
        #print( "hres_verts.shape={}, hres_faces.shape={}  : ".format(hres_verts.shape, hres_faces.shape) )

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
        # digital_wardrobe にある betas, thetas を保管した registration file からデータを抽出
        #----------------------------------------------------------
        if( os.path.exists(self.digital_wardrobe_registration_path) ):
            with open(self.digital_wardrobe_registration_path, 'rb') as f:
                params = pickle.load(f, encoding='latin1')
                if( debug ):
                    print( "params.keys() :\n", params.keys() )     # dict_keys(['gender', 'trans', 'pose', 'betas'])

            self.betas = torch.from_numpy(np.array(params['betas'])).float().requires_grad_(False).to(device)
            self.thetas = torch.from_numpy(np.array(params['pose'])).float().requires_grad_(False).to(device)
            self.trans = torch.from_numpy(np.array(params['trans'])).float().requires_grad_(False).to(device)
            self.gender = params['gender']
        else:
            self.betas = torch.zeros( (self.batch_size, 10), requires_grad=False).float().to(device)
            self.thetas = torch.zeros( (self.batch_size, 72), requires_grad=False).float().to(device)
            self.trans = torch.from_numpy(np.zeros((self.batch_size, 3))).float().requires_grad_(False).to(device)
            self.gender = "female"

        #----------------------------------------------------------
        # SMPL 人体メッシュと服メッシュの一致する頂点番号を保管した registration file からデータを抽出
        #----------------------------------------------------------
        if( os.path.exists(self.digital_wardrobe_registration_path) ):
            with open(self.cloth_smpl_fts_path, 'rb') as f:
                vert_indices, fts = pickle.load(f, encoding='latin1')
                if( debug ):
                    print( "vert_indices.keys() :\n", vert_indices.keys() )     # dict_keys(['Pants', 'ShirtNoCoat', 'TShirtNoCoat', 'ShortPants', 'LongCoat'])
                    print( "fts.keys() :\n", fts.keys() )                       # dict_keys(['Pants', 'ShirtNoCoat', 'TShirtNoCoat', 'ShortPants', 'LongCoat'])

            self.vert_indices = np.array(vert_indices[self.cloth_type])
            self.fts = np.array(fts[self.cloth_type])
        else:
            self.vert_indices = None
            self.fts = None

        if( debug ):
            print( "self.registration_path : ", self.registration_path )            
            print( "self.J_regressor.shape : ", self.J_regressor.shape )            # 
            print( "self.joint_regressor.shape : ", self.joint_regressor.shape )    # 
            print( "self.posedirs.shape : ", self.posedirs.shape )                  # 
            print( "self.v_template.shape : ", self.v_template.shape )              # 
            print( "self.v_personal.shape : ", self.v_personal.shape )              # 
            print( "self.weights.shape : ", self.weights.shape )                    # 
            print( "self.shapedirs.shape : ", self.shapedirs.shape )                # 
            print( "self.kintree_table.shape : ", self.kintree_table.shape )        # 
            print( "self.faces.shape : ", self.faces.shape )                        # 

            print( "self.betas.shape : ", self.betas.shape )
            print( "self.thetas.shape : ", self.thetas.shape )
            print( "self.trans.shape : ", self.trans.shape )
            #print( "self.betas : ", self.betas )
            #print( "self.thetas : ", self.thetas )
            #print( "self.trans : ", self.trans )
            print( "self.gender : ", self.gender )

            if( self.vert_indices is not None ):
                print( "self.vert_indices.shape : ", self.vert_indices.shape )
            if( self.fts is not None ):
                print( "self.fts.shape : ", self.fts.shape )
            #print( "self.vert_indices : ", self.vert_indices )
            #print( "self.fts : ", self.fts )

        return

    def forward(self, betas = None, thetas = None, trans = None, simplify = False ):
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
        return super().forward(betas, thetas, trans, simplify)

