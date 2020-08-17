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
from data.smpl import SMPLModel
from data.smpl_mgn import SMPLMGNModel
from utils.mesh import upsampling_mesh

class SMPLTailorModel(SMPLMGNModel):
    def __init__( 
        self,
        registration_path,                      # SMPL 人体メッシュの registration file
        cloth_info_path,                        #
        cloth_type = "old-t-shirt",
        batch_size = 1, 
        device = torch.device("cpu"), debug = False
    ):
        super(SMPLTailorModel, self).__init__(
            registration_path = registration_path, 
            digital_wardrobe_registration_path = "", 
            cloth_smpl_fts_path = "", 
            cloth_type = cloth_type,
            batch_size = batch_size, 
            device = device,
            debug = False
        )
        self.cloth_info_path = cloth_info_path
        with open(self.cloth_info_path, 'rb') as f:
            self.cloth_info = pickle.load(f)

        if( debug ):
            print( "self.cloth_info.keys() : ", self.cloth_info.keys() )
            print( "self.cloth_info[{}].keys() : {}".format(cloth_type,self.cloth_info[cloth_type].keys()) )

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
        verts_indices_cloth = self.cloth_info[self.cloth_type]['vert_indices'].tolist()
        if( cloth_displacements is not None ):
            self.v_personal[:,verts_indices_cloth,:] = cloth_displacements

        # SMPL 裸体メッシュの生成
        verts_body, faces_body, joints_body = super().forward(betas, thetas, trans, simplify)

        # 服メッシュの生成
        verts_cloth = verts_body[:,verts_indices_cloth,:]
        faces_cloth = torch.from_numpy(self.cloth_info[self.cloth_type]['f']).int().unsqueeze(0).requires_grad_(False).to(self.device)
        return verts_body, faces_body, joints_body, verts_cloth, faces_cloth

