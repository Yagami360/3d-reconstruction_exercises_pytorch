# -*- coding:utf-8 -*-
import os
import numpy as np
import pickle

import torch
import torch.nn as nn

# mesh
from psbody.mesh import Mesh

# 自作モジュール
from data.smpl import SMPLModel

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
        super(SMPLMGNModel, self).__init__(registration_path, batch_size, device, debug)
        self.digital_wardrobe_registration_path = digital_wardrobe_registration_path
        self.cloth_smpl_fts_path = cloth_smpl_fts_path
        self.cloth_type = cloth_type

        # digital_wardrobe にある betas, thetas を保管した registration file からデータを抽出
        with open(self.digital_wardrobe_registration_path, 'rb') as f:
            params = pickle.load(f, encoding='latin1')
            if( debug ):
                print( "params.keys() :\n", params.keys() )     # dict_keys(['gender', 'trans', 'pose', 'betas'])

        self.betas = torch.from_numpy(np.array(params['betas'])).float().requires_grad_(False).to(device)
        self.thetas = torch.from_numpy(np.array(params['pose'])).float().requires_grad_(False).to(device)
        self.trans = torch.from_numpy(np.array(params['trans'])).float().requires_grad_(False).to(device)
        self.gender = params['gender']

        # SMPL 人体メッシュと服メッシュの一致する頂点番号を保管した registration file からデータを抽出
        with open(self.cloth_smpl_fts_path, 'rb') as f:
            vert_indices, fts = pickle.load(f, encoding='latin1')
            if( debug ):
                print( "vert_indices.keys() :\n", vert_indices.keys() )     # dict_keys(['Pants', 'ShirtNoCoat', 'TShirtNoCoat', 'ShortPants', 'LongCoat'])
                print( "fts.keys() :\n", fts.keys() )                       # dict_keys(['Pants', 'ShirtNoCoat', 'TShirtNoCoat', 'ShortPants', 'LongCoat'])

        self.vert_indices = torch.from_numpy(np.array(vert_indices[self.cloth_type])).int().requires_grad_(False).to(device)
        self.fts = torch.from_numpy(np.array(fts[self.cloth_type])).int().requires_grad_(False).to(device)
        if( debug ):
            print( "self.betas.shape : ", self.betas.shape )
            print( "self.thetas.shape : ", self.thetas.shape )
            print( "self.trans.shape : ", self.trans.shape )
            #print( "self.betas : ", self.betas )
            #print( "self.thetas : ", self.thetas )
            #print( "self.trans : ", self.trans )
            print( "self.gender : ", self.gender )

            print( "self.vert_indices.shape : ", self.vert_indices.shape )
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

