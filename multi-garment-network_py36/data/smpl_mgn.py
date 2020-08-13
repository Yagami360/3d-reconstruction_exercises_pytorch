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
    def __init__( self, registration_path, digital_wardrobe_registration_path, batch_size = 1, device = torch.device( "cpu" ), debug = False ):
        super(SMPLMGNModel, self).__init__(registration_path, batch_size, device, debug)
        self.digital_wardrobe_registration_path = digital_wardrobe_registration_path

        # registration の読み込み        
        with open(self.digital_wardrobe_registration_path, 'rb') as f:
            # encoding='latin1' : python2 で書き込まれた pickle を python3 で読み込むときに必要 / 要 chumpy
            params = pickle.load(f, encoding='latin1')
            if( debug ):
                print( "params.keys() :\n", params.keys() )     # dict_keys(['gender', 'trans', 'pose', 'betas'])

        # registration からデータを抽出
        self.betas = torch.from_numpy(np.array(params['betas'])).float().requires_grad_(False).to(device)
        self.thetas = torch.from_numpy(np.array(params['pose'])).float().requires_grad_(False).to(device)
        self.trans = torch.from_numpy(np.array(params['trans'])).float().requires_grad_(False).to(device)
        self.gender = params['gender']
        if( debug ):
            print( "self.betas.shape : ", self.betas.shape )
            print( "self.thetas.shape : ", self.thetas.shape )
            print( "self.trans.shape : ", self.trans.shape )
            #print( "self.betas : ", self.betas )
            #print( "self.thetas : ", self.thetas )
            #print( "self.trans : ", self.trans )
            print( "self.gender : ", self.gender )

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

