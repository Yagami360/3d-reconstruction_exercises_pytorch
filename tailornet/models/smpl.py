# -*- coding:utf-8 -*-
import os
import numpy as np
import pickle
import scipy.sparse as sp

import torch
import torch.nn as nn

class SMPLModel(nn.Module):
    def __init__( self, registration_path, batch_size = 1, device = torch.device( "cpu" ), debug = False ):
        super(SMPLModel, self).__init__()
        self.device = device
        self.registration_path = registration_path
        self.batch_size = batch_size

        #----------------------------
        # registration の読み込み        
        #----------------------------
        with open(self.registration_path, 'rb') as f:
            # encoding='latin1' : python2 で書き込まれた pickle を python3 で読み込むときに必要 / 要 chumpy
            params = pickle.load(f, encoding='latin1')
            if( debug ):
                print( "params.keys() :\n", params.keys() ) # dict_keys(['J_regressor_prior', 'f', 'J_regressor', 'kintree_table', 'J', 'weights_prior', 'weights', 'vert_sym_idxs', 'posedirs', 'pose_training_info', 'bs_style', 'v_template', 'shapedirs', 'bs_type'])

        #--------------------------------------
        # smpl registration param を抽出
        #--------------------------------------
        self.weights = torch.from_numpy(np.array(params['weights'])).float().requires_grad_(False).to(device)
        self.posedirs = torch.from_numpy(np.array(params['posedirs'])).float().requires_grad_(False).to(device)
        self.v_template = torch.from_numpy(np.array(params['v_template'])).float().requires_grad_(False).to(device)
        self.shapedirs = torch.from_numpy(np.array(params['shapedirs'])).float().requires_grad_(False).to(device)
        self.kintree_table = params['kintree_table']
        self.faces = np.array(params['f'])

        self.J_reg_csr = params['J_regressor'].asformat('csr')
        #self.J_regressor = torch.from_numpy(np.array(params['J_regressor'].todense())).float().requires_grad_(False).to(device)
        self.J_regressor = torch.from_numpy(
            np.array(
                sp.csr_matrix(
                    ( self.J_reg_csr.data, self.J_reg_csr.indices, self.J_reg_csr.indptr ), shape=(24, self.v_template.shape[0] )
                ).todense()
            )
        ).float().requires_grad_(False).to(device)

        if 'joint_regressor' in params.keys():
            self.joint_regressor = torch.from_numpy(np.array(params['joint_regressor'].T.todense())).float().requires_grad_(False).to(device)
        else:
            #self.joint_regressor = torch.from_numpy( np.array(params['J_regressor'].todense())).float().requires_grad_(False).to(device)
            self.joint_regressor = torch.from_numpy(
                np.array(
                    sp.csr_matrix(
                        ( self.J_reg_csr.data, self.J_reg_csr.indices, self.J_reg_csr.indptr ), shape=(24, self.v_template.shape[0] )
                    ).todense()
                )
            ).float().requires_grad_(False).to(device)

        if 'bs_type' in params.keys():
            self.bs_type = params['bs_type']
        if 'bs_style' in params.keys():
            self.bs_style = params['bs_style']
        if 'J' in params.keys():
            self.J = params['J']
        if 'v_personal' in params.keys():
            self.v_personal = torch.from_numpy(np.array(params['v_personal'])).float().to(device)
        else:
            self.v_personal = torch.zeros( (self.v_template.shape), requires_grad=False).float().to(device)

        #-------------------------------
        # SMPL 制御パラメータ初期化
        #-------------------------------
        self.betas = torch.zeros( (self.batch_size, 10), requires_grad=False).float().to(device)
        self.thetas = torch.zeros( (self.batch_size, 72), requires_grad=False).float().to(device)
        self.trans = torch.from_numpy(np.zeros((self.batch_size, 3))).float().requires_grad_(False).to(self.device)
        if( debug ):
            print( "self.registration_path : ", self.registration_path )            
            print( "self.J_regressor.shape : ", self.J_regressor.shape )            # torch.Size([24, V])
            print( "self.joint_regressor.shape : ", self.joint_regressor.shape )    # torch.Size([24, V])
            print( "self.posedirs.shape : ", self.posedirs.shape )                  # torch.Size([24, V])
            print( "self.v_template.shape : ", self.v_template.shape )              # torch.Size([V, 3])
            print( "self.v_personal.shape : ", self.v_personal.shape )              # 
            print( "self.weights.shape : ", self.weights.shape )                    # 
            print( "self.shapedirs.shape : ", self.shapedirs.shape )                # torch.Size([V, 3, 10])
            print( "self.kintree_table.shape : ", self.kintree_table.shape )        # (2, 24)
            print( "self.faces.shape : ", self.faces.shape )                        # (F, 3)

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
        if( betas is None ):
            betas = self.betas
        if( thetas is None ):
            thetas = self.thetas
        if( trans is None ):
            trans = self.trans

        self.betas = betas
        self.thetas = thetas
        #print( "betas.shape : ", betas.shape )
        #print( "thetas.shape : ", thetas.shape )
        #print( "trans.shape : ", trans.shape )

        # idx から col へのマップ
        id_to_col = { self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1]) }
        #print( "id_to_col : ", id_to_col )

        # 
        parent = { i: id_to_col[self.kintree_table[0, i]] for i in range(1, self.kintree_table.shape[1]) }
        #print( "parent : ", parent )

        # SMPL の人物形状パラメータ beta による頂点 v の変形
        v_shaped = torch.tensordot(betas, self.shapedirs, dims=([1], [2])) + self.v_template
        J = torch.matmul(self.J_regressor, v_shaped)

        # SMPL の人物姿勢パラメータ theta による頂点 v の変形
        R_cube_big = self.rodrigues(thetas.view(-1, 1, 3)).reshape(self.batch_size, -1, 3, 3)
        #print( "R_cube_big.shape : ", R_cube_big.shape )

        if simplify:
            v_posed = v_shaped + self.v_personal
        else:
            R_cube = R_cube_big[:, 1:, :, :]
            I_cube = (torch.eye(3, dtype=torch.float32).unsqueeze(dim=0) + torch.zeros((self.batch_size, R_cube.shape[1], 3, 3), dtype=torch.float32)).to(self.device)
            lrotmin = (R_cube - I_cube).reshape(self.batch_size, -1, 1).squeeze(dim=2)
            #print( "self.posedirs.shape : ", self.posedirs.shape )
            #print( "lrotmin.shape : ", lrotmin.shape )
            v_posed = v_shaped + torch.tensordot(lrotmin, self.posedirs, dims=([1], [2])) + self.v_personal

        results = []
        results.append( self.with_zeros(torch.cat((R_cube_big[:, 0], torch.reshape(J[:, 0, :], (-1, 3, 1))), dim=2)) )
        for i in range(1, self.kintree_table.shape[1]):
            results.append(
                torch.matmul(
                    results[parent[i]],
                    self.with_zeros( torch.cat( (R_cube_big[:, i], torch.reshape(J[:, i, :] - J[:, parent[i], :], (-1, 3, 1))), dim=2) )
                )
            )
        
        stacked = torch.stack(results, dim=1)
        results = stacked - \
            self.pack(
                torch.matmul(
                    stacked,
                    torch.reshape( torch.cat((J, torch.zeros((self.batch_size, 24, 1), dtype=torch.float32).to(self.device)), dim=2), (self.batch_size, 24, 4, 1) )
                )
            )

        # Restart from here
        T = torch.tensordot(results, self.weights, dims=([1], [1])).permute(0, 3, 1, 2)
        rest_shape_h = torch.cat( (v_posed, torch.ones((self.batch_size, v_posed.shape[1], 1), dtype=torch.float32).to(self.device)), dim=2 )
        v = torch.matmul(T, torch.reshape(rest_shape_h, (self.batch_size, -1, 4, 1)))
        v = torch.reshape(v, (self.batch_size, -1, 4))[:, :, :3]
        #print( "v.shape : ", v.shape )
        #print( "trans.shape : ", trans.shape )

        # ワールド座標変換
        result = v + torch.reshape(trans, (self.batch_size, 1, 3))
        #print( "result.shape : ", result.shape )

        # faces 
        faces = torch.from_numpy(self.faces.astype(np.float32)).int().unsqueeze(0).requires_grad_(False).to(self.device)
        #print( "faces.shape : ", faces.shape )

        # estimate 3D joint locations
        #joints = torch.tensordot(result, self.joint_regressor, dims=([1], [0])).transpose(1, 2)
        joints = torch.zeros((self.batch_size,19,3), requires_grad=False).float().to(self.device)     # dummy
        #print( "joints.shape : ", joints.shape )

        return result, faces, joints

    @staticmethod
    def rodrigues(r):
        """
        Rodrigues' rotation formula that turns axis-angle tensor into rotation
        matrix in a batch-ed manner.
        Parameter:
        ----------
        r: Axis-angle rotation tensor of shape [batch_size * angle_num, 1, 3].
        Return:
        -------
        Rotation matrix of shape [batch_size * angle_num, 3, 3].
        """
        eps = r.clone().normal_(std=1e-8)
        theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)  # dim cannot be tuple
        theta_dim = theta.shape[0]
        r_hat = r / theta
        cos = torch.cos(theta)
        z_stick = torch.zeros(theta_dim, dtype=torch.float32).to(r.device)
        m = torch.stack(
            (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
            -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1
        )
        m = torch.reshape(m, (-1, 3, 3))
        i_cube = (torch.eye(3, dtype=torch.float32).unsqueeze(dim=0) \
                + torch.zeros((theta_dim, 3, 3), dtype=torch.float32)).to(r.device)
        A = r_hat.permute(0, 2, 1)
        dot = torch.matmul(A, r_hat)
        R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
        return R

    @staticmethod
    def with_zeros(x):
        """
        Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.
        Parameter:
        ---------
        x: Tensor to be appended.
        Return:
        ------
        Tensor after appending of shape [4,4]
        """
        ones = torch.tensor(
            [[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32
        ).expand(x.shape[0],-1,-1).to(x.device)
        ret = torch.cat((x, ones), dim=1)
        return ret

    @staticmethod
    def pack(x):
        """
        Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.
        Parameter:
        ----------
        x: A tensor of shape [batch_size, 4, 1]
        Return:
        ------
        A tensor of shape [batch_size, 4, 4] after appending.
        """
        zeros43 = torch.zeros(
            (x.shape[0], x.shape[1], 4, 3), dtype=torch.float32).to(x.device)
        ret = torch.cat((zeros43, x), dim=3)
        return ret


