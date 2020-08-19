import os
import numpy as np
import pickle
import random
import re
from PIL import Image, ImageDraw, ImageOps
import cv2

# PyTorch
import torch
import torch.utils.data as data
from utils import set_random_seed


class TailornetDataset(data.Dataset):
    def __init__(self, dataset_dir, cloth_type = "old-t-shirt", gender = "female", shape_style_pair_list = "avail.txt", debug = False ):
        super(TailornetDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.cloth_type = cloth_type
        self.debug = debug
        if( gender == "neutral" ):
            self.gender = "female"
        else:
            self.gender = gender

        #--------------------
        # betas, gammas
        #--------------------
        if( self.cloth_type == 'old-t-shirt' ):
            self.betas = torch.from_numpy( np.stack([np.load(os.path.join(self.dataset_dir, "{}_{}".format(cloth_type,self.gender), 'shape/beta_{:03d}.npy'.format(i))) for i in range(9)]).astype(np.float32)[:, :10] )
            self.gammas = torch.from_numpy( np.stack([np.load(os.path.join(self.dataset_dir, "{}_{}".format(cloth_type,self.gender), 'style/gamma_{:03d}.npy'.format(i))) for i in range(26)]).astype(np.float32) )
        else:
            self.betas = torch.from_numpy( np.load(os.path.join(self.dataset_dir, "{}_{}".format(cloth_type,self.gender), 'shape/betas.npy'))[:, :10] )
            self.gammas = torch.from_numpy( np.load(os.path.join(self.dataset_dir, "{}_{}".format(cloth_type,self.gender), 'style/gammas.npy')) )

        #--------------------
        # A-pose
        #--------------------
        with open(os.path.join(self.dataset_dir, 'apose.pkl'), 'rb') as f:
            apose = np.array(pickle.load(f, encoding='latin1')['pose']).astype(np.float32)

        flip_pose = self.flip_theta(apose)
        apose[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15]] = 0
        apose[[14, 17, 19, 21, 23]] = flip_pose[[14, 17, 19, 21, 23]]
        apose = apose.reshape([72])
        self.apose = torch.from_numpy(apose)

        #--------------------
        # shape and style
        #--------------------
        with open(os.path.join(self.dataset_dir, "{}_{}".format(cloth_type,self.gender), shape_style_pair_list), "r") as f:
            self.shape_style_pairs = [line.strip().split('_') for line in f.readlines()]
        
        #--------------------
        # unpose_v : スキンメッシュアニメーションを行う関数 W の逆変換 W^-1 で unpose したメッシュ頂点
        #--------------------
        unpose_v = []
        for shape_idx, style_idx in self.shape_style_pairs:
            fpath = os.path.join( self.dataset_dir, "{}_{}".format(cloth_type,self.gender), 'style_shape/beta{}_gamma{}.npy'.format(shape_idx, style_idx) )
            if not os.path.exists(fpath):
                print("shape {} and style {} not available".format(shape_idx, style_idx))
            unpose_v.append(np.load(fpath))

        unpose_v = np.stack(unpose_v)
        self.unpose_v = torch.from_numpy(unpose_v)

        if( self.debug ):
            print( "self.betas.shape : ", self.betas.shape )            # torch.Size([9, 10])
            print( "self.gammas.shape : ", self.gammas.shape )          # torch.Size([25, 4])
            #print( "self.betas : ", self.betas )
            #print( "self.gammas : ", self.gammas )
            print( "self.apose.shape : ", self.apose.shape )            # torch.Size([72])
            #print( "self.apose : ", self.apose )
            print( "self.unpose_v.shape : ", self.unpose_v.shape )      # torch.Size([201, 4718, 3])
            #print( "self.shape_style_pairs : ", self.shape_style_pairs )

        return

    @staticmethod
    def flip_theta(theta, batch=False):
        """
        flip SMPL theta along y-z plane
        if batch is True, theta shape is Nx72, otherwise 72
        """
        exg_idx = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
        if batch:
            new_theta = np.reshape(theta, [-1, 24, 3])
            new_theta = new_theta[:, exg_idx]
            new_theta[:, :, 1:3] *= -1
        else:
            new_theta = np.reshape(theta, [24, 3])
            new_theta = new_theta[exg_idx]
            new_theta[:, 1:3] *= -1
        new_theta = new_theta.reshape(theta.shape)
        return new_theta

    def __len__(self):
        return self.unpose_v.shape[0]

    def __getitem__(self, index):
        shape_idx, style_idx = self.shape_style_pairs[index]
        shape_idx, style_idx = int(shape_idx), int(style_idx)
        return self.unpose_v[index], self.apose, self.betas[shape_idx], self.gammas[style_idx], index


class TailornetDataLoader(object):
    def __init__(self, dataset, batch_size = 1, shuffle = True, n_workers = 4, pin_memory = True):
        super(TailornetDataLoader, self).__init__()
        self.data_loader = torch.utils.data.DataLoader(
                dataset, 
                batch_size = batch_size, 
                shuffle = shuffle,
                num_workers = n_workers,
                pin_memory = pin_memory,
        )

        self.dataset = dataset
        self.batch_size = batch_size
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch