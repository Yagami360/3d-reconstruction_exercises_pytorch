import os
import argparse
import numpy as np
import random
from tqdm import tqdm
from PIL import Image

# sklearn
from sklearn.model_selection import train_test_split

# PyTorch
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

# PyTorch 3D
import pytorch3d
#from pytorch3d import _C
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency

# 自作モジュール
from data.dataset import TempleteDataset, TempleteDataLoader
from models.networks import TempleteNetworks
from utils.utils import save_checkpoint, load_checkpoint
from utils.utils import board_add_image, board_add_images, save_image_w_norm, plot3d_mesh, save_plot3d_mesh

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="debug", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="dataset/templete_dataset")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument('--save_checkpoints_dir', type=str, default="checkpoints", help="モデルの保存ディレクトリ")
    parser.add_argument('--load_checkpoints_path', type=str, default="", help="モデルの読み込みファイルのパス")
    parser.add_argument('--tensorboard_dir', type=str, default="tensorboard", help="TensorBoard のディレクトリ")
    parser.add_argument("--n_epoches", type=int, default=100, help="エポック数")    
    parser.add_argument('--batch_size', type=int, default=4, help="バッチサイズ")
    parser.add_argument('--batch_size_valid', type=int, default=1, help="バッチサイズ")
    parser.add_argument('--lr', type=float, default=0.1, help="学習率")
    parser.add_argument('--beta1', type=float, default=0.5, help="学習率の減衰率")
    parser.add_argument('--beta2', type=float, default=0.999, help="学習率の減衰率")
    parser.add_argument("--n_diaplay_step", type=int, default=100,)
    parser.add_argument('--n_display_valid_step', type=int, default=500, help="valid データの tensorboard への表示間隔")
    parser.add_argument("--n_save_epoches", type=int, default=10,)
    parser.add_argument("--val_rate", type=float, default=0.01)
    parser.add_argument('--n_display_valid', type=int, default=8, help="valid データの tensorboard への表示数")

    parser.add_argument("--lambda_chamfer", type=float, default=1.0)
    parser.add_argument("--lambda_edge", type=float, default=1.0)
    parser.add_argument("--lambda_normal", type=float, default=0.01)
    parser.add_argument("--lambda_laplacian", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    parser.add_argument('--n_workers', type=int, default=4, help="CPUの並列化数（0 で並列化なし）")
    parser.add_argument('--use_cuda_benchmark', action='store_true', help="torch.backends.cudnn.benchmark の使用有効化")
    parser.add_argument('--use_cuda_deterministic', action='store_true', help="再現性確保のために cuDNN に決定論的振る舞い有効化")
    parser.add_argument('--detect_nan', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    if( args.debug ):
        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))

        print( "pytorch version : ", torch.__version__)
        print( "pytorch 3d version : ", pytorch3d.__version__)

    # 出力フォルダの作成
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    if not os.path.isdir( os.path.join(args.results_dir, args.exper_name) ):
        os.mkdir(os.path.join(args.results_dir, args.exper_name))
    if not( os.path.exists(args.save_checkpoints_dir) ):
        os.mkdir(args.save_checkpoints_dir)
    if not( os.path.exists(os.path.join(args.save_checkpoints_dir, args.exper_name)) ):
        os.mkdir( os.path.join(args.save_checkpoints_dir, args.exper_name) )

    # 実行 Device の設定
    if( args.device == "gpu" ):
        use_cuda = torch.cuda.is_available()
        if( use_cuda == True ):
            device = torch.device( "cuda" )
            #torch.cuda.set_device(args.gpu_ids[0])
            print( "実行デバイス :", device)
            print( "GPU名 :", torch.cuda.get_device_name(device))
            print("torch.cuda.current_device() =", torch.cuda.current_device())
        else:
            print( "can't using gpu." )
            device = torch.device( "cpu" )
            print( "実行デバイス :", device)
    else:
        device = torch.device( "cpu" )
        print( "実行デバイス :", device)

    # seed 値の固定
    if( args.use_cuda_deterministic ):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # NAN 値の検出
    if( args.detect_nan ):
        torch.autograd.set_detect_anomaly(True)

    # tensorboard 出力
    board_train = SummaryWriter( log_dir = os.path.join(args.tensorboard_dir, args.exper_name) )
    board_valid = SummaryWriter( log_dir = os.path.join(args.tensorboard_dir, args.exper_name + "_valid") )

    #================================
    # データセットの読み込み
    #================================    
    # メッシュファイルの読み込み / 頂点 vertexs と面 faces と aux の取得
    # verts : verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
    # faces : faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
    verts, faces, aux = load_obj( os.path.join( args.dataset_dir, "mesh", 'dolphin.obj' ) )
    verts = verts.to(device)
    #print( "verts : ", verts )      # tensor([[-0.0374,  0.4473,  0.1219], [ 0.0377,  0.4471,  0.1220], ...
    #print( "faces : ", faces )          # Faces(verts_idx=tensor([[   0,  646,  643], ... , materials_idx=tensor([-1, -1, -1,  ..., -1, -1, -1]))
    #print( "aux : ", aux )              # Properties(normals=None, verts_uvs=None, material_colors=None, texture_images=None, texture_atlas=None)
    #print( "verts.shape : ", verts.shape )  # torch.Size([2562, 3]) / [頂点数, xyz座標]

    faces_idx = faces.verts_idx.to(device)
    #print( "faces_idx : ", faces_idx )             # tensor([[   0,  646,  643], [   0,  643,  642], ...
    #print( "faces_idx.shape : ", faces_idx.shape )  # torch.Size([5120, 3]) / 

    # (0,0,0)を中心とする半径1の球にフィットするように正規化・中心化
    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale

    # メッシュのオブジェクト生成
    mesh_t = Meshes(verts=[verts], faces=[faces_idx])
    mesh_s = ico_sphere(4, device)
    #print( "mesh_t : ", mesh_t )    # <pytorch3d.structures.meshes.Meshes
    #print( "mesh_s : ", mesh_s )

    # メッシュの描写
    save_plot3d_mesh( mesh_t, os.path.join(args.results_dir, args.exper_name, "target_mesh.png" ), "target mesh" )
    save_plot3d_mesh( mesh_s, os.path.join(args.results_dir, args.exper_name, "source_mesh.png" ), "source mesh" )

    #================================
    # モデルの構造を定義する。
    #================================
    # 変換関数の形状は、src_meshの頂点数と同じ
    # mesh_s.verts_packed() : mesh に含まれる頂点リストを取得
    verts_deform = torch.full( mesh_s.verts_packed().shape, 0.0, device=device, requires_grad=True )
    #print( "mesh_s.verts_packed().shape : ", mesh_s.verts_packed().shape )
        
    #================================
    # optimizer_G の設定
    #================================
    optimizer_G = optim.Adam( params = [verts_deform], lr = args.lr, betas = (args.beta1,args.beta2) )

    #================================
    # loss 関数の設定
    #================================
    pass

    #================================
    # モデルの学習
    #================================ 
    print("Starting Training Loop...")
    n_print = 1
    step = 0
    for epoch in tqdm( range(args.n_epoches), desc = "epoches" ):
        #----------------------------------------------------
        # 生成器 の forword 処理
        #----------------------------------------------------
        # メッシュの変形
        mesh_s_new = mesh_s.offset_verts(verts_deform)

        # 各メッシュの表面から5000個の点をサンプリング
        sample_t = sample_points_from_meshes(mesh_t, 5000)
        sample_s = sample_points_from_meshes(mesh_s_new, 5000)

        #----------------------------------------------------
        # 生成器の更新処理
        #----------------------------------------------------
        # 損失関数を計算する
        loss_chamfer, _ = chamfer_distance(sample_t, sample_s)
        loss_edge = mesh_edge_loss(mesh_s_new)
        loss_normal = mesh_normal_consistency(mesh_s_new)
        loss_laplacian = mesh_laplacian_smoothing(mesh_s_new, method="uniform")
        loss_G = args.lambda_chamfer * loss_chamfer + args.lambda_edge * loss_edge + args.lambda_normal * loss_normal + args.lambda_laplacian * loss_laplacian

        # ネットワークの更新処理
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        #====================================================
        # 学習過程の表示
        #====================================================
        if( step == 0 or ( step % args.n_diaplay_step == 0 ) ):
            # lr
            for param_group in optimizer_G.param_groups:
                lr = param_group['lr']

            board_train.add_scalar('lr/learning rate', lr, step )

            # loss
            board_train.add_scalar('G/loss_G', loss_G.item(), step)
            board_train.add_scalar('G/loss_chamfer', loss_chamfer.item(), step)
            board_train.add_scalar('G/loss_edge', loss_edge.item(), step)
            board_train.add_scalar('G/loss_normal', loss_normal.item(), step)
            board_train.add_scalar('G/loss_laplacian', loss_laplacian.item(), step)
            print( "step={}, loss_G={:.5f}, loss_chamfer={:.5f}, loss_edge={:.5f}, loss_normal={:.5f}, loss_laplacian={:.5f}".format(step, loss_G.item(), loss_chamfer.item(), loss_edge.item(), loss_normal.item(), loss_laplacian.item()) )

            # visual images
            save_plot3d_mesh( mesh_s_new, os.path.join(args.results_dir, args.exper_name, "source_mesh_step{}.png".format(step) ), "source mesh" )
            """
            visuals = [
                [ image, target, output ],
            ]
            board_add_images(board_train, 'train', visuals, step+1)
            """

        step += 1
        n_print -= 1

        #====================================================
        # モデルの保存
        #====================================================
        if( epoch % args.n_save_epoches == 0 ):
            print( "saved mesh" )
            # Fetch the verts and faces of the final predicted mesh
            final_verts, final_faces = mesh_s_new.get_mesh_verts_faces(0)

            # Scale normalize back to the original target size
            final_verts = final_verts * scale + center

            # Store the predicted mesh using save_obj
            save_obj( os.path.join(os.path.join(args.save_checkpoints_dir), 'mesh_step{:.5f}.obj'.format(step)), final_verts, final_faces )

    print("Finished Training Loop.")
    save_obj( os.path.join(os.path.join(args.save_checkpoints_dir), 'mesh_final.obj'), final_verts, final_faces )
