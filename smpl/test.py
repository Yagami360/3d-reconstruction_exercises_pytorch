import os
import argparse
import numpy as np
import random
from tqdm import tqdm
from PIL import Image

# PyTorch
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

# PyTorch 3D
import pytorch3d
#from pytorch3d import _C
from pytorch3d.io import load_obj, save_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes                                                 # メッシュ関連
#from pytorch3d.structures import Textures                                               # テクスチャー関連
from pytorch3d.renderer import look_at_view_transform, OpenGLPerspectiveCameras         # カメラ関連
from pytorch3d.renderer import PointLights, DirectionalLights                           # ライト関連
from pytorch3d.renderer import Materials                                                # マテリアル関連
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer                    # ラスタライザー関連
from pytorch3d.renderer.mesh.shader import SoftSilhouetteShader, SoftPhongShader, TexturedSoftPhongShader     # シェーダー関連
from pytorch3d.renderer import MeshRenderer                                             # レンダラー関連

# 自作モジュール
from models.smpl import SMPLModel
from utils.utils import save_checkpoint, load_checkpoint
from utils.utils import board_add_image, board_add_images, save_image_w_norm, save_plot3d_mesh_img, get_plot3d_mesh_img, save_mesh_obj

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="smpl_render", help="実験名")
    parser.add_argument("--registration_path", type=str, default="datasets/registrations/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument('--save_checkpoints_dir', type=str, default="checkpoints", help="モデルの保存ディレクトリ")
    parser.add_argument('--load_checkpoints_path', type=str, default="", help="モデルの読み込みファイルのパス")
    parser.add_argument('--tensorboard_dir', type=str, default="tensorboard", help="TensorBoard のディレクトリ")
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--render_steps", type=int, default=100)
    parser.add_argument("--window_size", type=int, default=512)
    parser.add_argument('--shader', choices=['soft_silhouette_shader', 'soft_phong_shader', 'textured_soft_phong_shader'], default="soft_silhouette_shader", help="shader の種類")
    parser.add_argument("--light_pos_x", type=float, default=0.0)
    parser.add_argument("--light_pos_y", type=float, default=0.0)
    parser.add_argument("--light_pos_z", type=float, default=-5.0)
    parser.add_argument("--camera_dist", type=float, default=2.7)
    parser.add_argument("--camera_elev", type=float, default=25.0)
    parser.add_argument("--camera_azim", type=float, default=150.0)

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

    #================================
    # データセットの読み込み
    #================================
    # SMPL の読み込み
    smpl = SMPLModel( registration_path = args.registration_path, device = device, debug = args.debug )
    betas = torch.from_numpy( (np.random.rand(args.batch_size, 10) - 0.5) * 0.06 ).float().requires_grad_(False).to(device)
    thetas = torch.from_numpy( (np.random.rand(args.batch_size, 72) - 0.5) * 0.06 ).float().requires_grad_(False).to(device)
    verts, faces, joints = smpl( betas, thetas )
    print( "betas : ", betas )
    print( "thetas : ", thetas )
    if( args.debug ):
        print( "verts.shape={}, faces.shape={}, joints.shape={}".format(verts.shape, faces.shape, joints.shape) )

    # 頂点と面情報から Mesh 生成 / メッシュ : pytorch3d.structures.meshes.Meshes 型
    mesh = Meshes(verts, faces).to(device)
    print( "mesh.num_verts_per_mesh() : ", mesh.num_verts_per_mesh() )

    # メッシュの描写
    save_plot3d_mesh_img( mesh, os.path.join(args.results_dir, args.exper_name, "mesh.png" ), "mesh" )
    save_mesh_obj( verts[0], faces[0], os.path.join(args.results_dir, args.exper_name, "mesh.obj" ) )

    # メッシュのテクスチャー / テクスチャー : Tensor 型
    if( args.shader == "textured_soft_phong_shader" ):
        texture = mesh.textures.maps_padded()
        save_image( texture.transpose(1,3).transpose(2,3), os.path.join(args.results_dir, args.exper_name, "texture.png" ) )
    elif( args.shader == "soft_phong_shader" ):
        from pytorch3d.structures import Textures
        
        # 頂点カラーを設定
        verts_rgb_colors = torch.ones([1, mesh.num_verts_per_mesh().item(), 3]).to(device) * 0.9

        # 頂点カラーのテクスチャー生成
        texture = Textures(verts_rgb=verts_rgb_colors)

        # メッシュにテクスチャーを設定
        mesh.textures = texture

    #================================
    # レンダリングパイプラインの構成
    #================================
    # ビュー変換行列の作成
    rot_matrix, trans_matrix = look_at_view_transform( 
        dist = args.camera_dist,     # distance of the camera from the object
        elev = args.camera_elev,     # angle in degres or radians. This is the angle between the vector from the object to the camera, and the horizontal plane y = 0 (xz-plane).
        azim = args.camera_azim      # angle in degrees or radians. The vector from the object to the camera is projected onto a horizontal plane y = 0. azim is the angle between the projected vector and a reference vector at (1, 0, 0) on the reference plane (the horizontal plane).
    )

    # カメラの作成
    cameras = OpenGLPerspectiveCameras( device = device, R = rot_matrix, T = trans_matrix )

    # ラスタライザーの作成
    rasterizer = MeshRasterizer(
        cameras = cameras, 
        raster_settings = RasterizationSettings(
            image_size = args.window_size, 
            blur_radius = 0.0, 
            faces_per_pixel = 1, 
            bin_size = None,            # this setting controls whether naive or coarse-to-fine rasterization is used
            max_faces_per_bin = None    # this setting is for coarse rasterization
        )
    )

    # ライトの作成
    lights = PointLights( device = device, location = [[args.light_pos_x, args.light_pos_y, args.light_pos_z]] )

    # マテリアルの作成
    materials = Materials(
        device = device,
        specular_color = [[0.5, 0.5, 0.5]],
        shininess = 10.0
    )

    # シェーダーの作成
    if( args.shader == "soft_silhouette_shader" ):
        shader = SoftSilhouetteShader()
    elif( args.shader == "soft_phong_shader" ):
        shader = SoftPhongShader( device = device, cameras = cameras, lights = lights, materials = materials )
    elif( args.shader == "textured_soft_phong_shader" ):
        shader = TexturedSoftPhongShader( device = device, cameras = cameras, lights = lights, materials = materials )
    else:
        NotImplementedError()

    # レンダラーの作成
    renderer = MeshRenderer( rasterizer = rasterizer, shader = shader )

    #================================
    # レンダリングループ処理
    #================================
    camera_dist = args.camera_dist
    camera_elev = args.camera_elev
    camera_azim = args.camera_azim
    for step in tqdm( range(args.render_steps), desc="render"):
        #-----------------------------
        # カメラの移動＆再レンダリング
        #-----------------------------
        camera_dist += 0.1
        camera_elev += 5.0
        camera_azim += 5.0
        rot_matrix, trans_matrix = look_at_view_transform( dist = camera_dist, elev = camera_elev, azim = camera_azim )
        new_cameras = OpenGLPerspectiveCameras( device = device, R = rot_matrix, T = trans_matrix )

        # メッシュのレンダリング
        mesh_img_tsr = renderer( mesh, cameras = new_cameras, lights = lights, materials = materials ).transpose(1,3).transpose(2,3)

        # alpha belend
        mesh_img_tsr[:,0,:,:] = mesh_img_tsr[:,0,:,:] * mesh_img_tsr[:,-1,:,:]
        mesh_img_tsr[:,1,:,:] = mesh_img_tsr[:,1,:,:] * mesh_img_tsr[:,-1,:,:]
        mesh_img_tsr[:,2,:,:] = mesh_img_tsr[:,2,:,:] * mesh_img_tsr[:,-1,:,:]
        mesh_img_tsr = mesh_img_tsr[:,0:3,:,:]
        mesh_img_tsr = torch.from_numpy( (mesh_img_tsr.detach().cpu().numpy() > 0).astype(np.float32) * 2.0 ).to(device) - 1.0
        if( args.debug and step == 0 ):
            print( "min={}, max={}".format(torch.min(mesh_img_tsr), torch.max(mesh_img_tsr) ) )

        save_image( mesh_img_tsr, os.path.join(args.results_dir, args.exper_name, "mesh_camera.png" ) )

        # visual images
        visuals = [
            [ mesh_img_tsr ],
        ]
        board_add_images(board_train, 'render_camera', visuals, step+1)

        #-------------------------------------------------
        # SMPL の人物形状パラメータ beta ＆ 再レンダリング
        #-------------------------------------------------
        new_betas = torch.from_numpy( (np.random.rand(args.batch_size, 10) - 0.5) * 0.06 ).float().requires_grad_(False).to(device)
        print( "new_betas : ", new_betas )
        verts, faces, joints = smpl( new_betas, thetas )
        mesh = Meshes(verts, faces)

        mesh_img_tsr = renderer( mesh, cameras = cameras, lights = lights, materials = materials ).transpose(1,3).transpose(2,3)
        mesh_img_tsr[:,0,:,:] = mesh_img_tsr[:,0,:,:] * mesh_img_tsr[:,-1,:,:]
        mesh_img_tsr[:,1,:,:] = mesh_img_tsr[:,1,:,:] * mesh_img_tsr[:,-1,:,:]
        mesh_img_tsr[:,2,:,:] = mesh_img_tsr[:,2,:,:] * mesh_img_tsr[:,-1,:,:]
        mesh_img_tsr = mesh_img_tsr[:,0:3,:,:]
        mesh_img_tsr = torch.from_numpy( (mesh_img_tsr.detach().cpu().numpy() > 0).astype(np.float32) * 2.0 ).to(device) - 1.0
        save_image( mesh_img_tsr, os.path.join(args.results_dir, args.exper_name, "mesh_beta.png" ) )
        save_mesh_obj( verts[0], faces[0], os.path.join(args.results_dir, args.exper_name, "mesh_beta_step{}.obj".format(step) ) )

        # visual images
        visuals = [
            [ mesh_img_tsr ],
        ]
        board_add_images(board_train, 'render_beta', visuals, step+1)

        #-----------------------------
        # SMPL 制御パラメータの変更＆再レンダリング
        #-----------------------------
        new_thetas = torch.from_numpy( (np.random.rand(args.batch_size, 72) - 0.5) * 0.06 ).float().requires_grad_(False).to(device)
        print( "new_thetas : ", new_thetas )
        verts, faces, joints = smpl( betas, new_thetas )
        mesh = Meshes(verts, faces)

        mesh_img_tsr = renderer( mesh, cameras = cameras, lights = lights, materials = materials ).transpose(1,3).transpose(2,3)
        mesh_img_tsr[:,0,:,:] = mesh_img_tsr[:,0,:,:] * mesh_img_tsr[:,-1,:,:]
        mesh_img_tsr[:,1,:,:] = mesh_img_tsr[:,1,:,:] * mesh_img_tsr[:,-1,:,:]
        mesh_img_tsr[:,2,:,:] = mesh_img_tsr[:,2,:,:] * mesh_img_tsr[:,-1,:,:]
        mesh_img_tsr = mesh_img_tsr[:,0:3,:,:]
        mesh_img_tsr = torch.from_numpy( (mesh_img_tsr.detach().cpu().numpy() > 0).astype(np.float32) * 2.0 ).to(device) - 1.0
        save_image( mesh_img_tsr, os.path.join(args.results_dir, args.exper_name, "mesh_theta.png" ) )
        save_mesh_obj( verts[0], faces[0], os.path.join(args.results_dir, args.exper_name, "mesh_theta_step{}.obj".format(step) ) )

        # visual images
        visuals = [
            [ mesh_img_tsr ],
        ]
        board_add_images(board_train, 'render_theta', visuals, step+1)
