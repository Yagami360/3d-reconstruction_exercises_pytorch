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
from pytorch3d.structures import Meshes
from pytorch3d.renderer import look_at_view_transform, OpenGLPerspectiveCameras         # カメラ関連
from pytorch3d.renderer import PointLights, DirectionalLights                           # ライト関連
from pytorch3d.renderer import Materials                                                # マテリアル関連
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer                    # ラスタライザー関連
from pytorch3d.renderer.mesh.shader import TexturedSoftPhongShader                      # シェーダー関連
from pytorch3d.renderer import MeshRenderer                                             # レンダラー関連

# 自作モジュール
from utils.utils import save_checkpoint, load_checkpoint
from utils.utils import board_add_image, board_add_images, save_image_w_norm, save_plot3d_mesh_img, get_plot3d_mesh_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="render_mesh", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="dataset/cow_mesh")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument('--save_checkpoints_dir', type=str, default="checkpoints", help="モデルの保存ディレクトリ")
    parser.add_argument('--load_checkpoints_path', type=str, default="", help="モデルの読み込みファイルのパス")
    parser.add_argument('--tensorboard_dir', type=str, default="tensorboard", help="TensorBoard のディレクトリ")

    parser.add_argument("--render_steps", type=int, default=100)
    parser.add_argument("--window_size", type=int, default=512)
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
    # メッシュファイルの読み込み / メッシュ : pytorch3d.structures.meshes.Meshes 型
    mesh = load_objs_as_meshes( [os.path.join(args.dataset_dir, 'cow.obj')], device = device )
    print( "mesh : ", mesh )                    # <pytorch3d.structures.meshes.Meshes object at 0x13dc98da0>

    # メッシュの描写
    save_plot3d_mesh_img( mesh, os.path.join(args.results_dir, args.exper_name, "mesh.png" ), "mesh" )

    # メッシュのテクスチャー / テクスチャー : Tensor 型
    texture = mesh.textures.maps_padded()
    #print( "texture : ", texture )             # tensor([[[[1.0000, 0.9333, 0.9020], ...
    print( "texture.shape : ", texture.shape )  # torch.Size([1, 1024, 1024, 3])

    # テクスチャーの描写
    save_image( texture.transpose(1,3).transpose(2,3), os.path.join(args.results_dir, args.exper_name, "texture.png" ) )

    #================================
    # レンダリングパイプラインの構成
    #================================
    # ビュー変換行列の作成
    rot_matrix, trans_matrix = look_at_view_transform( 
        dist = args.camera_dist,     # distance of the camera from the object
        elev = args.camera_elev,     # angle in degres or radians. This is the angle between the vector from the object to the camera, and the horizontal plane y = 0 (xz-plane).
        azim = args.camera_azim      # angle in degrees or radians. The vector from the object to the camera is projected onto a horizontal plane y = 0. azim is the angle between the projected vector and a reference vector at (1, 0, 0) on the reference plane (the horizontal plane).
    )
    print( "rot_matrix.shape : ", rot_matrix.shape )    # tensor / torch.Size([1, 3, 3])
    print( "trans_matrix.shape : ", trans_matrix.shape )    # tensor / torch.Size([1, 3])

    # カメラの作成
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
    cameras = OpenGLPerspectiveCameras( device = device, R = rot_matrix, T = trans_matrix )

    # ラスタライザーの作成
    # Define the settings for rasterization and shading. Here we set the output image to be of size 512x512.
    # As we are rendering images for visualization purposes only we will set faces_per_pixel=1 and blur_radius=0.0.
    # We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used.
    # Refer to rasterize_meshes.py for explanations of these parameters. 
    # Refer to docs/notes/renderer.md for an explanation of the difference between naive and coarse-to-fine rasterization. 
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
        specular_color = [[0.0, 0.0, 0.0]],
        shininess = 10.0
    )

    # シェーダーの作成
    # The textured phong shader will interpolate the texture uv coordinates for each vertex, sample from a texture image and apply the Phong lighting model
    shader = TexturedSoftPhongShader( device = device, cameras = cameras, lights = lights, materials = materials )

    # レンダラーの作成
    # Create a phong renderer by composing a rasterizer and a shader.
    renderer = MeshRenderer( rasterizer = rasterizer, shader = shader )

    # メッシュのレンダリング
    #mesh_img_tsr = renderer(mesh, cameras = cameras, lights = lights)
    #save_image( mesh_img_tsr.transpose(1,3).transpose(2,3), os.path.join(args.results_dir, args.exper_name, "mesh_render.png" ) )

    #================================
    # レンダリングループ処理
    #================================
    new_lights = lights.clone()
    light_pos_x = args.light_pos_x
    light_pos_y = args.light_pos_y
    light_pos_z = args.light_pos_z
    camera_dist = args.camera_dist
    camera_elev = args.camera_elev
    camera_azim = args.camera_azim
    material_spec_r = 0.0
    material_spec_g = 0.0
    material_spec_b = 0.0
    shininess = 10.0
    for step in tqdm( range(args.render_steps), desc="render"):
        #-----------------------------
        # ライトの移動＆再レンダリング
        #-----------------------------
        light_pos_x += 0.0
        light_pos_y += 0.0
        light_pos_z += 0.5
        new_lights.location = torch.tensor( [light_pos_x, light_pos_y, light_pos_z], device = device )[None]

        # メッシュのレンダリング
        mesh_img_tsr = renderer( mesh, cameras = cameras, lights = new_lights, materials = materials )
        save_image( mesh_img_tsr.transpose(1,3).transpose(2,3), os.path.join(args.results_dir, args.exper_name, "mesh_render_light.png" ) )

        # visual images
        visuals = [
            [ mesh_img_tsr.transpose(1,3).transpose(2,3) ],
        ]
        board_add_images(board_train, 'render_light', visuals, step+1)

        #-----------------------------
        # カメラの移動＆再レンダリング
        #-----------------------------
        camera_dist += 0.1
        camera_elev += 5.0
        camera_azim += 5.0
        rot_matrix, trans_matrix = look_at_view_transform( dist = camera_dist, elev = camera_elev, azim = camera_azim )
        new_cameras = OpenGLPerspectiveCameras( device = device, R = rot_matrix, T = trans_matrix )

        # メッシュのレンダリング
        mesh_img_tsr = renderer( mesh, cameras = new_cameras, lights = lights, materials = materials )
        save_image( mesh_img_tsr.transpose(1,3).transpose(2,3), os.path.join(args.results_dir, args.exper_name, "mesh_render_camera.png" ) )

        # visual images
        visuals = [
            [ mesh_img_tsr.transpose(1,3).transpose(2,3) ],
        ]
        board_add_images(board_train, 'render_camera', visuals, step+1)

        #-----------------------------
        # マテリアル変更＆再レンダリング
        #-----------------------------
        material_spec_r += 0.1
        material_spec_g += 0.1
        material_spec_b += 0.1
        shininess += 1.0
        new_materials = Materials(
            device = device,
            specular_color = [[material_spec_r, material_spec_g, material_spec_b]],
            shininess = shininess
        )

        # メッシュのレンダリング
        mesh_img_tsr = renderer( mesh, cameras = new_cameras, lights = lights, materials = new_materials )
        save_image( mesh_img_tsr.transpose(1,3).transpose(2,3), os.path.join(args.results_dir, args.exper_name, "mesh_render_materials.png" ) )

        # visual images
        visuals = [
            [ mesh_img_tsr.transpose(1,3).transpose(2,3) ],
        ]
        board_add_images(board_train, 'render_materials', visuals, step+1)



