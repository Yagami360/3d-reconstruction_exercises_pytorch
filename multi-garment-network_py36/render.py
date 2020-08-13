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

# mesh
from psbody.mesh import Mesh

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
from data.smpl import SMPLModel
from data.smpl_mgn import SMPLMGNModel
from utils.utils import save_checkpoint, load_checkpoint
from utils.utils import board_add_image, board_add_images, save_image_w_norm, save_plot3d_mesh_img, get_plot3d_mesh_img, save_mesh_obj
from utils.mesh import deform_mesh_by_closest_vertices

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="smpl_g+", help="実験名")
    parser.add_argument("--smpl_registration_path", type=str, default="datasets/smpl_registrations/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl")
    parser.add_argument("--digital_wardrobe_registration_path", type=str, default="datasets/digital_wardrobe/Multi-Garment_dataset/125611508622317/registration.pkl")
    parser.add_argument("--digital_wardrobe_cloth_mesh_path", type=str, default="datasets/digital_wardrobe/Multi-Garment_dataset/125611508622317/TShirtNoCoat.obj")
    parser.add_argument("--cloth_smpl_fts_path", type=str, default="datasets/assets/garment_fts.pkl")
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
    smpl = SMPLMGNModel( 
        registration_path = args.smpl_registration_path, 
        digital_wardrobe_registration_path = args.digital_wardrobe_registration_path, 
        cloth_smpl_fts_path = args.cloth_smpl_fts_path,
        batch_size = args.batch_size, device = device, debug = args.debug
    )

    # SMPL を用いた初期Tポーズでの人体メッシュ
    betas_init = torch.zeros( (args.batch_size, 10), requires_grad=False).float().to(device)
    thetas_init = torch.zeros( (args.batch_size, 72), requires_grad=False).float().to(device)
    trans_init = torch.from_numpy(np.zeros((args.batch_size, 3))).float().requires_grad_(False).to(device)
    verts_init, faces_init, joints_init = smpl( betas = betas_init, thetas = thetas_init, trans = trans_init )
    mesh_body_init = Meshes(verts_init, faces_init).to(device)
    print( "mesh_body_init.num_verts_per_mesh() : ", mesh_body_init.num_verts_per_mesh() )
    print( "mesh_body_init.num_faces_per_mesh() : ", mesh_body_init.num_faces_per_mesh() )
    if( args.shader == "soft_phong_shader" ):
        # メッシュのテクスチャー設定
        from pytorch3d.structures import Textures
        verts_rgb_colors = torch.ones([1, mesh_body.num_verts_per_mesh().item(), 3]).to(device) * 0.9
        texture = Textures(verts_rgb=verts_rgb_colors)
        mesh_body.textures = texture

    save_mesh_obj( verts_init[0], faces_init[0], os.path.join(args.results_dir, args.exper_name, "mesh_body_init.obj" ) )

    # SMPL を用いた制御パラメータ β,θ を変えた場合の人体メッシュ
    betas = torch.from_numpy( (np.random.rand(args.batch_size, 10) - 0.5) * 0.06 ).float().requires_grad_(False).to(device)
    thetas = torch.from_numpy( (np.random.rand(args.batch_size, 72) - 0.5) * 0.06 ).float().requires_grad_(False).to(device)
    verts, faces, joints = smpl( betas, thetas )
    mesh_body = Meshes(verts, faces).to(device)
    print( "mesh_body.num_verts_per_mesh() : ", mesh_body.num_verts_per_mesh() )
    print( "mesh_body.num_faces_per_mesh() : ", mesh_body.num_faces_per_mesh() )
    if( args.shader == "soft_phong_shader" ):
        from pytorch3d.structures import Textures
        verts_rgb_colors = torch.ones([1, mesh_body_init.num_verts_per_mesh().item(), 3]).to(device) * 0.9   # 頂点カラーを設定
        texture = Textures(verts_rgb=verts_rgb_colors)                                                       # 頂点カラーのテクスチャー生成
        mesh_body_init.textures = texture                                                                    # メッシュにテクスチャーを設定

    save_mesh_obj( verts[0], faces[0], os.path.join(args.results_dir, args.exper_name, "mesh_body.obj" ) )
    if( args.debug ):
        print( "betas : ", betas )
        print( "thetas : ", thetas )
        print( "verts.shape={}, faces.shape={}, joints.shape={}".format(verts.shape, faces.shape, joints.shape) )

    # digital wardrobe にある服テンプレートメッシュを読み込み
    mesh_cloth = load_objs_as_meshes( [args.digital_wardrobe_cloth_mesh_path], device = device )
    print( "mesh_cloth.num_verts_per_mesh() : ", mesh_cloth.num_verts_per_mesh() )
    print( "mesh_cloth.num_faces_per_mesh() : ", mesh_cloth.num_faces_per_mesh() )
    save_mesh_obj( mesh_cloth.verts_packed(), mesh_cloth.faces_packed(), os.path.join(args.results_dir, args.exper_name, "mesh_cloth.obj" ) )

    # 服テンプレートメッシュを 制御パラメータ β,θ を変えた場合の人体メッシュの形状に変形する / Re-target
    mesh_cloth = deform_mesh_by_closest_vertices( mesh_cloth, mesh_body_init, mesh_body )
    save_mesh_obj( mesh_cloth.verts_packed(), mesh_cloth.faces_packed(), os.path.join(args.results_dir, args.exper_name, "mesh_cloth_deformed.obj" ) )

    # 衣装テンプレートメッシュの対応する SMPL 体型メッシュへの頂点変形 D を論文中の (3) 式で計算し、服メッシュを変形する？ / Re-pose
    pass

    # Laplacian deformation による衣装テンプレートメッシュの変形
    pass

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
        mesh_img_tsr = renderer( mesh_body, cameras = new_cameras, lights = lights, materials = materials )            
        mesh_img_tsr = mesh_img_tsr * 2.0 - 1.0
        if( args.debug and step == 0 ):
            print( "min={}, max={}".format(torch.min(mesh_img_tsr), torch.max(mesh_img_tsr) ) )

        save_image( mesh_img_tsr.transpose(1,3).transpose(2,3), os.path.join(args.results_dir, args.exper_name, "mesh_camera.png" ) )

        # visual images
        visuals = [
            [ mesh_img_tsr.transpose(1,3).transpose(2,3) ],
        ]
        board_add_images(board_train, 'render_camera', visuals, step+1)

        #-------------------------------------------------
        # SMPL の人物形状パラメータ beta 変更 ＆ 再レンダリング
        #-------------------------------------------------
        new_betas = torch.from_numpy( (np.random.rand(args.batch_size, 10) - 0.5) * 0.06 ).float().requires_grad_(False).to(device)
        print( "new_betas : ", new_betas )
        verts, faces, joints = smpl( new_betas, thetas )
        mesh_body = Meshes(verts, faces)
        mesh_body_img_tsr = renderer( mesh_body, cameras = cameras, lights = lights, materials = materials ) * 2.0 - 1.0
        save_image( mesh_body_img_tsr.transpose(1,3).transpose(2,3), os.path.join(args.results_dir, args.exper_name, "mesh_beta_step{}.png".format(step) ) )
        save_mesh_obj( verts[0], faces[0], os.path.join(args.results_dir, args.exper_name, "mesh_beta_step{}.obj".format(step) ) )

        # visual images
        mesh_body_init_img_tsr = renderer( mesh_body_init, cameras = cameras, lights = lights, materials = materials ) * 2.0 - 1.0
        visuals = [
            [ mesh_body_init_img_tsr.transpose(1,3).transpose(2,3), mesh_body_img_tsr.transpose(1,3).transpose(2,3) ],
        ]
        board_add_images(board_train, 'render_beta', visuals, step+1)

        #-----------------------------
        # SMPL 制御パラメータ theta 変更＆再レンダリング
        #-----------------------------
        new_thetas = torch.from_numpy( (np.random.rand(args.batch_size, 72) - 0.5) * 0.06 ).float().requires_grad_(False).to(device)
        print( "new_thetas : ", new_thetas )
        verts, faces, joints = smpl( betas, new_thetas )
        mesh_body = Meshes(verts, faces)

        mesh_body_img_tsr = renderer( mesh_body, cameras = cameras, lights = lights, materials = materials ) * 2.0 - 1.0
        save_image( mesh_body_img_tsr.transpose(1,3).transpose(2,3), os.path.join(args.results_dir, args.exper_name, "mesh_theta_step{}.png".format(step) ) )
        save_mesh_obj( verts[0], faces[0], os.path.join(args.results_dir, args.exper_name, "mesh_theta_step{}.obj".format(step) ) )

        # visual images
        mesh_body_init_img_tsr = renderer( mesh_body_init, cameras = cameras, lights = lights, materials = materials ) * 2.0 - 1.0
        visuals = [
            [ mesh_body_init_img_tsr.transpose(1,3).transpose(2,3), mesh_body_img_tsr.transpose(1,3).transpose(2,3) ],
        ]
        board_add_images(board_train, 'render_theta', visuals, step+1)
