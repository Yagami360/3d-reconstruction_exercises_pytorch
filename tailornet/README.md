# tailornet
論文「TailorNet: Predicting Clothing in 3D as a Function of Human Pose, Shape and Garment Style」の実装<br>
推論コードのみ

- 参考コード
    - https://github.com/chaitanya100100/TailorNet

- 論文まとめ記事
    - https://github.com/Yagami360/MachineLearning-Papers_Survey/issues/91

## 動作環境
- Python : 3.6
- pytorch : 1.5
- [pytorch3d](https://github.com/facebookresearch/pytorch3d) : 
    - fvcore
- [psbody.mesh](https://github.com/MPI-IS/mesh)
    - boost
- chumpy
- [opendr](https://github.com/polmorenoc/opendr)
- tqdm
- tensorboardx
- Pillow : < 7.0.0
- OpenCV

## 使用法

1. SMPL データセットのダウンロード
    - https://smpl.is.tue.mpg.de/downloads より「SMPL for Python Users」のデータをダウンロード
    - SMPL の registration files `basicModel_f_lbs_10_207_0_v1.0.0.pkl`, `basicmodel_m_lbs_10_207_0_v1.0.0.pkl` を以下のデータ構造で配置
    ```sh
    datasets/
    +--- smpl_registrations/
         + basicModel_f_lbs_10_207_0_v1.0.0.pkl
         + basicmodel_m_lbs_10_207_0_v1.0.0.pkl
    ```

1. TailorNet データセットと事前学習済みモデルのダウンロード
    ```sh
    $ sh download_datasets.sh
    ```

1. スクリプトを実行
    1. 女性 + old-t-shirt の場合
        ```sh
        python render.py \
            --exper_name ${EXPER_NAME} \
            --cloth_type old-t-shirt --gender female \
            --smpl_registration_dir datasets/smpl_registrations \
            --tailornet_dataset_dir datasets/tailornet_dataset \
            --kernel_sigma 0.01 \
            --shader soft_silhouette_shader
        ```

    1. 女性 + t-shirt の場合
        ```sh
        python render.py \
            --exper_name ${EXPER_NAME} \
            --cloth_type t-shirt --gender female \
            --smpl_registration_dir datasets/smpl_registrations \
            --tailornet_dataset_dir datasets/tailornet_dataset \
            --kernel_sigma 0.01 \
            --shader soft_silhouette_shader
        ```