# multi-garment-network_py36
論文「Multi-Garment Net: Learning to Dress 3D People from Images」の実装
推論コードのみ

- 参考コード
    - https://github.com/bharat-b7/MultiGarmentNetwork

- 論文まとめ記事
    - https://github.com/Yagami360/MachineLearning-Papers_Survey/issues/87

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
         + basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
    ```

1. MGN データセットと事前学習済みモデルのダウンロード
    - https://github.com/bharat-b7/MultiGarmentNetwork#dress-smpl-body-model-with-our-digital-wardrobe を参考にデータをダウンロード
    - 以下のデータ構造で配置
        ```
        datasets
        +--- digital_wardrobe/
             +--- Multi-Garment_dataset
             +--- Multi-Garment_dataset_02
        +--- assets
        ```

1. スクリプトを実行
    1. 女性 の場合
        ```sh
        python render.py \
            --exper_name ${EXPER_NAME} \
            --smpl_registration_path datasets/smpl_registrations/basicModel_f_lbs_10_207_0_v1.0.0.pkl \
            --shader soft_silhouette_shader \
        ```

    1. 男性の場合
        ```sh
        python render.py \
            --exper_name ${EXPER_NAME} \
            --smpl_registration_path datasets/smpl_registrations/basicModel_m_lbs_10_207_0_v1.0.0.pkl \
            --shader soft_silhouette_shader \
        ```
    1. 中性の場合
        ```sh
        python render.py \
            --exper_name ${EXPER_NAME} \
            --smpl_registration_path datasets/smpl_registrations/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl \
            --shader soft_silhouette_shader \
        ```