# smpl
論文「SMPL: A skinned multi-person linear model」の python 3.6 + pytorch での実装<br>
推論コードのみ

- 参考コード
    - https://smpl.is.tue.mpg.de/downloads

- 論文まとめ記事
    - https://github.com/Yagami360/MachineLearning-Papers_Survey/issues/86

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

1. データセットのダウンロード
    - https://smpl.is.tue.mpg.de/downloads より「SMPL for Python Users」のデータをダウンロード

1. `basicModel_f_lbs_10_207_0_v1.0.0.pkl`, `basicmodel_m_lbs_10_207_0_v1.0.0.pkl` を以下のデータ構造で配置
    ```
    datasets/
    +---- registrations/
          + basicModel_f_lbs_10_207_0_v1.0.0.pkl
          + basicmodel_m_lbs_10_207_0_v1.0.0.pkl
    ```

1. スクリプトを実行
    1. 男性の場合
        ```
        python render.py \
        --exper_name ${EXPER_NAME} \
        --registration_path datasets/registrations/basicmodel_m_lbs_10_207_0_v1.0.0.pkl \
        --shader soft_silhouette_shader
        ```

    1. 女性の場合
        ```
        python render.py \
        --exper_name ${EXPER_NAME} \
        --registration_path datasets/registrations/basicmodel_f_lbs_10_207_0_v1.0.0.pkl \
        --shader soft_silhouette_shader
        ```
