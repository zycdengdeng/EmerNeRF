# nuScenes 静态场景 1 秒 clip · SfM + MVS 重建数据集

> 50 个独立 clip = 5 场景 × 10 个 1 秒片段。每片段 = 6 相机 × 10 帧。已跑完 COLMAP **known-pose triangulation**（稀疏 SfM）+ **patch_match_stereo + stereo_fusion**（稠密 MVS）。

---

## 1. 数据路径速查

| 用途 | 绝对路径 |
|---|---|
| **原始 nuScenes** | `/mnt/public_datasets/nuscenes_10Hz/trainval/<scene>/` |
| **每 clip 完整产物（含 colmap workspace、symlink images）** | `/mnt/zihanw/gsnet_nusc/<scene>_clip_<NN>/` |
| **平铺的 PLY（直接给 CloudCompare 看）** | `/mnt/zihanw/nusc_points/<scene>_clip_<NN>_{sparse,dense}.ply` |
| **传感器 rig 可视化** | `/mnt/zihanw/gsnet_nusc/viz/sensor_nuscenes.png` |
| **场景挑选元数据** | `/mnt/zihanw/gsnet_nusc/selected_scenes.json` |

---

## 2. 5 个被选中的场景

| scene | n_dyn | 轨迹长 (m) | 速度 cv | 备注 |
|---|---|---|---|---|
| 348 | 1 | 120 | 0.18 | 最干净 |
| 332 | 3 | 140 | 0.15 | |
| 331 | 3 | 130 | 0.18 | |
| 299 | 4 | 81  | 0.26 | 手动改选（白天） |
| 325 | 3 | 160 | 0.17 | 手动改选（白天） |

> Selector 默认排名前两名是 797 和 830，但都是夜间场景；改为人工挑选 299 和 325 这两个白天场景。
> n_dyn = 在 ~20s 内位移 > 1m 的车辆 / 行人实例数（位移 < 1m 的视为停车）。

---

## 3. 每个 clip 的目录结构

完整 workspace `/mnt/zihanw/gsnet_nusc/<scene>_clip_<NN>/`：

```
<scene>_clip_<NN>/
├── images/                     # 60 张图，按相机分子目录
│   ├── cam0/000.jpg ~ 009.jpg  # FRONT
│   ├── cam1/                   # FRONT_LEFT
│   ├── cam2/                   # FRONT_RIGHT
│   ├── cam3/                   # BACK_LEFT
│   ├── cam4/                   # BACK_RIGHT
│   └── cam5/                   # BACK
│       └── 图像本身是 symlink → /mnt/public_datasets/nuscenes_10Hz/trainval/...
├── selection_meta.json         # ⭐ 相机内参 + 每张图的 world→cam 位姿（4×4）
├── sfm_attempt1.log            # SfM 日志
├── mvs.log                     # MVS 日志
└── colmap/
    ├── database.db             # COLMAP 特征/匹配数据库
    ├── sparse_in/              # 喂给 point_triangulator 的输入 txt
    ├── sparse/
    │   └── 0/
    │       ├── cameras.bin     # 6 相机 PINHOLE 内参（fx, fy, cx, cy）
    │       ├── images.bin      # 60 张图的位姿（world→cam）+ 2D 特征
    │       └── points3D.bin    # ⭐ 稀疏 SfM 3D 点（带 RGB + track）
    └── dense/                  # image_undistorter + MVS 产物
        ├── images/             # PINHOLE undistorted（这里其实是 copy，因为已是 PINHOLE）
        ├── sparse/0/           # 同上但 PINHOLE 化的版本（3DGS 直接读这里）
        ├── stereo/             # patch_match 中间产物
        │   ├── depth_maps/     # 120 个 .bin（光度 + 几何）
        │   ├── normal_maps/
        │   └── consistency_graphs/
        └── fused.ply           # ⭐ MVS 稠密点云（带 RGB + 法线）
```

平铺版 `/mnt/zihanw/nusc_points/`（每个 clip 2 个文件，方便挑/下载）：

```
<scene>_clip_<NN>_sparse.ply    # 由 points3D.bin 转出，几千–几万点
<scene>_clip_<NN>_dense.ply     # = fused.ply 拷贝，几十–几百万点
```

---

## 4. 坐标系与位姿约定（**任何后续处理都得先看这一段**）

### 4.1 相机 ID ↔ 物理相机

| camID | 名称 | nuScenes 原名 |
|---|---|---|
| 0 | FRONT | CAM_FRONT |
| 1 | FRONT_LEFT | CAM_FRONT_LEFT |
| 2 | FRONT_RIGHT | CAM_FRONT_RIGHT |
| 3 | BACK_LEFT | CAM_BACK_LEFT |
| 4 | BACK_RIGHT | CAM_BACK_RIGHT |
| 5 | BACK | CAM_BACK |

### 4.2 相机内参

- 模型：**PINHOLE**（4 参数：fx, fy, cx, cy）
- 图像分辨率：1600 × 900
- **已去畸变**（原始 nuscenes_10Hz 保存的图像就是 undistorted）
- cam0–4 焦距 ≈ 1260 像素；cam5 (BACK) 焦距 ≈ 809 像素（视场更广）

### 4.3 相机坐标系

- **OpenCV 约定**：x 右、y 下、z 前（光轴 = +z）
- 和 KITTI / COLMAP / Open3D 一致；和 Waymo protobuf 原生 FLU 不同

### 4.4 位姿约定

- **`images.bin` 中的位姿是 world → camera_OpenCV**（也就是 COLMAP 标准格式）
- **`selection_meta.json` 里 `world_to_camcv` 字段同样是 world → camera_OpenCV**
- 原始 nuScenes 文件 `extrinsics/<frame>_<cam>.txt` 是 **camera → world**（cam2world），上面那两份是它的逆

### 4.5 世界系

- 所有 3D 点（sparse + dense）都在 **nuScenes global map frame**（米制）
- 这不是以车为中心的局部系。**不同 scene 的点云在世界系里位置相距很远**（不同 trip），所以不能直接合并所有 50 个 clip 的点云
- 同一个 scene 内的 10 个 clip 在世界系里是**连续衔接**的（车沿轨迹走过 10 秒）
- 自车在世界系里的位置：见 `/mnt/public_datasets/nuscenes_10Hz/trainval/<scene>/lidar_pose/<frame>.txt`（4×4 ego→world）

---

## 5. selection_meta.json schema

```json
{
  "scene": "348",
  "clip_global_frames": [0, 1, 2, ..., 9],
  "fps": 10,
  "extrinsic_dir": "cam2world",       # 原始数据的方向（仅记录用）
  "cameras": [
    {
      "cam_id": 0,
      "label": "FRONT",
      "width": 1600,
      "height": 900,
      "intrinsic_pinhole": [fx, fy, cx, cy]
    },
    ...
  ],
  "images": [
    {
      "cam_id": 0,
      "local_idx": 0,
      "filename": "cam0/000.jpg",
      "global_frame": 0,
      "world_to_camcv": [[r11, r12, r13, t1], ...]   # 4x4
    },
    ...   # 60 张
  ]
}
```

---

## 6. 快速使用片段

### 6.1 用 CloudCompare / MeshLab 看点云

直接拖 `.ply` 进去即可，不需要任何转换。
- `*_dense.ply`：稠密 MVS 点（带 RGB + 法线）
- `*_sparse.ply`：稀疏 SfM 点（带 RGB）

### 6.2 Python 读 sparse 3D 点 + 位姿

```python
# pip install pycolmap
import pycolmap
rec = pycolmap.Reconstruction("/mnt/zihanw/gsnet_nusc/348_clip_00/colmap/dense/sparse/0")
print(len(rec.points3D), "sparse points")
for img_id, img in rec.images.items():
    # img.qvec, img.tvec 是 world->cam
    R = pycolmap.qvec2rotmat(img.qvec)
    t = img.tvec
    cam_in_world = -R.T @ t           # 相机光学中心在世界系
    print(img.name, "->", cam_in_world)
```

### 6.3 Python 读 dense fused.ply

```python
import open3d as o3d
pcd = o3d.io.read_point_cloud("/mnt/zihanw/nusc_points/348_clip_00_dense.ply")
print(pcd)   # 例：PointCloud with 350000 points
```

### 6.4 喂给 3DGS（Inria gaussian-splatting）

每个 clip 目录已经是 3DGS-ready：
```bash
python gaussian-splatting/train.py \
    -s /mnt/zihanw/gsnet_nusc/348_clip_00 \
    -m output/348_clip_00
```
3DGS 会读 `colmap/dense/sparse/0/{cameras,images,points3D}.bin` + `colmap/dense/images/`。

要用稠密点云做初始化：把 `fused.ply` 拷成 `colmap/dense/sparse/0/points3D.ply`（3DGS 优先读 .ply）。

---

## 7. 已知问题 / 注意

1. **动态物体的"鬼影"**：5 个 scene 都不是完全静态的（n_dyn 1–4），稠密点云里可能能看到移动车辆形成的飘点/双层。SfM 是 known-pose triangulation，自身不受影响；MVS 的 patch_match 在动态物体上会失败或产生噪点。
2. **天空噪点**：stereo_fusion 在天空区域常产生远处飘点（无穷远纹理匹配不稳）。可以加 sky mask 缓解，目前没加。
3. **跨 clip 不能直接合点云**：同 scene 内 10 个 clip 在世界系里相邻；不同 scene 之间相距甚远，合一起没意义。
4. **没动 distortion**：nuscenes_10Hz 保存的图像已经去过畸变，我们的 cameras.bin 直接用 PINHOLE 模型，没做二次 undistort。

---

## 8. 重建参数（备查）

- COLMAP 3.11.1 (CUDA-SIFT)
- SfM: `feature_extractor`（SIFT GPU）+ `exhaustive_matcher` + DB camera 内参 patch + `point_triangulator`（已知 world→cam 位姿）
- Undistort: `image_undistorter --output_type COLMAP`
- MVS: `patch_match_stereo --PatchMatchStereo.geom_consistency true --max_image_size 1600` + `stereo_fusion --input_type geometric`

工具脚本：`tools/nusc_prep/` in [EmerNeRF repo branch claude/confident-cray-MFupr](https://github.com/zycdengdeng/EmerNeRF/tree/claude/confident-cray-MFupr)。
