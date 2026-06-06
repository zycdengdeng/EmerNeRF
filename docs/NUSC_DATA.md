# nuScenes 静态场景 1 秒 clip · SfM + MVS 重建数据集

## 概览

- **5 个 scene**：348, 332, 331, 299, 325
- **每个 scene 切 10 个 1 秒 clip**，共 **50 个 clip**
- **每 clip 60 张图** = 6 相机 × 10 帧 @ 10 Hz

## 路径

| 内容 | 路径 |
|---|---|
| 完整 workspace（每 clip 一个目录） | `/mnt/zihanw/gsnet_nusc/<scene>_clip_<NN>/` |
| 场景挑选元数据 | `/mnt/zihanw/gsnet_nusc/selected_scenes.json` |
| 传感器 rig 可视化 | `/mnt/zihanw/gsnet_nusc/viz/sensor_nuscenes.png` |
| 原始 nuScenes | `/mnt/public_datasets/nuscenes_10Hz/trainval/<scene>/` |

## 单 clip 目录结构

```
/mnt/zihanw/gsnet_nusc/<scene>_clip_<NN>/
├── images/                          # 60 张图，分相机子目录
│   ├── cam0/000.jpg ~ 009.jpg       # FRONT
│   ├── cam1/                        # FRONT_LEFT
│   ├── cam2/                        # FRONT_RIGHT
│   ├── cam3/                        # BACK_LEFT
│   ├── cam4/                        # BACK_RIGHT
│   └── cam5/                        # BACK
├── selection_meta.json              # 相机内参 + 每张图 world→cam 4×4 位姿
└── colmap/
    ├── database.db
    ├── sparse/
    │   └── 0/
    │       ├── cameras.bin          # 6 相机 PINHOLE (fx, fy, cx, cy)
    │       ├── images.bin           # 60 张图位姿（world→cam）
    │       └── points3D.bin         # SfM 稀疏点云
    └── dense/
        ├── images/                  # PINHOLE undistorted
        ├── sparse/0/                # PINHOLE 化的 sparse（3DGS 读这里）
        ├── stereo/                  # patch_match 中间产物
        └── fused.ply                # MVS 稠密点云（带 RGB + 法线）
```

## 相机 ID 映射

| camID | 物理相机 | nuScenes 原名 |
|---|---|---|
| 0 | FRONT | CAM_FRONT |
| 1 | FRONT_LEFT | CAM_FRONT_LEFT |
| 2 | FRONT_RIGHT | CAM_FRONT_RIGHT |
| 3 | BACK_LEFT | CAM_BACK_LEFT |
| 4 | BACK_RIGHT | CAM_BACK_RIGHT |
| 5 | BACK | CAM_BACK |

## 坐标系 / 位姿约定

| 项目 | 约定 |
|---|---|
| 相机模型 | PINHOLE（fx, fy, cx, cy），1600 × 900，**已去畸变** |
| 相机帧 | OpenCV：x 右、y 下、z 前 |
| 位姿方向 | `images.bin` / `selection_meta.json.world_to_camcv` 都是 **world → cam** |
| 世界系 | nuScenes global map frame（米制） |
| 自车轨迹 | 见原始 `lidar_pose/<frame>.txt`（4×4 ego→world） |

> 不同 scene 的点云在世界系里位置相距很远，不能直接合并；同 scene 内 10 个 clip 在世界系中是连续的。

## selection_meta.json schema

```json
{
  "scene": "348",
  "clip_global_frames": [0, 1, ..., 9],
  "fps": 10,
  "cameras": [
    {"cam_id": 0, "label": "FRONT", "width": 1600, "height": 900,
     "intrinsic_pinhole": [fx, fy, cx, cy]},
    ...
  ],
  "images": [
    {"cam_id": 0, "local_idx": 0, "filename": "cam0/000.jpg",
     "global_frame": 0,
     "world_to_camcv": [[r11, r12, r13, t1], ...]},
    ...
  ]
}
```
