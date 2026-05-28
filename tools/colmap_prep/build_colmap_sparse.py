#!/usr/bin/env python3
"""Build a COLMAP sparse model for one Waymo scene window using known poses.

Pipeline:
  1. colmap feature_extractor (OPENCV, single_camera_per_folder=1)
  2. Patch each camera row in database.db with Waymo intrinsics
  3. colmap exhaustive_matcher
  4. Write cameras.txt / images.txt (poses from Waymo) / points3D.txt (empty)
  5. colmap point_triangulator -> <scene>/colmap/sparse/0

Run per-scene:
    python tools/colmap_prep/build_colmap_sparse.py \
        --scene_root data/waymo/colmap_input/segment-XXXX...
"""
import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys

import numpy as np

# Axis flip: Waymo camera frame (x fwd, y left, z up) -> OpenCV (x right, y down, z fwd)
R_FLU2RDF = np.array([
    [0, -1, 0],
    [0,  0, -1],
    [1,  0, 0],
], dtype=np.float64)

# COLMAP camera model IDs
OPENCV_MODEL_ID = 4   # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2


def waymo_pose_to_world_to_camcv(T_cam_flu_to_vehicle, T_vehicle_to_world):
    """Compose Waymo cam->vehicle->world and a fixed FLU->RDF flip into a
    world -> camera_opencv 4x4 transform."""
    inv_R = R_FLU2RDF.T
    T_inv_R = np.eye(4)
    T_inv_R[:3, :3] = inv_R
    T_camcv_to_vehicle = T_cam_flu_to_vehicle @ T_inv_R
    T_camcv_to_world = T_vehicle_to_world @ T_camcv_to_vehicle
    return np.linalg.inv(T_camcv_to_world)


def rotmat_to_quat_wxyz(R):
    """Shepperd's method. Returns (qw, qx, qy, qz) with qw >= 0."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m[2, 1] - m[1, 2]) / S
        qy = (m[0, 2] - m[2, 0]) / S
        qz = (m[1, 0] - m[0, 1]) / S
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw = (m[2, 1] - m[1, 2]) / S
        qx = 0.25 * S
        qy = (m[0, 1] + m[1, 0]) / S
        qz = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw = (m[0, 2] - m[2, 0]) / S
        qx = (m[0, 1] + m[1, 0]) / S
        qy = 0.25 * S
        qz = (m[1, 2] + m[2, 1]) / S
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw = (m[1, 0] - m[0, 1]) / S
        qx = (m[0, 2] + m[2, 0]) / S
        qy = (m[1, 2] + m[2, 1]) / S
        qz = 0.25 * S
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_root", required=True)
    ap.add_argument("--colmap_bin", default="colmap")
    ap.add_argument("--gpu_index", default="4",
                    help="GPU index for SIFT extract/match. "
                         "Comma-separated for multi-GPU, '-1' for CPU.")
    ap.add_argument("--fresh", action="store_true",
                    help="delete colmap/ and re-run from scratch")
    args = ap.parse_args()

    scene_root = os.path.abspath(args.scene_root)
    images_dir = os.path.join(scene_root, "images")
    workspace = os.path.join(scene_root, "colmap")
    if args.fresh and os.path.exists(workspace):
        shutil.rmtree(workspace)
    sparse_in = os.path.join(workspace, "sparse_in")
    sparse_out = os.path.join(workspace, "sparse", "0")
    os.makedirs(sparse_in, exist_ok=True)
    os.makedirs(sparse_out, exist_ok=True)
    database = os.path.join(workspace, "database.db")

    meta_path = os.path.join(scene_root, "selection_meta.json")
    if not os.path.exists(meta_path):
        sys.exit(f"missing {meta_path}; run extract_window.py first")
    meta = json.load(open(meta_path))

    # ---- 1. feature_extractor ----
    if not os.path.exists(database):
        run([args.colmap_bin, "feature_extractor",
             "--database_path", database,
             "--image_path", images_dir,
             "--ImageReader.single_camera_per_folder", "1",
             "--ImageReader.camera_model", "OPENCV",
             "--SiftExtraction.gpu_index", args.gpu_index])

    # ---- 2. patch DB cameras with real intrinsics ----
    conn = sqlite3.connect(database)
    cur = conn.cursor()
    cur.execute("SELECT image_id, name, camera_id FROM images ORDER BY image_id")
    img_rows = cur.fetchall()

    folder_to_camid = {}
    for image_id, name, camera_id in img_rows:
        folder = name.split("/")[0]
        folder_to_camid.setdefault(folder, camera_id)

    cams_by_local_id = {c["cam_id"]: c for c in meta["cameras"]}

    for folder, cam_id in folder_to_camid.items():
        local_id = int(folder.replace("cam", ""))
        c = cams_by_local_id[local_id]
        fu, fv, cu, cv, k1, k2, p1, p2, _k3 = c["intrinsic"]
        params_blob = np.array([fu, fv, cu, cv, k1, k2, p1, p2],
                               dtype=np.float64).tobytes()
        cur.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, "
            "prior_focal_length=1 WHERE camera_id=?",
            (OPENCV_MODEL_ID, c["width"], c["height"], params_blob, cam_id))
    conn.commit()
    conn.close()

    # ---- 3. matcher ----
    run([args.colmap_bin, "exhaustive_matcher",
         "--database_path", database,
         "--SiftMatching.gpu_index", args.gpu_index])

    # ---- 4. write cameras.txt / images.txt / points3D.txt ----
    # vehicle->world pose at the moment THIS specific image was triggered
    per_image_pose = {}
    for f in meta["frames"]:
        local_idx = f["local_idx"]
        for im in f["images"]:
            cam_id = im["cam_id"]
            T_v2w = np.array(im["vehicle_pose_at_trigger_to_world"], dtype=np.float64)
            T_cam_flu_to_v = np.array(
                cams_by_local_id[cam_id]["extrinsic_cam_to_vehicle_flu"],
                dtype=np.float64)
            T_world_to_camcv = waymo_pose_to_world_to_camcv(T_cam_flu_to_v, T_v2w)
            name = f"cam{cam_id}/{local_idx:03d}.jpg"
            per_image_pose[name] = T_world_to_camcv

    with open(os.path.join(sparse_in, "cameras.txt"), "w") as fc:
        fc.write("# Camera list (OPENCV: fx fy cx cy k1 k2 p1 p2)\n")
        for folder, cam_id in folder_to_camid.items():
            local_id = int(folder.replace("cam", ""))
            c = cams_by_local_id[local_id]
            fu, fv, cu, cv, k1, k2, p1, p2, _k3 = c["intrinsic"]
            fc.write(f"{cam_id} OPENCV {c['width']} {c['height']} "
                     f"{fu} {fv} {cu} {cv} {k1} {k2} {p1} {p2}\n")

    with open(os.path.join(sparse_in, "images.txt"), "w") as fi:
        fi.write("# Image list with poses (world->camera)\n")
        for image_id, name, camera_id in img_rows:
            T = per_image_pose[name]
            R = T[:3, :3]
            t = T[:3, 3]
            qw, qx, qy, qz = rotmat_to_quat_wxyz(R)
            fi.write(f"{image_id} {qw} {qx} {qy} {qz} "
                     f"{t[0]} {t[1]} {t[2]} {camera_id} {name}\n\n")

    open(os.path.join(sparse_in, "points3D.txt"), "w").close()

    # ---- 5. triangulator ----
    run([args.colmap_bin, "point_triangulator",
         "--database_path", database,
         "--image_path", images_dir,
         "--input_path", sparse_in,
         "--output_path", sparse_out])

    print(f"\n[OK] Sparse model: {sparse_out}")
    print(f"     Inspect: colmap gui --database_path {database} "
          f"--image_path {images_dir} --import_path {sparse_out}")


if __name__ == "__main__":
    main()
