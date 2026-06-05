#!/usr/bin/env python3
"""Build COLMAP sparse model for one nuScenes clip using known poses.

Reads <clip_root>/selection_meta.json:
    cameras[].intrinsic_pinhole  ->  PINHOLE (fx, fy, cx, cy)
    images[].world_to_camcv      ->  per-image world->camera_opencv 4x4

Pipeline (same shape as the Waymo build_colmap_sparse.py we trust, but with
PINHOLE cameras and poses provided directly without any axis flip):
  1. colmap feature_extractor (PINHOLE, single_camera_per_folder=1) -> 6 cams
  2. patch the 6 DB cameras with the meta's exact intrinsics
  3. colmap exhaustive_matcher
  4. write cameras.txt / images.txt / points3D.txt(empty)
  5. colmap point_triangulator -> sparse/0

If you suspect the extrinsic_dir setting was wrong (matrices are world2cam
not cam2world), the reprojection error after triangulation will be huge
(many tens of pixels) and the run_pipeline.py harness will auto-retry
this clip, then mark it failed, and the convention-issue heuristic in
the pipeline orchestrator will flag a global abort.
"""
import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys

import numpy as np


OPENCV_MODEL_PINHOLE = 1  # COLMAP camera model id 1 = PINHOLE


def rotmat_to_quat_wxyz(R):
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m[2, 1] - m[1, 2]) / S
        qy = (m[0, 2] - m[2, 0]) / S
        qz = (m[1, 0] - m[0, 1]) / S
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
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
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    subprocess.run(cmd, check=True, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_root", required=True)
    ap.add_argument("--colmap_bin", default="colmap")
    ap.add_argument("--gpu_index", default="0")
    ap.add_argument("--cpu_sift", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    clip_root = os.path.abspath(args.clip_root)
    images_dir = os.path.join(clip_root, "images")
    workspace = os.path.join(clip_root, "colmap")
    if args.fresh and os.path.exists(workspace):
        shutil.rmtree(workspace)
    sparse_in = os.path.join(workspace, "sparse_in")
    sparse_out = os.path.join(workspace, "sparse", "0")
    os.makedirs(sparse_in, exist_ok=True)
    os.makedirs(sparse_out, exist_ok=True)
    database = os.path.join(workspace, "database.db")

    meta_path = os.path.join(clip_root, "selection_meta.json")
    if not os.path.exists(meta_path):
        sys.exit(f"missing {meta_path}; run extract_clips.py first")
    meta = json.load(open(meta_path))
    cams_by_id = {c["cam_id"]: c for c in meta["cameras"]}

    # ---- 1. feature_extractor ----
    if not os.path.exists(database):
        cmd = [args.colmap_bin, "feature_extractor",
               "--database_path", database,
               "--image_path", images_dir,
               "--ImageReader.single_camera_per_folder", "1",
               "--ImageReader.camera_model", "PINHOLE"]
        if args.cpu_sift:
            cmd += ["--SiftExtraction.use_gpu", "0"]
        else:
            cmd += ["--SiftExtraction.gpu_index", args.gpu_index]
        run(cmd)

    # ---- 2. patch DB cameras to PINHOLE with real intrinsics ----
    conn = sqlite3.connect(database)
    cur = conn.cursor()
    cur.execute("SELECT image_id, name, camera_id FROM images ORDER BY image_id")
    img_rows = cur.fetchall()
    if not img_rows:
        sys.exit("feature_extractor produced no images; check images dir")

    folder_to_camid = {}
    for image_id, name, camera_id in img_rows:
        folder = name.split("/")[0]
        folder_to_camid.setdefault(folder, camera_id)
    for folder, db_cam_id in folder_to_camid.items():
        local_id = int(folder.replace("cam", ""))
        c = cams_by_id[local_id]
        fx, fy, cx, cy = c["intrinsic_pinhole"]
        params_blob = np.array([fx, fy, cx, cy],
                               dtype=np.float64).tobytes()
        cur.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, "
            "prior_focal_length=1 WHERE camera_id=?",
            (OPENCV_MODEL_PINHOLE, c["width"], c["height"],
             params_blob, db_cam_id))
    conn.commit()
    conn.close()

    # ---- 3. matcher ----
    cmd = [args.colmap_bin, "exhaustive_matcher", "--database_path", database]
    if args.cpu_sift:
        cmd += ["--SiftMatching.use_gpu", "0"]
    else:
        cmd += ["--SiftMatching.gpu_index", args.gpu_index]
    run(cmd)

    # ---- 4. cameras.txt / images.txt / points3D.txt ----
    name_to_pose = {im["filename"]: np.array(im["world_to_camcv"],
                                              dtype=np.float64)
                    for im in meta["images"]}

    with open(os.path.join(sparse_in, "cameras.txt"), "w") as fc:
        fc.write("# CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy\n")
        for folder, db_cam_id in folder_to_camid.items():
            local_id = int(folder.replace("cam", ""))
            c = cams_by_id[local_id]
            fx, fy, cx, cy = c["intrinsic_pinhole"]
            fc.write(f"{db_cam_id} PINHOLE {c['width']} {c['height']} "
                     f"{fx} {fy} {cx} {cy}\n")

    with open(os.path.join(sparse_in, "images.txt"), "w") as fi:
        fi.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
        for image_id, name, camera_id in img_rows:
            if name not in name_to_pose:
                sys.exit(f"meta has no pose for {name}; "
                         f"DB and meta mismatch")
            T = name_to_pose[name]
            R = T[:3, :3]; t = T[:3, 3]
            qw, qx, qy, qz = rotmat_to_quat_wxyz(R)
            fi.write(f"{image_id} {qw} {qx} {qy} {qz} "
                     f"{t[0]} {t[1]} {t[2]} {camera_id} {name}\n\n")

    open(os.path.join(sparse_in, "points3D.txt"), "w").close()

    # ---- 5. triangulate ----
    run([args.colmap_bin, "point_triangulator",
         "--database_path", database,
         "--image_path", images_dir,
         "--input_path", sparse_in,
         "--output_path", sparse_out])

    print(f"\n[OK] sparse model: {sparse_out}", flush=True)


if __name__ == "__main__":
    main()
