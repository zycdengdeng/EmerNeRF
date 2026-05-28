#!/usr/bin/env python3
"""Run COLMAP image_undistorter on a scene to produce 3DGS-ready PINHOLE data.

Reads:   <scene>/images/             (raw, OPENCV-distorted)
         <scene>/colmap/sparse/0/    (OPENCV cameras, world->cam poses)

Writes:  <scene>/colmap/dense/
            images/         # undistorted PINHOLE images, same subfolder layout
            sparse/0/       # PINHOLE cameras + remapped poses + points3D
            stereo/         # patch_match scaffolding (used by run_mvs.py)

3DGS (Inria gaussian-splatting) only reads SIMPLE_PINHOLE / PINHOLE, so this
step is mandatory before 3DGS training even if you skip MVS.
"""
import argparse
import os
import shutil
import subprocess


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    subprocess.run(cmd, check=True, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_root", required=True)
    ap.add_argument("--colmap_bin", default="colmap")
    ap.add_argument("--max_image_size", type=int, default=0,
                    help="0 = keep original resolution (1920x1280 for Waymo)")
    args = ap.parse_args()

    scene_root = os.path.abspath(args.scene_root)
    images_dir = os.path.join(scene_root, "images")
    sparse_in = os.path.join(scene_root, "colmap", "sparse", "0")
    dense_dir = os.path.join(scene_root, "colmap", "dense")

    if not os.path.exists(sparse_in):
        raise SystemExit(f"missing {sparse_in}; run build_colmap_sparse.py first")

    if os.path.exists(dense_dir):
        shutil.rmtree(dense_dir)
    os.makedirs(dense_dir)

    cmd = [args.colmap_bin, "image_undistorter",
           "--image_path", images_dir,
           "--input_path", sparse_in,
           "--output_path", dense_dir,
           "--output_type", "COLMAP"]
    if args.max_image_size > 0:
        cmd += ["--max_image_size", str(args.max_image_size)]
    run(cmd)

    # image_undistorter writes sparse/{cameras,images,points3D}.bin flat;
    # 3DGS expects sparse/0/.
    sparse_flat = os.path.join(dense_dir, "sparse")
    sparse_zero = os.path.join(sparse_flat, "0")
    os.makedirs(sparse_zero, exist_ok=True)
    for f in ["cameras.bin", "images.bin", "points3D.bin"]:
        src = os.path.join(sparse_flat, f)
        if os.path.exists(src):
            shutil.move(src, os.path.join(sparse_zero, f))

    print(f"\n[OK] 3DGS-ready data at:")
    print(f"     images: {os.path.join(dense_dir, 'images')}")
    print(f"     sparse: {sparse_zero}")


if __name__ == "__main__":
    main()
