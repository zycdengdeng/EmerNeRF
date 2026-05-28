#!/usr/bin/env python3
"""Run COLMAP MVS on an undistorted scene workspace.

Pipeline:
  1. patch_match_stereo (geometric consistency) -> stereo/{depth,normal}_maps
  2. stereo_fusion -> dense/fused.ply

Requires undistort_for_3dgs.py to have been run first.

Pinning to a specific GPU: set CUDA_VISIBLE_DEVICES outside the script and
keep --gpu_index 0 (default). Multi-GPU patch-match: pass --gpu_index 0,1
after exporting CUDA_VISIBLE_DEVICES with two GPUs.
"""
import argparse
import os
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
    ap.add_argument("--gpu_index", default="0",
                    help="GPU index inside CUDA_VISIBLE_DEVICES masking")
    ap.add_argument("--max_image_size", type=int, default=1600,
                    help="downscale longest side for MVS to save memory/time. "
                         "0 = no limit. Default 1600 trades a little detail for "
                         "much faster patch_match.")
    args = ap.parse_args()

    scene_root = os.path.abspath(args.scene_root)
    dense_dir = os.path.join(scene_root, "colmap", "dense")
    if not os.path.isdir(dense_dir):
        raise SystemExit(f"missing {dense_dir}; run undistort_for_3dgs.py first")

    pm_cmd = [args.colmap_bin, "patch_match_stereo",
              "--workspace_path", dense_dir,
              "--workspace_format", "COLMAP",
              "--PatchMatchStereo.geom_consistency", "true",
              "--PatchMatchStereo.gpu_index", args.gpu_index]
    if args.max_image_size > 0:
        pm_cmd += ["--PatchMatchStereo.max_image_size", str(args.max_image_size)]
    run(pm_cmd)

    fused_ply = os.path.join(dense_dir, "fused.ply")
    run([args.colmap_bin, "stereo_fusion",
         "--workspace_path", dense_dir,
         "--workspace_format", "COLMAP",
         "--input_type", "geometric",
         "--output_path", fused_ply])

    print(f"\n[OK] Dense cloud: {fused_ply}")


if __name__ == "__main__":
    main()
