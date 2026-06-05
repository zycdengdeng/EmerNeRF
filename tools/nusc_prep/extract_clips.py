#!/usr/bin/env python3
"""Extract per-scene 1-second clips for nuScenes -> COLMAP-ready layout.

For each selected scene, take continuous frames starting at frame
(start_sec * fps), and split into n_clips groups of fps frames each
(so each clip covers exactly 1 second).

Per-clip output:
    <out_root>/<scene>_clip_<NN>/
        images/
            cam0/000.jpg ... 009.jpg     (10 frames, FRONT)
            cam1/...                      (FRONT_LEFT)
            cam2/...                      (FRONT_RIGHT)
            cam3/...                      (BACK_LEFT)
            cam4/...                      (BACK_RIGHT)
            cam5/...                      (BACK)
        selection_meta.json:
          {
            "scene": "<scene>", "clip_idx": NN,
            "fps": 10, "global_frame_start": M,
            "cameras": [
              {"cam_id": 0, "label": "FRONT", "width": 1600, "height": 900,
               "intrinsic_pinhole": [fx, fy, cx, cy]}, ...
            ],
            "images": [
              {"cam_id": K, "local_idx": I, "filename": "camK/III.jpg",
               "global_frame": F,
               "world_to_camcv": [[4x4]]}, ...
            ]
          }

Pose convention: nuScenes extrinsic file = 4x4 cam2world in OpenCV RDF
camera frame (right, down, forward). We invert directly to get
world_to_camcv. If your data turns out to use world2cam instead, pass
--extrinsic_dir world2cam to skip the inversion.
"""
import argparse
import json
import os

import numpy as np


CAM_LABEL = {0: "FRONT", 1: "FRONT_LEFT", 2: "FRONT_RIGHT",
             3: "BACK_LEFT", 4: "BACK_RIGHT", 5: "BACK"}


def load_intrinsics(path):
    v = np.loadtxt(path).reshape(-1)
    # nuScenes 10Hz: 9 numbers = fx, fy, cx, cy, k1, k2, p1, p2, k3 (last 5 are 0)
    if len(v) < 4:
        raise ValueError(f"unexpected intrinsic length {len(v)} in {path}")
    return float(v[0]), float(v[1]), float(v[2]), float(v[3])


def load_extrinsic(path):
    """Return 4x4 matrix from the file. Caller decides direction."""
    return np.loadtxt(path).reshape(4, 4)


def extrinsic_to_world_to_camcv(T, direction):
    """T is the matrix loaded from disk; convert to world_to_camcv."""
    if direction == "cam2world":
        return np.linalg.inv(T)
    elif direction == "world2cam":
        return T.copy()
    raise ValueError(direction)


def list_frames(scene_dir):
    """Return sorted unique frame-id strings ('000', '001', ...) seen under
    images/, restricted to those that also have all 6 cam files."""
    img_dir = os.path.join(scene_dir, "images")
    if not os.path.isdir(img_dir):
        return []
    by_frame = {}
    for name in os.listdir(img_dir):
        if not name.endswith(".jpg"):
            continue
        stem = name[:-4]
        if "_" not in stem:
            continue
        fr, cam = stem.split("_", 1)
        by_frame.setdefault(fr, set()).add(cam)
    full = sorted(fr for fr, cams in by_frame.items() if len(cams) >= 6)
    return full


def extract_one_clip(scene_dir, frames, out_dir, extrinsic_dir, scene_intrinsics,
                    overwrite):
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "selection_meta.json")
    if not overwrite and os.path.exists(meta_path):
        return False

    img_src_root = os.path.join(scene_dir, "images")
    ext_src_root = os.path.join(scene_dir, "extrinsics")
    img_out_root = os.path.join(out_dir, "images")
    for cam in range(6):
        os.makedirs(os.path.join(img_out_root, f"cam{cam}"), exist_ok=True)

    cameras = []
    for cam in range(6):
        fx, fy, cx, cy = scene_intrinsics[cam]
        cameras.append({
            "cam_id": cam,
            "label": CAM_LABEL[cam],
            "width": 1600, "height": 900,
            "intrinsic_pinhole": [fx, fy, cx, cy],
        })

    images_meta = []
    for local_idx, fr in enumerate(frames):
        for cam in range(6):
            src_img = os.path.join(img_src_root, f"{fr}_{cam}.jpg")
            src_ext = os.path.join(ext_src_root, f"{fr}_{cam}.txt")
            if not (os.path.exists(src_img) and os.path.exists(src_ext)):
                # Skip missing; rare in nuscenes_10Hz but handle defensively
                continue
            dst_name = f"cam{cam}/{local_idx:03d}.jpg"
            dst = os.path.join(img_out_root, dst_name)
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src_img), dst)
            T = load_extrinsic(src_ext)
            w2c = extrinsic_to_world_to_camcv(T, extrinsic_dir)
            images_meta.append({
                "cam_id": cam,
                "local_idx": local_idx,
                "filename": dst_name,
                "global_frame": int(fr),
                "world_to_camcv": w2c.tolist(),
            })

    meta = {
        "scene": os.path.basename(scene_dir.rstrip("/")),
        "clip_global_frames": [int(fr) for fr in frames],
        "fps": 10,
        "extrinsic_dir": extrinsic_dir,
        "cameras": cameras,
        "images": images_meta,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True,
                    help="e.g. /mnt/public_datasets/nuscenes_10Hz/trainval/000")
    ap.add_argument("--out_root", required=True,
                    help="parent dir; clips go to <out_root>/<scene>_clip_NN/")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--start_sec", type=float, default=0.0)
    ap.add_argument("--n_clips", type=int, default=10)
    ap.add_argument("--extrinsic_dir", choices=["cam2world", "world2cam"],
                    default="cam2world",
                    help="direction of the 4x4 matrices in extrinsics/*.txt")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    scene = os.path.basename(args.scene_dir.rstrip("/"))
    intr_dir = os.path.join(args.scene_dir, "intrinsics")
    scene_intrinsics = {cam: load_intrinsics(
        os.path.join(intr_dir, f"{cam}.txt")) for cam in range(6)}

    all_frames = list_frames(args.scene_dir)
    fpc = args.fps
    start = int(args.start_sec * args.fps)
    total_need = start + args.n_clips * fpc
    if len(all_frames) < total_need:
        print(f"[warn] scene {scene}: {len(all_frames)} frames available, "
              f"need {total_need}; will extract what fits.", flush=True)

    n_made = 0
    for ci in range(args.n_clips):
        s = start + ci * fpc
        e = s + fpc
        clip_frames = all_frames[s:e]
        if len(clip_frames) < 2:
            print(f"[skip] {scene} clip_{ci:02d}: only {len(clip_frames)} "
                  f"frames in [{s},{e})", flush=True)
            continue
        out_dir = os.path.join(args.out_root, f"{scene}_clip_{ci:02d}")
        ok = extract_one_clip(args.scene_dir, clip_frames, out_dir,
                              args.extrinsic_dir, scene_intrinsics,
                              args.overwrite)
        status = "new" if ok else "skip (exists)"
        print(f"[{status}] {scene}/clip_{ci:02d}: frames "
              f"{clip_frames[0]}..{clip_frames[-1]} -> {out_dir}", flush=True)
        if ok:
            n_made += 1
    print(f"\nDone scene {scene}: {n_made} new clips, "
          f"{args.n_clips - n_made} skipped.", flush=True)


if __name__ == "__main__":
    main()
