#!/usr/bin/env python3
"""Extract FRONT-camera JPEGs from Waymo tfrecords so a human can scrub through
each scene and pick a 2 s (20-frame) window before doing the full 3-camera
extraction for COLMAP/3DGS.

Output layout (per scene):
    <out_dir>/<scene_name>/front/<frame_idx:03d>.jpg
    <out_dir>/<scene_name>/front_meta.json   # per-frame timestamps
"""
import argparse
import json
import os

import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2

FRONT_NAME = 1  # Waymo CameraName.FRONT


def list_segments(data_root, scene_list_file, scene_ids):
    names = open(scene_list_file).read().splitlines()
    if scene_ids:
        names = [names[i] for i in scene_ids]
    return [(name, os.path.join(data_root, name + ".tfrecord")) for name in names]


def process_one(name, tfrecord_path, out_dir, overwrite):
    scene_out = os.path.join(out_dir, name)
    img_out = os.path.join(scene_out, "front")
    meta_path = os.path.join(scene_out, "front_meta.json")
    if not overwrite and os.path.exists(meta_path):
        return f"skip (exists): {name}"
    os.makedirs(img_out, exist_ok=True)

    meta = []
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="")
    for frame_idx, data in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        for img in frame.images:
            if img.name != FRONT_NAME:
                continue
            with open(os.path.join(img_out, f"{frame_idx:03d}.jpg"), "wb") as fp:
                fp.write(img.image)
            meta.append({
                "frame_idx": frame_idx,
                "frame_timestamp_micros": int(frame.timestamp_micros),
                "camera_trigger_time": float(img.camera_trigger_time),
                "pose_timestamp": float(img.pose_timestamp),
                "shutter": float(img.shutter),
            })
            break

    with open(meta_path, "w") as f:
        json.dump({"scene": name, "num_frames": len(meta), "frames": meta}, f, indent=2)
    return f"done: {name} ({len(meta)} frames)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/waymo/raw")
    ap.add_argument("--scene_list", default="data/waymo_train_list.txt")
    ap.add_argument("--scene_ids", type=int, nargs="+", default=None,
                    help="row indices into scene_list (0-based). Omit = all.")
    ap.add_argument("--out_dir", default="data/waymo/preview")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    jobs = list_segments(args.data_root, args.scene_list, args.scene_ids)
    print(f"Will process {len(jobs)} scenes -> {args.out_dir}")
    for name, path in tqdm(jobs, desc="scenes"):
        msg = process_one(name, path, args.out_dir, args.overwrite)
        tqdm.write(msg)


if __name__ == "__main__":
    main()
