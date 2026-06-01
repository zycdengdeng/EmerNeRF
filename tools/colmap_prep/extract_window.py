#!/usr/bin/env python3
"""Extract a 20-frame, 3-camera window (FRONT, FRONT_LEFT, FRONT_RIGHT) from
each selected Waymo scene, together with all the calibration + pose metadata
needed to build COLMAP files later.

Per-scene output layout:
    <out_root>/<scene>/
        images/
            cam0/000.jpg ... 019.jpg     # FRONT       (waymo CameraName=1)
            cam1/000.jpg ... 019.jpg     # FRONT_LEFT  (waymo CameraName=2)
            cam2/000.jpg ... 019.jpg     # FRONT_RIGHT (waymo CameraName=3)
        selection_meta.json              # cameras + per-frame poses + timestamps

The per-image vehicle pose (`vehicle_pose_at_trigger_to_world`) is what we'll
later combine with the FLU->RDF axis flip to write each image's COLMAP pose.

Run after `extract_front_preview.py` and after editing selection.json.
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2

# Waymo CameraName: 1=FRONT, 2=FRONT_LEFT, 3=FRONT_RIGHT, 4=SIDE_LEFT, 5=SIDE_RIGHT
LOCAL_ID_TO_WAYMO_NAME = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
LOCAL_ID_TO_LABEL = {0: "FRONT", 1: "FRONT_LEFT", 2: "FRONT_RIGHT",
                     3: "SIDE_LEFT", 4: "SIDE_RIGHT"}


def mat44(transform):
    return np.asarray(transform, dtype=np.float64).reshape(4, 4).tolist()


def process_one(scene_name, start_frame, window_len, cam_local_ids,
                data_root, out_root, overwrite):
    # Map: waymo CameraName -> our local id
    waymo_name_to_local = {LOCAL_ID_TO_WAYMO_NAME[i]: i for i in cam_local_ids}
    scene_out = os.path.join(out_root, scene_name)
    meta_path = os.path.join(scene_out, "selection_meta.json")
    if not overwrite and os.path.exists(meta_path):
        return f"skip (exists): {scene_name}"
    img_dir = os.path.join(scene_out, "images")
    for c in cam_local_ids:
        os.makedirs(os.path.join(img_dir, f"cam{c}"), exist_ok=True)

    tfpath = os.path.join(data_root, scene_name + ".tfrecord")
    end_frame = start_frame + window_len
    cameras_info = None
    frames_meta = []

    dataset = tf.data.TFRecordDataset(tfpath, compression_type="")
    for frame_idx, data in enumerate(dataset):
        if frame_idx < start_frame:
            continue
        if frame_idx >= end_frame:
            break
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        if cameras_info is None:
            cameras_info = {}
            for calib in frame.context.camera_calibrations:
                if calib.name not in waymo_name_to_local:
                    continue
                local_id = waymo_name_to_local[calib.name]
                cameras_info[local_id] = {
                    "cam_id": local_id,
                    "label": LOCAL_ID_TO_LABEL[local_id],
                    "waymo_name": int(calib.name),
                    "width": int(calib.width),
                    "height": int(calib.height),
                    "intrinsic": [float(x) for x in calib.intrinsic],
                    "extrinsic_cam_to_vehicle_flu": mat44(calib.extrinsic.transform),
                }

        local_idx = frame_idx - start_frame
        per_image = []
        for img in frame.images:
            if img.name not in waymo_name_to_local:
                continue
            local_cam = waymo_name_to_local[img.name]
            outp = os.path.join(img_dir, f"cam{local_cam}", f"{local_idx:03d}.jpg")
            with open(outp, "wb") as fp:
                fp.write(img.image)
            per_image.append({
                "cam_id": local_cam,
                "camera_trigger_time": float(img.camera_trigger_time),
                "pose_timestamp": float(img.pose_timestamp),
                "shutter": float(img.shutter),
                "vehicle_pose_at_trigger_to_world": mat44(img.pose.transform),
            })

        frames_meta.append({
            "local_idx": local_idx,
            "global_idx": frame_idx,
            "frame_timestamp_micros": int(frame.timestamp_micros),
            "frame_pose_vehicle_to_world": mat44(frame.pose.transform),
            "images": sorted(per_image, key=lambda x: x["cam_id"]),
        })

    if not frames_meta:
        return f"FAIL: {scene_name} (no frames in [{start_frame}, {end_frame}))"

    meta = {
        "scene": scene_name,
        "start_frame": int(start_frame),
        "num_frames": len(frames_meta),
        "cameras": [cameras_info[c] for c in sorted(cameras_info)],
        "frames": frames_meta,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return f"done: {scene_name} (frames {start_frame}..{start_frame+len(frames_meta)-1})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="tools/colmap_prep/selection.json")
    ap.add_argument("--data_root", default="data/waymo/raw")
    ap.add_argument("--out_root", default="data/waymo/colmap_input")
    ap.add_argument("--window_len", type=int, default=20,
                    help="number of consecutive frames to extract per scene")
    ap.add_argument("--cams", default="0,1,2",
                    help="comma-separated local camera ids. "
                         "0=FRONT 1=FRONT_LEFT 2=FRONT_RIGHT "
                         "3=SIDE_LEFT 4=SIDE_RIGHT. "
                         "default '0,1,2' = front three.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cam_local_ids = [int(x) for x in args.cams.split(",") if x.strip() != ""]
    for c in cam_local_ids:
        if c not in LOCAL_ID_TO_WAYMO_NAME:
            raise SystemExit(f"invalid cam id {c}; must be in 0..4")
    sel = json.load(open(args.selection))
    print(f"Will process {len(sel)} scenes (window={args.window_len}, "
          f"cams={[LOCAL_ID_TO_LABEL[c] for c in cam_local_ids]}) "
          f"-> {args.out_root}")
    for scene_name, start in tqdm(list(sel.items()), desc="scenes"):
        msg = process_one(scene_name, int(start), args.window_len,
                          cam_local_ids, args.data_root, args.out_root,
                          args.overwrite)
        tqdm.write(msg)


if __name__ == "__main__":
    main()
