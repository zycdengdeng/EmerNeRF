#!/usr/bin/env python3
"""Pre-flight verification of nuScenes_10Hz data layout + pose conventions.

Run this BEFORE the full pipeline to catch obvious data issues in a few
seconds instead of after 50 clips of MVS.

Checks (with hard/soft tags):
  [HARD] root exists, has >=1 scene, scene has required subdirs
  [HARD] intrinsics file parses; first 4 are plausible (fx,fy,cx,cy ~ 1k,500)
  [HARD] extrinsics file is 4x4
  [INFO] last row of extrinsic 0/0 (should be [0,0,0,1])
  [INFO] cam0 extrinsic translation vs lidar_pose translation distance
         => if < 2m on cam0, file is cam2world ✓
         => if > 5m,         file is world2cam (would translate to/from
                              cam in world units, doesn't match ego)
  [INFO] cam0 first-frame vs last-frame translation distance
         => should be ~trajectory length; if 0 each frame has local origin
  [INFO] instances_info has expected schema (frame_annotations.obj_to_world)
  [INFO] writes one preview JPG to /tmp/nusc_verify_cam0.jpg so you can
         eyeball it really is the front camera
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np


def hr(msg):
    print(f"\n--- {msg} ---", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/public_datasets/nuscenes_10Hz/trainval")
    ap.add_argument("--scene", default=None,
                    help="default = first lexicographic scene under root")
    args = ap.parse_args()

    fails = 0
    warns = 0

    hr("[1] root + scene structure")
    if not os.path.isdir(args.root):
        print(f"[HARD FAIL] root not a directory: {args.root}")
        sys.exit(1)
    print(f"root OK: {args.root}")

    scenes = sorted(d for d in os.listdir(args.root)
                    if os.path.isdir(os.path.join(args.root, d)))
    print(f"#scenes in root: {len(scenes)}  first 5: {scenes[:5]}")

    scene = args.scene or scenes[0]
    scene_dir = os.path.join(args.root, scene)
    print(f"using scene: {scene}  ->  {scene_dir}")

    required = ["images", "intrinsics", "extrinsics", "lidar_pose"]
    for sub in required:
        p = os.path.join(scene_dir, sub)
        if not os.path.isdir(p):
            print(f"[HARD FAIL] missing {p}")
            fails += 1
        else:
            n = len(os.listdir(p))
            print(f"  {sub:12s} OK ({n} files)")
    optional = ["instances", "sky_masks", "lidar"]
    for sub in optional:
        p = os.path.join(scene_dir, sub)
        if os.path.isdir(p):
            print(f"  {sub:12s} OK ({len(os.listdir(p))} files) [optional]")
        else:
            print(f"  {sub:12s} MISSING [optional]")
    if fails:
        sys.exit(f"\n[ABORT] {fails} hard failures before we can continue")

    hr("[2] intrinsics format")
    for cam in range(6):
        p = os.path.join(scene_dir, "intrinsics", f"{cam}.txt")
        if not os.path.exists(p):
            print(f"[HARD FAIL] missing {p}")
            fails += 1; continue
        v = np.loadtxt(p).reshape(-1)
        if len(v) < 4:
            print(f"[HARD FAIL] cam{cam} intrinsic has only {len(v)} numbers")
            fails += 1; continue
        fx, fy, cx, cy = v[0], v[1], v[2], v[3]
        rest = v[4:].tolist() if len(v) > 4 else []
        nonzero_rest = [x for x in rest if abs(x) > 1e-6]
        print(f"  cam{cam}: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}  "
              f"distortion[{len(rest)}] nonzero={len(nonzero_rest)}")
        if not (500 < fx < 5000 and 200 < cx < 1500):
            print(f"    [WARN] fx/cx out of plausible PINHOLE range")
            warns += 1
        if nonzero_rest:
            print(f"    [WARN] distortion is nonzero (expected undistorted "
                  f"in nuscenes_10Hz): {nonzero_rest}")
            warns += 1

    hr("[3] extrinsics format")
    e0 = os.path.join(scene_dir, "extrinsics", "000_0.txt")
    if not os.path.exists(e0):
        print(f"[HARD FAIL] missing {e0}")
        fails += 1
    else:
        T = np.loadtxt(e0)
        if T.size != 16:
            print(f"[HARD FAIL] {e0} has {T.size} numbers, not 16")
            fails += 1
        else:
            T = T.reshape(4, 4)
            print(f"  shape OK 4x4")
            last = T[3]
            print(f"  last row = {last}  (should be ~[0,0,0,1])")
            if not np.allclose(last, [0, 0, 0, 1], atol=1e-3):
                print(f"  [WARN] last row not homogeneous")
                warns += 1

    if fails:
        sys.exit(f"\n[ABORT] {fails} hard failures")

    hr("[4] extrinsic direction sanity (cam2world vs world2cam)")
    T_cam0 = np.loadtxt(os.path.join(scene_dir, "extrinsics", "000_0.txt")
                       ).reshape(4, 4)
    L0 = np.loadtxt(os.path.join(scene_dir, "lidar_pose", "000.txt")
                   ).reshape(4, 4)
    cam0_t = T_cam0[:3, 3]
    ego_t = L0[:3, 3]
    d = float(np.linalg.norm(cam0_t - ego_t))
    print(f"  cam0 t = {cam0_t}")
    print(f"  ego  t = {ego_t}")
    print(f"  ||cam0 - ego|| = {d:.3f} m")
    if d < 3.0:
        print(f"  => extrinsic is cam2world (cam in world coords, "
              f"close to ego center). [GOOD]")
    elif 3.0 <= d < 100.0:
        print(f"  => extrinsic could still be cam2world but offset is odd. "
              f"[CHECK]"); warns += 1
    else:
        print(f"  => probably WORLD2CAM (translation in cam frame, not "
              f"world). Set --extrinsic_dir world2cam in pipeline.")
        warns += 1

    hr("[5] shared world frame across frames (global vs per-frame local)")
    frames = sorted(set(os.path.basename(f).split("_")[0]
                        for f in glob.glob(os.path.join(scene_dir,
                                                       "extrinsics", "*.txt"))))
    if len(frames) >= 2:
        T_first = np.loadtxt(os.path.join(scene_dir, "extrinsics",
                                          f"{frames[0]}_0.txt")).reshape(4, 4)
        T_last = np.loadtxt(os.path.join(scene_dir, "extrinsics",
                                         f"{frames[-1]}_0.txt")).reshape(4, 4)
        L_first = np.loadtxt(os.path.join(scene_dir, "lidar_pose",
                                          f"{frames[0]}.txt")).reshape(4, 4)
        L_last = np.loadtxt(os.path.join(scene_dir, "lidar_pose",
                                         f"{frames[-1]}.txt")).reshape(4, 4)
        cam0_disp = float(np.linalg.norm(T_last[:3, 3] - T_first[:3, 3]))
        ego_disp = float(np.linalg.norm(L_last[:3, 3] - L_first[:3, 3]))
        print(f"  cam0 first vs last translation: {cam0_disp:.2f} m")
        print(f"  ego  first vs last translation: {ego_disp:.2f} m")
        if cam0_disp > 5.0 and ego_disp > 5.0:
            print(f"  => GLOBAL world frame shared across frames [GOOD]")
        elif cam0_disp < 0.5 and ego_disp < 0.5:
            print(f"  => maybe PER-FRAME local origin?? need investigation")
            warns += 1
        else:
            print(f"  => mixed signal, frame count {len(frames)} -- inspect")
            warns += 1

    hr("[6] instances_info schema (for select_static_scenes.py)")
    ins_p = os.path.join(scene_dir, "instances", "instances_info.json")
    if not os.path.exists(ins_p):
        print(f"  [WARN] no instances_info.json; selector will only use ego "
              f"stats (n_dyn=NA)")
        warns += 1
    else:
        with open(ins_p) as f:
            info = json.load(f)
        print(f"  #instances = {len(info)}")
        if info:
            iid = next(iter(info))
            v = info[iid]
            cls = v.get("class_name", "<missing>")
            fa = v.get("frame_annotations", {})
            o2w = fa.get("obj_to_world")
            print(f"  sample id={iid} class={cls} #obj_to_world={len(o2w) if o2w else 0}")
            if o2w:
                T0 = np.array(o2w[0])
                print(f"  first obj_to_world shape = {T0.shape}")
                if T0.shape == (4, 4):
                    print(f"  first center = {T0[:3, 3]}")
                elif T0.size == 16:
                    print(f"  flat 16 nums; OK after reshape")

    hr("[7] image preview")
    src = os.path.join(scene_dir, "images", "000_0.jpg")
    if os.path.exists(src):
        dst = "/tmp/nusc_verify_cam0.jpg"
        shutil.copyfile(src, dst)
        sz = os.path.getsize(dst) / 1024
        print(f"  copied {src} -> {dst}  ({sz:.1f} KB)")
        print(f"  please view it on your machine (scp or open over VSCode "
              f"remote): it should be the FRONT camera (road ahead).")
    else:
        print(f"  [WARN] no {src}")
        warns += 1

    hr("verification result")
    print(f"hard failures: {fails}")
    print(f"warnings:      {warns}")
    if fails == 0 and warns == 0:
        print("\nALL CLEAR. safe to run the full pipeline.")
    elif fails == 0:
        print("\nNo hard failures. Review warnings; auto-detect in pipeline "
              "should still handle most.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
