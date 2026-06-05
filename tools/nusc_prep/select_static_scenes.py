#!/usr/bin/env python3
"""Rank nuScenes scenes by "static surroundings + uniform ego motion".

For each scene under <root>:
  ego stats from lidar_pose/*.txt:
    - trajectory length (sum of consecutive xy positional deltas)
    - speed CV (std/mean of per-step length; lower => more uniform speed)
    - total heading change (sum of |dtheta| between successive position
      deltas; lower => straighter, doesn't rely on a specific column of R)
  dynamic-object stats from instances/instances_info.json:
    - n_dyn: count of vehicle/human-class instances whose center moved > 1m
    - move_sum: total dynamic displacement (sum of those instances' paths)

Filter rules (CLI-adjustable):
    length >= --min_len
    total_turn <= --max_total_turn
    speed_cv <= --max_speed_cv
    n_dyn <= --max_dyn

Two-tier picking: if fewer than --num scenes survive at the strict
--max_dyn, automatically retry with a looser --max_dyn so we still
return enough scenes.

Output:
  - prints ranked table
  - writes JSON {scene_id: stats} of the top --num to --out
"""
import argparse
import concurrent.futures
import glob
import json
import os
import sys

import numpy as np


DYNAMIC_KEYS = ("vehicle", "car", "truck", "bus", "trailer",
                "motorcycle", "bicycle", "pedestrian", "human")
MOVE_THRESH = 1.0   # meters; instance center disp > this counts as "really moving"


def ego_stats(scene_dir):
    files = sorted(glob.glob(os.path.join(scene_dir, "lidar_pose", "*.txt")))
    if len(files) < 5:
        return None
    pos = np.array([np.loadtxt(f).reshape(4, 4)[:3, 3] for f in files])
    step_vec = np.diff(pos[:, :2], axis=0)
    step_len = np.linalg.norm(step_vec, axis=1)
    length = float(step_len.sum())
    speed_cv = float(step_len.std() / (step_len.mean() + 1e-8))

    # Heading change from successive step directions, no rotation-matrix assumption
    valid = step_len > 1e-3
    dirs = step_vec[valid] / (step_len[valid, None] + 1e-8)
    if len(dirs) >= 2:
        cos = np.clip((dirs[1:] * dirs[:-1]).sum(1), -1, 1)
        total_turn = float(np.degrees(np.arccos(cos)).sum())
    else:
        total_turn = 0.0
    return dict(n=len(files), length=length, speed_cv=speed_cv,
                total_turn=total_turn)


def dynamic_stats(scene_dir):
    p = os.path.join(scene_dir, "instances", "instances_info.json")
    if not os.path.exists(p):
        return dict(n_dyn=-1, move_sum=-1.0)
    try:
        with open(p) as f:
            info = json.load(f)
    except Exception:
        return dict(n_dyn=-1, move_sum=-1.0)
    n_dyn, move_sum = 0, 0.0
    for iid, v in info.items():
        cls = str(v.get("class_name", "")).lower()
        if not any(k in cls for k in DYNAMIC_KEYS):
            continue
        fa = v.get("frame_annotations", {})
        o2w = fa.get("obj_to_world")
        if not o2w or len(o2w) < 2:
            continue
        try:
            centers = np.array([np.array(M).reshape(4, 4)[:3, 3] for M in o2w])
        except Exception:
            continue
        disp = float(np.linalg.norm(centers[-1] - centers[0]))
        path = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())
        move = max(disp, path)
        if move > MOVE_THRESH:
            n_dyn += 1
            move_sum += move
    return dict(n_dyn=n_dyn, move_sum=move_sum)


def scan_one(scene_dir):
    name = os.path.basename(scene_dir)
    e = ego_stats(scene_dir)
    if e is None:
        return None
    d = dynamic_stats(scene_dir)
    return dict(scene=name, **e, **d)


def filter_and_rank(rows, max_dyn, max_total_turn, max_speed_cv, min_len):
    good = [r for r in rows
            if r["length"] >= min_len
            and r["total_turn"] <= max_total_turn
            and r["speed_cv"] <= max_speed_cv
            and (r["n_dyn"] < 0 or r["n_dyn"] <= max_dyn)]
    good.sort(key=lambda r: (
        r["n_dyn"] if r["n_dyn"] >= 0 else 999,
        r["move_sum"] if r["move_sum"] >= 0 else 1e9,
        r["speed_cv"],
        r["total_turn"],
        -r["length"],
    ))
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="e.g. /mnt/public_datasets/nuscenes_10Hz/trainval")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--need_sec", type=float, default=10.0)
    ap.add_argument("--num", type=int, default=5,
                    help="how many scenes to pick")
    ap.add_argument("--max_total_turn", type=float, default=60.0)
    ap.add_argument("--max_speed_cv", type=float, default=0.6)
    ap.add_argument("--min_len", type=float, default=20.0)
    ap.add_argument("--max_dyn", type=int, default=0,
                    help="start strict; if not enough scenes, auto-loosen")
    ap.add_argument("--out", required=True,
                    help="write selected scenes JSON here")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--cache", default=None,
                    help="optional cache JSON of scanned stats; reuse on re-run")
    args = ap.parse_args()

    need_frames = int(args.fps * args.need_sec)

    if args.cache and os.path.exists(args.cache):
        print(f"[cache] loading {args.cache}", flush=True)
        with open(args.cache) as f:
            rows = json.load(f)
    else:
        scene_dirs = sorted([d for d in glob.glob(os.path.join(args.root, "*"))
                             if os.path.isdir(d)])
        print(f"scanning {len(scene_dirs)} scenes "
              f"with {args.workers} workers...", flush=True)
        rows = []
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers) as pool:
            for r in pool.map(scan_one, scene_dirs, chunksize=4):
                if r is not None and r["n"] >= need_frames:
                    rows.append(r)
        rows.sort(key=lambda r: r["scene"])
        if args.cache:
            os.makedirs(os.path.dirname(os.path.abspath(args.cache)) or ".",
                        exist_ok=True)
            with open(args.cache, "w") as f:
                json.dump(rows, f, indent=2)
            print(f"[cache] wrote {args.cache}", flush=True)

    print(f"\n{len(rows)} scenes have >= {need_frames} frames", flush=True)

    # Progressive loosening: try strict tiers; fall back to score-rank-top-N.
    # Each tier widens all three axes together.
    base = [r for r in rows if r["length"] >= args.min_len]
    tiers = [
        (args.max_dyn, args.max_speed_cv,    args.max_total_turn),
        (max(args.max_dyn, 2), 0.8, 90.0),
        (max(args.max_dyn, 4), 1.0, 120.0),
        (max(args.max_dyn, 8), 1.5, 180.0),
        (10**6, 10**6, 10**6),    # no constraint
    ]
    good = []
    chosen_tier = None
    for tier_idx, (md, sv, tt) in enumerate(tiers):
        good = filter_and_rank(rows, md, tt, sv, args.min_len)
        if len(good) >= args.num:
            chosen_tier = (tier_idx, md, sv, tt)
            break
    if chosen_tier is None:
        # fall back to "ignore filters, just rank everything that has min_len"
        good = filter_and_rank(rows, 10**6, 10**6, 10**6, args.min_len)
        chosen_tier = (len(tiers), 10**6, 10**6, 10**6)
    print(f"\nUsing filter tier {chosen_tier[0]}: "
          f"max_dyn<={chosen_tier[1]} max_spd_cv<={chosen_tier[2]} "
          f"max_turn<={chosen_tier[3]}; {len(good)} scenes survive.",
          flush=True)

    print(f"\n{'scene':>6} {'n_fr':>5} {'len_m':>7} {'spd_cv':>7} "
          f"{'turn°':>7} {'n_dyn':>6} {'dyn_m':>8}")
    print("-" * 60)
    for r in good[:max(args.num, 20)]:
        nd = "NA" if r["n_dyn"] < 0 else r["n_dyn"]
        ms = "NA" if r["move_sum"] < 0 else f"{r['move_sum']:.1f}"
        print(f"{r['scene']:>6} {r['n']:>5} {r['length']:>7.1f} "
              f"{r['speed_cv']:>7.2f} {r['total_turn']:>7.1f} "
              f"{str(nd):>6} {ms:>8}")

    picked = good[:args.num]
    if len(picked) < args.num:
        sys.exit(f"\n[FAIL] only found {len(picked)}/{args.num} "
                 f"scenes meeting criteria")

    # Soft warning if our best picks are still imperfect
    worst = picked[-1]
    if worst.get("n_dyn", 0) > 2:
        print(f"\n[WARN] picks include scenes with up to "
              f"{worst['n_dyn']} moving objects -- nuScenes is mostly urban, "
              f"truly static scenes are rare. SfM will still work but "
              f"dynamic objects will show as outliers in dense reconstruction.",
              flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out_obj = {"selected": [r["scene"] for r in picked],
               "stats": {r["scene"]: r for r in picked}}
    with open(args.out, "w") as f:
        json.dump(out_obj, f, indent=2)
    print(f"\nWrote {args.out}", flush=True)
    print(f"PICKED: {' '.join(out_obj['selected'])}", flush=True)


if __name__ == "__main__":
    main()
