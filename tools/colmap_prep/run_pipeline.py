#!/usr/bin/env python3
"""End-to-end Waymo -> COLMAP SfM + MVS pipeline. One command runs all stages.

Stages, each idempotent (re-running skips finished scenes):
  1. extract: extract_window.py             (N cams x M frames per scene)
  2. sfm:     build_colmap_sparse.py        (per scene, sequential, 1 GPU)
  3. check:   model_analyzer sanity check   (abort if any scene reg<expected
                                             or reprojection error > 5 px)
  4. undist:  undistort_for_3dgs.py         (per scene, sequential)
  5. mvs:     mvs_batch.py                  (parallel MVS across GPU pool)

Prerequisites in the current shell:
  - conda activate waymoprep
  - export PATH=/path/to/colmap-3.11-cuda/exe:$PATH   (verify: which colmap)

Example (the sparse-wide-baseline 20-scene experiment):
  python tools/colmap_prep/run_pipeline.py \\
      --selection tools/colmap_prep/selection_5cam_wide20.json \\
      --out_root  data/waymo/colmap_input_5cam_wide20 \\
      --window_len 8 --stride 4 --cams 0,1,2,3,4 \\
      --sfm_gpu 0 --gpus 0,1,2,3,4,5,6,7

To stop after a specific stage (e.g. just SfM + check):
  ... --stop_after check
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def banner(msg):
    print(f"\n{'=' * 80}\n=== {msg}\n{'=' * 80}", flush=True)


def run_subprocess(cmd, env=None):
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)


def list_scene_dirs(out_root):
    if not os.path.isdir(out_root):
        return []
    return sorted(os.path.join(out_root, d) for d in os.listdir(out_root)
                  if d.startswith("segment-") and
                  os.path.isdir(os.path.join(out_root, d)))


# -------------------- stages --------------------

def stage_extract(args):
    banner("Stage 1/5: extract")
    cmd = [sys.executable, os.path.join(HERE, "extract_window.py"),
           "--selection", args.selection,
           "--out_root", args.out_root,
           "--window_len", str(args.window_len),
           "--stride", str(args.stride),
           "--cams", args.cams]
    if args.overwrite_extract:
        cmd.append("--overwrite")
    run_subprocess(cmd)


def stage_sfm(args):
    banner("Stage 2/5: SfM (build_colmap_sparse)")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.sfm_gpu)
    for d in list_scene_dirs(args.out_root):
        if os.path.isfile(os.path.join(d, "colmap/sparse/0/cameras.bin")):
            print(f"  skip (done): {os.path.basename(d)}", flush=True)
            continue
        print(f"  build: {os.path.basename(d)}", flush=True)
        run_subprocess([sys.executable,
                        os.path.join(HERE, "build_colmap_sparse.py"),
                        "--scene_root", d, "--fresh"],
                       env=env)


def parse_model_analyzer(sparse_dir):
    """Return (reg, pts, err) or (None, None, None) on failure."""
    out = subprocess.run(["colmap", "model_analyzer", "--path", sparse_dir],
                         capture_output=True, text=True)
    log = (out.stdout or "") + (out.stderr or "")
    reg = pts = err = None
    for line in log.splitlines():
        if "Registered images" in line:
            m = re.search(r"(\d+)\s*$", line)
            if m: reg = int(m.group(1))
        elif "] Points:" in line or "Points: " in line:
            m = re.search(r"Points:\s*(\d+)", line)
            if m: pts = int(m.group(1))
        elif "Mean reprojection error" in line:
            m = re.search(r"([\d.]+)\s*px", line)
            if m: err = float(m.group(1))
    return reg, pts, err


def stage_check(args):
    banner("Stage 3/5: SfM sanity check")
    n_cams = len([c for c in args.cams.split(",") if c.strip()])
    expected = args.window_len * n_cams
    print(f"expected registered images per scene = "
          f"{args.window_len} frames x {n_cams} cams = {expected}\n")
    print(f"{'scene':<90s} {'reg':>5} {'pts':>7} {'err(px)':>9}  status")
    print("-" * 130)
    bad = []
    scenes = list_scene_dirs(args.out_root)
    for d in scenes:
        name = os.path.basename(d)
        sparse = os.path.join(d, "colmap/sparse/0")
        if not os.path.isfile(os.path.join(sparse, "cameras.bin")):
            print(f"{name:<90s}                          MISSING")
            bad.append((name, "no sparse model"))
            continue
        reg, pts, err = parse_model_analyzer(sparse)
        status = "OK"
        if reg is None or reg < expected:
            status = f"BAD reg"
            bad.append((name, f"reg={reg}/{expected}"))
        elif err is None or err > 5.0:
            status = f"BAD err"
            bad.append((name, f"err={err}"))
        print(f"{name:<90s} {reg!s:>5} {pts!s:>7} {err!s:>9}  {status}")
    if bad:
        print("\n[FAIL] Some scenes did not pass sanity check:", flush=True)
        for name, why in bad:
            print(f"  - {name}: {why}", flush=True)
        sys.exit(1)
    print(f"\n[OK] all {len(scenes)} scenes passed.\n")


def stage_undistort(args):
    banner("Stage 4/5: undistort")
    for d in list_scene_dirs(args.out_root):
        if os.path.isdir(os.path.join(d, "colmap/dense/images")):
            print(f"  skip (done): {os.path.basename(d)}", flush=True)
            continue
        run_subprocess([sys.executable,
                        os.path.join(HERE, "undistort_for_3dgs.py"),
                        "--scene_root", d])


def stage_mvs(args):
    banner("Stage 5/5: MVS (mvs_batch)")
    cmd = [sys.executable, os.path.join(HERE, "mvs_batch.py"),
           "--inputs_root", args.out_root,
           "--gpus", args.gpus,
           "--max_image_size", str(args.mvs_max_image_size)]
    run_subprocess(cmd)


def stage_summary(args):
    banner("Final summary")
    for d in list_scene_dirs(args.out_root):
        name = os.path.basename(d)
        fused = os.path.join(d, "colmap/dense/fused.ply")
        sparse = os.path.join(d, "colmap/sparse/0")
        reg, pts_s, err = parse_model_analyzer(sparse) \
            if os.path.isfile(os.path.join(sparse, "cameras.bin")) \
            else (None, None, None)
        if os.path.isfile(fused):
            n_pts = None
            with open(fused, "rb") as f:
                for raw in f.read(4096).split(b"\n"):
                    line = raw.decode("ascii", errors="ignore")
                    if line.startswith("element vertex"):
                        n_pts = int(line.split()[-1])
                        break
                    if line == "end_header":
                        break
            size_mb = os.path.getsize(fused) / 1e6
            print(f"  OK  {name}  sparse={pts_s}pts err={err}px  "
                  f"dense={n_pts}pts ({size_mb:.1f} MB)")
        else:
            print(f"  MISSING  {name}  sparse={pts_s}pts err={err}px  no fused.ply")
    print(f"\nABSOLUTE ROOT: {os.path.abspath(args.out_root)}", flush=True)


# -------------------- main --------------------

STAGES = ["extract", "sfm", "check", "undistort", "mvs"]


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--selection", required=True,
                    help="JSON: {scene_name: start_frame}")
    ap.add_argument("--out_root", required=True,
                    help="output root (per-scene subdirs created inside)")
    ap.add_argument("--window_len", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--cams", default="0,1,2,3,4",
                    help="0=FRONT 1=FL 2=FR 3=SL 4=SR")
    ap.add_argument("--sfm_gpu", default="0",
                    help="single GPU index for the sequential SfM stage")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7",
                    help="GPU pool for parallel MVS")
    ap.add_argument("--mvs_max_image_size", type=int, default=1600)
    ap.add_argument("--overwrite_extract", action="store_true",
                    help="force re-extract even if selection_meta.json exists")
    ap.add_argument("--stop_after", choices=STAGES, default="mvs",
                    help="run stages up to and including this one")
    ap.add_argument("--start_from", choices=STAGES, default="extract",
                    help="skip stages before this one (useful for resuming)")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    start_idx = STAGES.index(args.start_from)
    stop_idx = STAGES.index(args.stop_after)
    if stop_idx < start_idx:
        sys.exit(f"--stop_after {args.stop_after} comes before "
                 f"--start_from {args.start_from}")
    fn_for = {"extract": stage_extract, "sfm": stage_sfm,
              "check": stage_check, "undistort": stage_undistort,
              "mvs": stage_mvs}
    t0 = time.time()
    for i in range(start_idx, stop_idx + 1):
        fn_for[STAGES[i]](args)
    if stop_idx == STAGES.index("mvs"):
        stage_summary(args)
    print(f"\nTotal wall time: {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
