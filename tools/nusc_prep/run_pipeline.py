#!/usr/bin/env python3
"""End-to-end nuScenes -> 1s clips -> SfM + MVS, with auto-convention check.

Stages:
  0. select static scenes (cached)
  1. extract 5 scenes x 10 clips = 50 clip dirs (per-cam-folder layout
     + selection_meta.json with PINHOLE intrinsics and world->cam_cv poses
     per image)
  2. SfM smoke test: build the first 2 clips. If both have reproj err
     > 50 px, auto-flip --extrinsic_dir (cam2world <-> world2cam),
     re-extract those 2 clips, and re-SfM. If still both bad, ABORT
     with a clear message before wasting time on 48 more clips.
  3. SfM the rest of the clips (sequential, per-clip with auto-retry)
  4. soft-fail sanity check (failing clips are skipped, others proceed)
  5. undistort passing clips (sequential)
  6. MVS using a local GPU-pool worker (parallel across --gpus)
  7. final per-clip summary

Run from waymoprep conda env with PATH set to your CUDA-capable colmap.

Example:
  python tools/nusc_prep/run_pipeline.py \\
      --root /mnt/public_datasets/nuscenes_10Hz/trainval \\
      --out_root /mnt/zihanw/gsnet_nusc \\
      --num_scenes 5 --n_clips 10 \\
      --sfm_gpu 0 --gpus 0,1,2,3,4,5,6,7
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
SELECT = os.path.join(HERE, "select_static_scenes.py")
EXTRACT = os.path.join(HERE, "extract_clips.py")
BUILD = os.path.join(HERE, "build_clip_sparse.py")
COLMAP_PREP = os.path.normpath(os.path.join(HERE, "..", "colmap_prep"))
UNDISTORT = os.path.join(COLMAP_PREP, "undistort_for_3dgs.py")
RUN_MVS = os.path.join(COLMAP_PREP, "run_mvs.py")

EXPECTED_REG_PER_CLIP = 60   # 10 frames x 6 cams


def banner(msg):
    print(f"\n{'=' * 80}\n=== {msg}\n{'=' * 80}", flush=True)


def run_subprocess(cmd, check=True, env=None):
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=check, env=env)


def parse_model_analyzer(sparse_dir):
    try:
        out = subprocess.run(["colmap", "model_analyzer", "--path", sparse_dir],
                             capture_output=True, text=True, timeout=60)
    except Exception:
        return None, None, None
    log = (out.stdout or "") + (out.stderr or "")
    reg = pts = err = None
    for line in log.splitlines():
        if "Registered images" in line:
            m = re.search(r"(\d+)\s*$", line)
            if m: reg = int(m.group(1))
        elif "Points:" in line and "model.cc" in line:
            m = re.search(r"Points:\s*(\d+)", line)
            if m: pts = int(m.group(1))
        elif "Mean reprojection error" in line:
            m = re.search(r"([\d.]+)\s*px", line)
            if m: err = float(m.group(1))
    return reg, pts, err


def list_clip_dirs(out_root):
    if not os.path.isdir(out_root):
        return []
    return sorted(os.path.join(out_root, d) for d in os.listdir(out_root)
                  if os.path.isdir(os.path.join(out_root, d))
                  and os.path.isfile(os.path.join(out_root, d,
                                                  "selection_meta.json")))


def sfm_ok(clip_dir, max_err):
    sparse = os.path.join(clip_dir, "colmap/sparse/0")
    if not os.path.isfile(os.path.join(sparse, "cameras.bin")):
        return False, "no sparse model", None
    reg, pts, err = parse_model_analyzer(sparse)
    if reg is None:
        return False, "model_analyzer parse fail", None
    if reg < EXPECTED_REG_PER_CLIP:
        return False, f"reg={reg}/{EXPECTED_REG_PER_CLIP}", err
    if err is None or err > max_err:
        return False, f"err={err}>{max_err}", err
    return True, "ok", err


# -------------------- stage: select --------------------

def stage_select(args):
    banner("Stage 0: select static scenes")
    sel_json = os.path.join(args.out_root, "selected_scenes.json")
    cache = os.path.join(args.out_root, "_scan_cache.json")
    if args.skip_select and os.path.exists(sel_json):
        print(f"--skip_select: reusing {sel_json}", flush=True)
    else:
        run_subprocess([sys.executable, SELECT,
                        "--root", args.root,
                        "--num", str(args.num_scenes),
                        "--max_dyn", str(args.max_dyn),
                        "--need_sec", str(args.start_sec + args.n_clips),
                        "--out", sel_json,
                        "--cache", cache,
                        "--workers", str(args.scan_workers)])
    sel = json.load(open(sel_json))
    print(f"selected: {sel['selected']}", flush=True)
    return sel["selected"]


# -------------------- stage: extract --------------------

def stage_extract(args, scenes, extrinsic_dir, overwrite=False):
    banner(f"Stage 1: extract clips (extrinsic_dir={extrinsic_dir})")
    for sc in scenes:
        cmd = [sys.executable, EXTRACT,
               "--scene_dir", os.path.join(args.root, sc),
               "--out_root", args.out_root,
               "--fps", str(args.fps),
               "--start_sec", str(args.start_sec),
               "--n_clips", str(args.n_clips),
               "--extrinsic_dir", extrinsic_dir]
        if overwrite:
            cmd.append("--overwrite")
        run_subprocess(cmd)


# -------------------- stage: sfm (per clip with retry) --------------------

def build_one_clip(clip_dir, sfm_gpu, max_attempts, max_err):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(sfm_gpu)
    name = os.path.basename(clip_dir)
    ok, reason, _ = sfm_ok(clip_dir, max_err)
    if ok:
        print(f"  skip (already ok): {name}", flush=True)
        return True
    for attempt in range(1, max_attempts + 1):
        print(f"  build attempt {attempt}/{max_attempts}: {name}", flush=True)
        try:
            subprocess.run(
                [sys.executable, BUILD, "--clip_root", clip_dir, "--fresh"],
                check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"    [crash rc={e.returncode}]", flush=True)
            continue
        ok, reason, err = sfm_ok(clip_dir, max_err)
        if ok:
            print(f"    [ok] err={err}", flush=True)
            return True
        print(f"    [below threshold] {reason}", flush=True)
    print(f"  [GIVE UP] {name}", flush=True)
    return False


def stage_sfm_smoke(args):
    """Build the first 2 clips and report mean err."""
    banner("Stage 2a: SfM smoke test (first 2 clips)")
    clips = list_clip_dirs(args.out_root)[:2]
    if len(clips) < 2:
        sys.exit("need at least 2 clips for smoke test")
    errs = []
    for d in clips:
        build_one_clip(d, args.sfm_gpu, args.sfm_max_attempts, 1000.0)
        _, _, err = sfm_ok(d, 1000.0)   # we only want the err here
        errs.append(err if err is not None else 1e6)
    print(f"\nSmoke test errors: {errs}", flush=True)
    return errs


def stage_sfm_rest(args):
    banner("Stage 2b: SfM for remaining clips")
    for d in list_clip_dirs(args.out_root):
        build_one_clip(d, args.sfm_gpu,
                       args.sfm_max_attempts, args.sfm_max_err)


# -------------------- stage: check --------------------

def stage_check(args):
    banner("Stage 3: sanity check (non-blocking)")
    print(f"{'clip':<40} {'reg':>4} {'pts':>7} {'err(px)':>8}  status")
    print("-" * 80)
    failed = []
    for d in list_clip_dirs(args.out_root):
        name = os.path.basename(d)
        sparse = os.path.join(d, "colmap/sparse/0")
        if not os.path.isfile(os.path.join(sparse, "cameras.bin")):
            print(f"{name:<40} {'-':>4} {'-':>7} {'-':>8}  NO_SPARSE")
            failed.append(name)
            continue
        reg, pts, err = parse_model_analyzer(sparse)
        ok = (reg is not None and reg >= EXPECTED_REG_PER_CLIP and
              err is not None and err <= args.sfm_max_err)
        status = "OK" if ok else "FAIL"
        print(f"{name:<40} {reg!s:>4} {pts!s:>7} {err!s:>8}  {status}")
        if not ok:
            failed.append(name)
    if failed:
        print(f"\n[WARN] {len(failed)} clip(s) failed; "
              f"they will be skipped in undistort + MVS:")
        for n in failed:
            print(f"  - {n}")
    return failed


# -------------------- stage: undistort --------------------

def stage_undistort(args, failed_set):
    banner("Stage 4: undistort")
    for d in list_clip_dirs(args.out_root):
        name = os.path.basename(d)
        if name in failed_set:
            print(f"  skip (sfm failed): {name}", flush=True)
            continue
        if os.path.isdir(os.path.join(d, "colmap/dense/images")):
            print(f"  skip (done): {name}", flush=True)
            continue
        try:
            run_subprocess([sys.executable, UNDISTORT, "--scene_root", d])
        except subprocess.CalledProcessError as e:
            print(f"  [FAIL] undistort {name}: rc={e.returncode}", flush=True)


# -------------------- stage: mvs (inline GPU pool) --------------------

def stage_mvs(args, failed_set):
    banner("Stage 5: MVS (GPU pool)")
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    print(f"GPUs: {gpus}", flush=True)

    for attempt in range(1, args.mvs_max_attempts + 1):
        pending = []
        for d in list_clip_dirs(args.out_root):
            name = os.path.basename(d)
            if name in failed_set:
                continue
            if not os.path.isdir(os.path.join(d, "colmap/dense")):
                continue
            if os.path.isfile(os.path.join(d, "colmap/dense/fused.ply")):
                continue
            pending.append(d)
        if not pending:
            print("nothing pending.", flush=True)
            return
        print(f"\nMVS wave {attempt}/{args.mvs_max_attempts}: "
              f"{len(pending)} clips pending\n", flush=True)
        run_mvs_pool(pending, gpus, args.mvs_max_image_size)


def run_mvs_pool(pending, gpus, max_image_size):
    queue = deque(pending)
    workers = {g: None for g in gpus}

    def launch(gpu):
        if not queue:
            return None
        clip_dir = queue.popleft()
        log = os.path.join(clip_dir, "mvs.log")
        cmd = [sys.executable, RUN_MVS,
               "--scene_root", clip_dir,
               "--max_image_size", str(max_image_size)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        fp = open(log, "w")
        proc = subprocess.Popen(cmd, env=env, stdout=fp,
                                stderr=subprocess.STDOUT)
        print(f"  [GPU {gpu}] LAUNCH {os.path.basename(clip_dir)} "
              f"pid={proc.pid}", flush=True)
        return (proc, clip_dir, fp, time.time())

    def cleanup(*_):
        for w in workers.values():
            if w is None: continue
            try: w[0].terminate()
            except Exception: pass
        sys.exit(1)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    while queue or any(workers[g] for g in gpus):
        for g in gpus:
            if workers[g] is None and queue:
                workers[g] = launch(g)
        time.sleep(5)
        for g in gpus:
            if workers[g] is None: continue
            proc, clip_dir, fp, t0 = workers[g]
            rc = proc.poll()
            if rc is None: continue
            fp.close()
            dt = (time.time() - t0) / 60
            ok = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"  [GPU {g}] FINISH {os.path.basename(clip_dir)} "
                  f"{ok} ({dt:.1f}min)", flush=True)
            workers[g] = None


# -------------------- summary --------------------

def stage_summary(args):
    banner("Final summary")
    print(f"{'clip':<40}  {'sfm':<6}  {'und':<4}  {'mvs':<5}  detail")
    print("-" * 110)
    n_ok = n_sfm_fail = n_mvs_fail = 0
    for d in list_clip_dirs(args.out_root):
        name = os.path.basename(d)
        ok, reason, err = sfm_ok(d, args.sfm_max_err)
        sfm_tag = "OK" if ok else "FAIL"
        und_tag = "OK" if os.path.isdir(
            os.path.join(d, "colmap/dense/images")) else "-"
        fused = os.path.join(d, "colmap/dense/fused.ply")
        if os.path.isfile(fused):
            mvs_tag = "OK"
            n_pts = None
            with open(fused, "rb") as f:
                for raw in f.read(4096).split(b"\n"):
                    line = raw.decode("ascii", errors="ignore")
                    if line.startswith("element vertex"):
                        n_pts = int(line.split()[-1]); break
                    if line == "end_header":
                        break
            size_mb = os.path.getsize(fused) / 1e6
            detail = f"sfm_err={err}  dense={n_pts}pts ({size_mb:.1f} MB)"
        else:
            mvs_tag = "FAIL" if und_tag == "OK" else "-"
            detail = f"sfm_reason={reason}"
        if sfm_tag == "OK" and mvs_tag == "OK":
            n_ok += 1
        elif sfm_tag != "OK":
            n_sfm_fail += 1
        elif mvs_tag == "FAIL":
            n_mvs_fail += 1
        print(f"{name:<40}  {sfm_tag:<6}  {und_tag:<4}  {mvs_tag:<5}  {detail}")
    print()
    print(f"Totals: all_ok={n_ok}  sfm_failed={n_sfm_fail}  "
          f"mvs_failed={n_mvs_fail}")
    print(f"\nABSOLUTE ROOT: {os.path.abspath(args.out_root)}", flush=True)


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--root", required=True,
                    help="/mnt/public_datasets/nuscenes_10Hz/trainval")
    ap.add_argument("--out_root", required=True,
                    help="/mnt/zihanw/gsnet_nusc")
    ap.add_argument("--num_scenes", type=int, default=5)
    ap.add_argument("--n_clips", type=int, default=10)
    ap.add_argument("--start_sec", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max_dyn", type=int, default=0,
                    help="strict 'static' by default; selector auto-loosens")
    ap.add_argument("--skip_select", action="store_true",
                    help="reuse <out_root>/selected_scenes.json if present")
    ap.add_argument("--scan_workers", type=int, default=16)
    ap.add_argument("--extrinsic_dir",
                    choices=["cam2world", "world2cam", "auto"],
                    default="auto",
                    help="auto = try cam2world first; if SfM smoke fails, "
                         "auto-flip to world2cam and retry")
    ap.add_argument("--sfm_gpu", default="0",
                    help="single GPU for sequential SfM")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7",
                    help="GPU pool for parallel MVS")
    ap.add_argument("--sfm_max_attempts", type=int, default=2)
    ap.add_argument("--sfm_max_err", type=float, default=5.0)
    ap.add_argument("--mvs_max_attempts", type=int, default=2)
    ap.add_argument("--mvs_max_image_size", type=int, default=1600)
    ap.add_argument("--start_from",
                    choices=["select", "extract", "sfm", "check",
                             "undistort", "mvs"],
                    default="select")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    t0 = time.time()

    if args.start_from in ("select",):
        scenes = stage_select(args)
    else:
        sel_json = os.path.join(args.out_root, "selected_scenes.json")
        if os.path.exists(sel_json):
            scenes = json.load(open(sel_json))["selected"]
        else:
            sys.exit(f"--start_from {args.start_from} but no "
                     f"{sel_json}; run select stage first")

    # ----- extract + SfM smoke with auto convention detection -----
    if args.start_from in ("select", "extract"):
        first_try_dir = "cam2world" if args.extrinsic_dir == "auto" \
            else args.extrinsic_dir
        stage_extract(args, scenes, first_try_dir)
    if args.start_from in ("select", "extract", "sfm"):
        errs = stage_sfm_smoke(args)
        bad = sum(1 for e in errs if e is None or e > 50.0)
        if bad >= 2 and args.extrinsic_dir == "auto":
            print("\n[AUTO] both smoke clips failed with high reproj err; "
                  "trying extrinsic_dir=world2cam...", flush=True)
            # nuke all clip dirs and re-extract with flipped convention
            for d in list_clip_dirs(args.out_root):
                shutil.rmtree(d)
            stage_extract(args, scenes, "world2cam", overwrite=True)
            errs2 = stage_sfm_smoke(args)
            bad2 = sum(1 for e in errs2 if e is None or e > 50.0)
            if bad2 >= 2:
                sys.exit(f"\n[ABORT] both extrinsic_dir attempts failed:\n"
                         f"  cam2world errors: {errs}\n"
                         f"  world2cam errors: {errs2}\n"
                         f"Either the data layout is different than expected, "
                         f"or the camera frame is not OpenCV RDF. Inspect "
                         f"one extrinsic + one image manually before retry.")
            print(f"[AUTO] world2cam works; continuing.\n", flush=True)
        elif bad >= 2:
            sys.exit(f"\n[ABORT] smoke test failed with errors {errs} and "
                     f"extrinsic_dir={args.extrinsic_dir} is locked. "
                     f"Re-run with --extrinsic_dir auto.")
        stage_sfm_rest(args)

    # Always compute the failed set from current on-disk SfM state, so that
    # --start_from=undistort/mvs still skips broken clips correctly.
    failed_set = set()
    for d in list_clip_dirs(args.out_root):
        ok, _, _ = sfm_ok(d, args.sfm_max_err)
        if not ok:
            failed_set.add(os.path.basename(d))

    if args.start_from in ("select", "extract", "sfm", "check"):
        stage_check(args)

    if args.start_from in ("select", "extract", "sfm", "check", "undistort"):
        stage_undistort(args, failed_set)

    stage_mvs(args, failed_set)
    stage_summary(args)
    print(f"\nTotal wall time: {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
