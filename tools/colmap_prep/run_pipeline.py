#!/usr/bin/env python3
"""End-to-end Waymo -> COLMAP SfM + MVS pipeline with self-healing.

Stages (each idempotent; rerun the same command to resume):
  1. extract:   extract_window.py             (N cams x M frames per scene)
  2. sfm:       build_colmap_sparse.py        (per scene, sequential, 1 GPU)
                + auto-retry up to --sfm_max_attempts on crash or
                  quality-below-threshold (reg < expected or err > 5 px).
  3. check:     print pass/fail table. Failed scenes are EXCLUDED from
                downstream stages but the pipeline DOES NOT abort, so good
                scenes still progress.
  4. undistort: undistort_for_3dgs.py per passing scene
  5. mvs:       mvs_batch.py across GPU pool, auto-rerun on any scene
                whose fused.ply is still missing, up to --mvs_max_attempts.
  6. summary:   per-scene table {sfm ok?, dense pts, fused size}.

Prerequisites in current shell:
  - conda activate waymoprep
  - export PATH=/path/to/colmap-3.11-cuda/exe:$PATH    (verify: which colmap)

Example (sparse-wide-baseline 20-scene experiment):
  python tools/colmap_prep/run_pipeline.py \\
      --selection tools/colmap_prep/selection_5cam_wide20.json \\
      --out_root  data/waymo/colmap_input_5cam_wide20 \\
      --window_len 8 --stride 4 --cams 0,1,2,3,4 \\
      --sfm_gpu 0 --gpus 0,1,2,3,4,5,6,7
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


def list_scene_dirs(out_root):
    if not os.path.isdir(out_root):
        return []
    return sorted(os.path.join(out_root, d) for d in os.listdir(out_root)
                  if d.startswith("segment-") and
                  os.path.isdir(os.path.join(out_root, d)))


def parse_model_analyzer(sparse_dir):
    """Return (reg, pts, err) or (None, None, None) if colmap fails."""
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


def sfm_quality_ok(scene_dir, expected_reg, max_err):
    sparse = os.path.join(scene_dir, "colmap/sparse/0")
    if not os.path.isfile(os.path.join(sparse, "cameras.bin")):
        return False, "no sparse model"
    reg, _, err = parse_model_analyzer(sparse)
    if reg is None:
        return False, "model_analyzer parse failure"
    if reg < expected_reg:
        return False, f"reg={reg} < {expected_reg}"
    if err is None or err > max_err:
        return False, f"err={err} > {max_err}"
    return True, "ok"


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
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def stage_sfm(args):
    """Per-scene SfM with auto-retry. Crashes / poor quality both trigger retry."""
    banner("Stage 2/5: SfM (with auto-retry)")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.sfm_gpu)
    n_cams = len([c for c in args.cams.split(",") if c.strip()])
    expected = args.window_len * n_cams

    for d in list_scene_dirs(args.out_root):
        name = os.path.basename(d)
        ok, reason = sfm_quality_ok(d, expected, args.sfm_max_err)
        if ok:
            print(f"  skip (already ok): {name}", flush=True)
            continue
        for attempt in range(1, args.sfm_max_attempts + 1):
            print(f"\n  build {name}  attempt {attempt}/{args.sfm_max_attempts}",
                  flush=True)
            try:
                subprocess.run(
                    [sys.executable,
                     os.path.join(HERE, "build_colmap_sparse.py"),
                     "--scene_root", d, "--fresh"],
                    check=True, env=env)
            except subprocess.CalledProcessError as e:
                print(f"  [crash] attempt {attempt}: rc={e.returncode}",
                      flush=True)
                continue
            ok, reason = sfm_quality_ok(d, expected, args.sfm_max_err)
            if ok:
                print(f"  [ok] {name}", flush=True)
                break
            print(f"  [below threshold] attempt {attempt}: {reason}",
                  flush=True)
        else:
            print(f"  [GIVE UP] {name} after {args.sfm_max_attempts} attempts",
                  flush=True)


def stage_check(args):
    """Classify scenes. Failed ones get printed but don't abort the pipeline."""
    banner("Stage 3/5: SfM sanity check (non-blocking)")
    n_cams = len([c for c in args.cams.split(",") if c.strip()])
    expected = args.window_len * n_cams
    print(f"expected registered = {args.window_len} frames x {n_cams} cams "
          f"= {expected}\n")
    print(f"{'scene':<90s} {'reg':>5} {'pts':>8} {'err(px)':>9}  status")
    print("-" * 130)
    failed = []
    for d in list_scene_dirs(args.out_root):
        name = os.path.basename(d)
        sparse = os.path.join(d, "colmap/sparse/0")
        if not os.path.isfile(os.path.join(sparse, "cameras.bin")):
            print(f"{name:<90s} {'-':>5} {'-':>8} {'-':>9}  NO_SPARSE")
            failed.append((name, "no sparse model"))
            continue
        reg, pts, err = parse_model_analyzer(sparse)
        ok = (reg is not None and reg >= expected and
              err is not None and err <= args.sfm_max_err)
        status = "OK"
        if not ok:
            status = "FAIL"
            failed.append((name, f"reg={reg} err={err}"))
        print(f"{name:<90s} {reg!s:>5} {pts!s:>8} {err!s:>9}  {status}")
    if failed:
        print(f"\n[WARN] {len(failed)} scenes did not pass; "
              f"they will be SKIPPED in undistort/MVS:")
        for name, why in failed:
            print(f"  - {name}: {why}")
        print()


def stage_undistort(args):
    """Only on scenes that passed SfM quality."""
    banner("Stage 4/5: undistort")
    n_cams = len([c for c in args.cams.split(",") if c.strip()])
    expected = args.window_len * n_cams
    for d in list_scene_dirs(args.out_root):
        name = os.path.basename(d)
        ok, reason = sfm_quality_ok(d, expected, args.sfm_max_err)
        if not ok:
            print(f"  skip (sfm not ok: {reason}): {name}", flush=True)
            continue
        if os.path.isdir(os.path.join(d, "colmap/dense/images")):
            print(f"  skip (done): {name}", flush=True)
            continue
        print(f"  undistort: {name}", flush=True)
        try:
            subprocess.run(
                [sys.executable, os.path.join(HERE, "undistort_for_3dgs.py"),
                 "--scene_root", d], check=True)
        except subprocess.CalledProcessError as e:
            print(f"  [FAIL] undistort {name}: {e}", flush=True)


def stage_mvs(args):
    """MVS across GPU pool, auto-rerun on any still-missing fused.ply."""
    banner("Stage 5/5: MVS (with auto-retry)")
    for attempt in range(1, args.mvs_max_attempts + 1):
        pending = [d for d in list_scene_dirs(args.out_root)
                   if os.path.isdir(os.path.join(d, "colmap/dense"))
                   and not os.path.isfile(
                       os.path.join(d, "colmap/dense/fused.ply"))]
        if not pending:
            print("  nothing pending", flush=True)
            return
        print(f"\nMVS attempt {attempt}/{args.mvs_max_attempts}: "
              f"{len(pending)} scenes pending\n", flush=True)
        try:
            subprocess.run(
                [sys.executable, os.path.join(HERE, "mvs_batch.py"),
                 "--inputs_root", args.out_root,
                 "--gpus", args.gpus,
                 "--max_image_size", str(args.mvs_max_image_size)],
                check=False)  # mvs_batch survives individual scene crashes
        except Exception as e:
            print(f"  mvs_batch crashed: {e}", flush=True)


def stage_summary(args):
    banner("Final summary")
    n_cams = len([c for c in args.cams.split(",") if c.strip()])
    expected = args.window_len * n_cams
    print(f"{'scene':<90s}  {'sfm':<6}  {'undist':<7}  {'mvs':<5}  detail")
    print("-" * 150)
    all_ok = 0
    sfm_fail = 0
    mvs_fail = 0
    for d in list_scene_dirs(args.out_root):
        name = os.path.basename(d)
        sfm = sfm_quality_ok(d, expected, args.sfm_max_err)
        sfm_tag = "OK" if sfm[0] else "FAIL"
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
            detail = f"sparse_reason={sfm[1]}  dense={n_pts}pts ({size_mb:.1f} MB)"
        else:
            mvs_tag = "FAIL" if und_tag == "OK" else "-"
            detail = f"sparse_reason={sfm[1]}"
        if sfm_tag == "OK" and mvs_tag == "OK":
            all_ok += 1
        elif sfm_tag != "OK":
            sfm_fail += 1
        elif mvs_tag == "FAIL":
            mvs_fail += 1
        print(f"{name:<90s}  {sfm_tag:<6}  {und_tag:<7}  {mvs_tag:<5}  {detail}")
    print()
    print(f"Totals: all_ok={all_ok}  sfm_failed={sfm_fail}  mvs_failed={mvs_fail}")
    print(f"\nABSOLUTE ROOT: {os.path.abspath(args.out_root)}")


# -------------------- main --------------------

STAGES = ["extract", "sfm", "check", "undistort", "mvs"]


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--selection", required=True,
                    help="JSON: {scene_name: start_frame}")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--window_len", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--cams", default="0,1,2,3,4",
                    help="0=FRONT 1=FL 2=FR 3=SL 4=SR")
    ap.add_argument("--sfm_gpu", default="0",
                    help="single GPU index for the sequential SfM stage")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7",
                    help="GPU pool for parallel MVS")
    ap.add_argument("--mvs_max_image_size", type=int, default=1600)
    ap.add_argument("--sfm_max_attempts", type=int, default=2,
                    help="retry SfM this many times before giving up on a scene")
    ap.add_argument("--sfm_max_err", type=float, default=5.0,
                    help="reproj error threshold (px) for SfM 'ok'")
    ap.add_argument("--mvs_max_attempts", type=int, default=2,
                    help="re-run mvs_batch up to this many waves on missing scenes")
    ap.add_argument("--overwrite_extract", action="store_true")
    ap.add_argument("--start_from", choices=STAGES, default="extract")
    ap.add_argument("--stop_after", choices=STAGES, default="mvs")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    start_idx = STAGES.index(args.start_from)
    stop_idx = STAGES.index(args.stop_after)
    if stop_idx < start_idx:
        sys.exit(f"--stop_after {args.stop_after} precedes "
                 f"--start_from {args.start_from}")
    fn_for = {"extract": stage_extract, "sfm": stage_sfm,
              "check": stage_check, "undistort": stage_undistort,
              "mvs": stage_mvs}
    t0 = time.time()
    for i in range(start_idx, stop_idx + 1):
        fn_for[STAGES[i]](args)
    if stop_idx >= STAGES.index("mvs") - 0:
        # Always show final summary at the end if we ran to mvs (or stopped at it)
        if STAGES[stop_idx] in ("mvs",):
            stage_summary(args)
    elif STAGES[stop_idx] in ("undistort", "check"):
        # still useful to see status
        stage_summary(args)
    print(f"\nTotal wall time: {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
