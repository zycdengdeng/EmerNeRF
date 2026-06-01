#!/usr/bin/env python3
"""Drive parallel COLMAP MVS across a GPU pool. One scene per GPU at a time;
as soon as one scene finishes, the next pending one is dispatched to the
freed GPU.

Per-run log: <scene_root>/mvs.log

Run this from inside the waymoprep conda env (only needs python stdlib).
The actual colmap binary lookup happens in the child via PATH, so either
export PATH first or pass --colmap_bin <absolute path>.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from collections import deque


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs_root", default="data/waymo/colmap_input_5cam")
    ap.add_argument("--gpus", required=True,
                    help="comma-separated GPU indices, e.g. '0,1,2,3,4,5,6,7'")
    ap.add_argument("--colmap_bin", default="colmap")
    ap.add_argument("--max_image_size", type=int, default=1600)
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    inputs_root = os.path.abspath(args.inputs_root)

    scenes = sorted(d for d in os.listdir(inputs_root)
                    if d.startswith("segment-"))
    queue = deque()
    for s in scenes:
        scene_dir = os.path.join(inputs_root, s)
        fused = os.path.join(scene_dir, "colmap", "dense", "fused.ply")
        dense = os.path.join(scene_dir, "colmap", "dense")
        if os.path.exists(fused):
            print(f"skip {s} (fused.ply exists)", flush=True)
            continue
        if not os.path.isdir(dense):
            print(f"skip {s} (no dense/; run undistort_for_3dgs first)",
                  flush=True)
            continue
        queue.append(s)
    print(f"GPUs: {gpus}   pending: {len(queue)}/{len(scenes)}", flush=True)
    if not queue:
        return

    workers = {g: None for g in gpus}
    here = os.path.dirname(os.path.abspath(__file__))
    run_mvs = os.path.join(here, "run_mvs.py")

    def launch(gpu, name):
        scene_dir = os.path.join(inputs_root, name)
        log = os.path.join(scene_dir, "mvs.log")
        cmd = [sys.executable, run_mvs,
               "--scene_root", scene_dir,
               "--colmap_bin", args.colmap_bin,
               "--max_image_size", str(args.max_image_size)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        fp = open(log, "w")
        proc = subprocess.Popen(cmd, env=env, stdout=fp,
                                stderr=subprocess.STDOUT)
        print(f"[GPU {gpu}] LAUNCH pid={proc.pid}  {name}", flush=True)
        return (proc, name, fp, time.time())

    def cleanup(*_):
        print("\n[!] caught signal, terminating MVS children...", flush=True)
        for gpu, w in workers.items():
            if w is None:
                continue
            try:
                w[0].terminate()
            except Exception:
                pass
        deadline = time.time() + 5
        for w in workers.values():
            if w is None:
                continue
            remaining = max(0, deadline - time.time())
            try:
                w[0].wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                w[0].kill()
        sys.exit(1)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    while queue or any(workers[g] for g in gpus):
        for gpu in gpus:
            if workers[gpu] is None and queue:
                workers[gpu] = launch(gpu, queue.popleft())
        time.sleep(5)
        for gpu in gpus:
            if workers[gpu] is None:
                continue
            proc, name, fp, t0 = workers[gpu]
            rc = proc.poll()
            if rc is None:
                continue
            fp.close()
            dt = (time.time() - t0) / 60.0
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"[GPU {gpu}] FINISH {name}  {status}  ({dt:.1f}min)",
                  flush=True)
            workers[gpu] = None

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
