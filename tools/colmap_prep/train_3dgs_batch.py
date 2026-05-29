#!/usr/bin/env python3
"""Drive parallel 3DGS training across a fixed GPU pool.

For each scene root under --inputs_root, run

    python <gs_repo>/train.py -s <scene_root> -m <outputs_root>/<scene>/ \
        --iterations N --disable_viewer --eval --quiet

with CUDA_VISIBLE_DEVICES=<one GPU>. The pool keeps exactly one training
process per GPU; as jobs finish, the next pending scene is dispatched to the
freed GPU, so no GPU sits idle.

Run this script from inside the gaussian-splatting conda env so sys.executable
resolves to that env's python (with torch + diff_gaussian_rasterization).

Per-run logs: <outputs_root>/<scene>/train.log
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
    ap.add_argument("--inputs_root", default="data/waymo/3dgs_input")
    ap.add_argument("--outputs_root", default="data/waymo/3dgs_output")
    ap.add_argument("--gs_repo", required=True,
                    help="path to gaussian-splatting checkout (with train.py)")
    ap.add_argument("--gpus", required=True,
                    help="comma-separated GPU indices, e.g. '4,5,6,7'")
    ap.add_argument("--iterations", type=int, default=7000)
    ap.add_argument("--filter", default=None,
                    help="only run scene dirs whose name contains this string "
                         "(use 'sparseinit' or 'mvsinit' to run one half)")
    ap.add_argument("--extra_args", default="",
                    help="extra args appended verbatim to train.py command")
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    inputs_root = os.path.abspath(args.inputs_root)
    outputs_root = os.path.abspath(args.outputs_root)
    train_py = os.path.join(os.path.abspath(args.gs_repo), "train.py")
    if not os.path.exists(train_py):
        sys.exit(f"no train.py at {train_py}")
    os.makedirs(outputs_root, exist_ok=True)

    print(f"GPUs: {gpus}   iterations: {args.iterations}", flush=True)

    scenes = sorted(d for d in os.listdir(inputs_root)
                    if d.startswith("segment-")
                    and (args.filter is None or args.filter in d))
    queue = deque()
    for s in scenes:
        done = os.path.join(outputs_root, s, "point_cloud",
                            f"iteration_{args.iterations}", "point_cloud.ply")
        if os.path.exists(done):
            print(f"skip {s} (iter_{args.iterations} already exists)", flush=True)
            continue
        queue.append(s)
    print(f"Total scenes: {len(scenes)}   pending: {len(queue)}", flush=True)
    if not queue:
        return

    # workers[gpu] = (Popen, name, log_fp, t_start) or None
    workers = {g: None for g in gpus}

    def launch(gpu, name):
        src = os.path.join(inputs_root, name)
        out = os.path.join(outputs_root, name)
        os.makedirs(out, exist_ok=True)
        log_path = os.path.join(out, "train.log")
        cmd = [sys.executable, train_py,
               "-s", src, "-m", out,
               "--iterations", str(args.iterations),
               "--disable_viewer", "--eval", "--quiet"]
        if args.extra_args:
            cmd += args.extra_args.split()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        fp = open(log_path, "w")
        proc = subprocess.Popen(cmd, cwd=os.path.abspath(args.gs_repo),
                                env=env, stdout=fp, stderr=subprocess.STDOUT)
        print(f"[GPU {gpu}] LAUNCH pid={proc.pid}  {name}", flush=True)
        return (proc, name, fp, time.time())

    def cleanup(*_):
        print("\n[!] caught signal, terminating children...", flush=True)
        for gpu, w in workers.items():
            if w is None:
                continue
            proc, name, fp, _ = w
            try:
                proc.terminate()
            except Exception:
                pass
        deadline = time.time() + 5
        for w in workers.values():
            if w is None:
                continue
            proc = w[0]
            remaining = max(0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                proc.kill()
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
            print(f"[GPU {gpu}] FINISH {name}  {status}  ({dt:.1f} min)",
                  flush=True)
            workers[gpu] = None

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
