#!/usr/bin/env python3
"""Build per-scene 3DGS-ready dataset roots, one with SfM-only init and one
with MVS dense init, sharing images/cameras/images via symlinks so we only
duplicate the small points file.

3DGS (Inria gaussian-splatting) reads:
  <scene>/images/
  <scene>/sparse/0/cameras.bin
  <scene>/sparse/0/images.bin
  <scene>/sparse/0/points3D.{ply|bin|txt}
    - if .ply exists, it is used directly for Gaussian initialization
      (must have float xyz, float nx ny nz, uchar red green blue)
    - if only .bin exists, it is auto-converted to .ply with zero normals
      on first training run

So:
  sparseinit/  has only points3D.bin -> 3DGS auto-creates points3D.ply
               (sparse SfM points after undistortion)
  mvsinit/     has points3D.ply = copy of fused.ply (MVS dense cloud).
               fused.ply already has the required 9-column layout.
"""
import argparse
import os
import shutil


def link(src, dst):
    src = os.path.abspath(src)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def build_one(scene_root, out_root, scene_name):
    dense = os.path.join(scene_root, "colmap", "dense")
    fused = os.path.join(dense, "fused.ply")
    sparse_bin = os.path.join(dense, "sparse", "0", "points3D.bin")
    cams_bin = os.path.join(dense, "sparse", "0", "cameras.bin")
    imgs_bin = os.path.join(dense, "sparse", "0", "images.bin")
    imgs_dir = os.path.join(dense, "images")
    for p in [fused, sparse_bin, cams_bin, imgs_bin, imgs_dir]:
        if not os.path.exists(p):
            return f"skip {scene_name}: missing {p}"

    def setup(out_dir, ply_src):
        os.makedirs(os.path.join(out_dir, "sparse", "0"), exist_ok=True)
        link(imgs_dir, os.path.join(out_dir, "images"))
        link(cams_bin, os.path.join(out_dir, "sparse", "0", "cameras.bin"))
        link(imgs_bin, os.path.join(out_dir, "sparse", "0", "images.bin"))
        # points3D file:
        #  - sparseinit: symlink points3D.bin -> 3DGS will create points3D.ply
        #    next to it (a real file in this dir, NOT in the shared dense dir)
        #  - mvsinit: copy fused.ply to points3D.ply -> used directly
        if ply_src == "sparse":
            link(sparse_bin, os.path.join(out_dir, "sparse", "0", "points3D.bin"))
        else:
            shutil.copy2(fused, os.path.join(out_dir, "sparse", "0", "points3D.ply"))
            # Keep the sparse .bin around too in case some tool wants it;
            # 3DGS prefers .ply when present, so this is benign.
            link(sparse_bin, os.path.join(out_dir, "sparse", "0", "points3D.bin"))

    sparse_out = os.path.join(out_root, scene_name + "_sparseinit")
    mvs_out = os.path.join(out_root, scene_name + "_mvsinit")
    setup(sparse_out, "sparse")
    setup(mvs_out, "mvs")
    return f"ok {scene_name}: -> {sparse_out}  /  {mvs_out}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colmap_input_root", default="data/waymo/colmap_input")
    ap.add_argument("--out_root", default="data/waymo/3dgs_input")
    args = ap.parse_args()

    scenes = sorted(d for d in os.listdir(args.colmap_input_root)
                    if d.startswith("segment-"))
    print(f"Found {len(scenes)} scenes")
    for s in scenes:
        msg = build_one(os.path.join(args.colmap_input_root, s),
                        args.out_root, s)
        print(msg)


if __name__ == "__main__":
    main()
