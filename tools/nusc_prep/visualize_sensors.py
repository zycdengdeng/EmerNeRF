#!/usr/bin/env python3
"""Visualize Waymo and nuScenes camera rigs in 3D: positions, orientations,
and frustums. Both rigs are drawn in a FLU-equivalent ego frame (x=forward,
y=left, z=up) so the layouts are directly comparable.

Output:
  <out_dir>/sensor_waymo.png        Waymo (5 cams, original FLU camera frame)
  <out_dir>/sensor_nuscenes.png     nuScenes (6 cams, OpenCV RDF camera frame
                                     -> ego converted from nuScenes ego
                                     [+X right, +Y forward, +Z up] to FLU)
  <out_dir>/sensor_both.png         side-by-side comparison
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CAM_COLOR = {
    "FRONT":      "#1f77b4",
    "FRONT_LEFT": "#17becf",
    "FRONT_RIGHT":"#ff7f0e",
    "SIDE_LEFT":  "#9467bd",
    "SIDE_RIGHT": "#d62728",
    "BACK_LEFT":  "#9467bd",
    "BACK_RIGHT": "#d62728",
    "BACK":       "#7f7f7f",
}


# ------------------- frustum corner unprojection -------------------

def unproject_waymo_flu(u, v, fx, fy, cx, cy, D):
    """Waymo camera frame: x = optical axis (forward), y = left, z = up.
    Pixel u increases to the right, v down. So:
      u = -fx * (Y/X) + cx -> Y = -(u - cx) * X / fx
      v = -fy * (Z/X) + cy -> Z = -(v - cy) * X / fy
    With depth D = X."""
    X = D
    Y = -(u - cx) * D / fx
    Z = -(v - cy) * D / fy
    return np.array([X, Y, Z])


def unproject_opencv_rdf(u, v, fx, fy, cx, cy, D):
    """Standard OpenCV pinhole: z = optical axis, x right, y down."""
    X = (u - cx) * D / fx
    Y = (v - cy) * D / fy
    Z = D
    return np.array([X, Y, Z])


def frustum_corners(W, H, fx, fy, cx, cy, D, convention):
    f = unproject_waymo_flu if convention == "waymo_flu" else unproject_opencv_rdf
    return [f(0, 0, fx, fy, cx, cy, D),
            f(W, 0, fx, fy, cx, cy, D),
            f(W, H, fx, fy, cx, cy, D),
            f(0, H, fx, fy, cx, cy, D)]


# ------------------- draw one camera -------------------

def draw_camera(ax, T_cam_to_ego, corners_cam, label, color):
    origin = T_cam_to_ego[:3, 3]
    R = T_cam_to_ego[:3, :3]
    corners_ego = [R @ c + origin for c in corners_cam]
    # rays from origin
    for c in corners_ego:
        ax.plot([origin[0], c[0]], [origin[1], c[1]], [origin[2], c[2]],
                color=color, linewidth=0.8, alpha=0.6)
    # far-plane rectangle
    rect = np.array(corners_ego + [corners_ego[0]])
    ax.plot(rect[:, 0], rect[:, 1], rect[:, 2], color=color, linewidth=1.4)
    # camera origin dot + label
    ax.scatter([origin[0]], [origin[1]], [origin[2]],
               color=color, s=40, edgecolors="k", linewidths=0.5, zorder=5)
    ax.text(origin[0], origin[1], origin[2] + 0.3, label,
            color=color, fontsize=8, weight="bold")


# ------------------- ego car schematic -------------------

def draw_ego(ax, length=4.7, width=1.9, height=1.6, color="lightgray"):
    """Wireframe rectangular box centred at origin with +x forward."""
    L, W, H = length / 2, width / 2, height
    pts = np.array([
        [ L,  W, 0], [ L, -W, 0], [-L, -W, 0], [-L,  W, 0],
        [ L,  W, H], [ L, -W, H], [-L, -W, H], [-L,  W, H],
    ])
    # bottom + top loops
    for base in (0, 4):
        loop = np.r_[base:base + 4, [base]]
        ax.plot(pts[loop, 0], pts[loop, 1], pts[loop, 2],
                color=color, linewidth=0.7)
    # vertical edges
    for i in range(4):
        ax.plot([pts[i, 0], pts[i + 4, 0]],
                [pts[i, 1], pts[i + 4, 1]],
                [pts[i, 2], pts[i + 4, 2]],
                color=color, linewidth=0.7)
    # forward arrow
    ax.quiver(L + 0.3, 0, H / 2, 1.2, 0, 0, color="red",
              arrow_length_ratio=0.4, linewidth=2)
    ax.text(L + 1.6, 0, H / 2, "forward", color="red", fontsize=8)


def axes_setup(ax, title, lim=12):
    ax.set_xlabel("X / forward (m)")
    ax.set_ylabel("Y / left (m)")
    ax.set_zlabel("Z / up (m)")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-1, 5)
    ax.set_box_aspect([2, 2, 0.6])
    # nicer view
    ax.view_init(elev=22, azim=-70)


# ------------------- per-dataset visualization -------------------

def viz_waymo(ax, meta_path, depth=8.0):
    meta = json.load(open(meta_path))
    draw_ego(ax)
    for cam in meta["cameras"]:
        fx, fy, cx, cy = cam["intrinsic"][0:4]
        W, H = cam["width"], cam["height"]
        T = np.array(cam["extrinsic_cam_to_vehicle_flu"])
        corners = frustum_corners(W, H, fx, fy, cx, cy, depth, "waymo_flu")
        draw_camera(ax, T, corners, cam["label"],
                    CAM_COLOR.get(cam["label"], "black"))
    axes_setup(ax, f"Waymo  (5 cams, vehicle frame FLU; frustums @ {depth} m)")


# nuScenes ego -> FLU rotation:
# nuScenes ego: +X right, +Y forward, +Z up
# FLU:          +X forward, +Y left, +Z up
# Therefore: [x_flu, y_flu, z_flu] = [y_n, -x_n, z_n]
R_NUSC_TO_FLU = np.eye(4)
R_NUSC_TO_FLU[:3, :3] = np.array([[0, 1, 0],
                                  [-1, 0, 0],
                                  [0, 0, 1]], dtype=np.float64)


def viz_nuscenes(ax, scene_dir, depth=8.0):
    extr = os.path.join(scene_dir, "extrinsics")
    intr = os.path.join(scene_dir, "intrinsics")
    ego2w = np.loadtxt(os.path.join(scene_dir, "lidar_pose", "000.txt")
                       ).reshape(4, 4)
    w2ego_nusc = np.linalg.inv(ego2w)

    labels = {0: "FRONT", 1: "FRONT_LEFT", 2: "FRONT_RIGHT",
              3: "BACK_LEFT", 4: "BACK_RIGHT", 5: "BACK"}
    draw_ego(ax)
    for cam in range(6):
        v = np.loadtxt(os.path.join(intr, f"{cam}.txt")).reshape(-1)
        fx, fy, cx, cy = float(v[0]), float(v[1]), float(v[2]), float(v[3])
        cam2w = np.loadtxt(os.path.join(extr, f"000_{cam}.txt")
                          ).reshape(4, 4)
        # cam -> nuScenes_ego -> FLU
        T_cam_to_ego_nusc = w2ego_nusc @ cam2w
        T_cam_to_ego_flu = R_NUSC_TO_FLU @ T_cam_to_ego_nusc
        corners = frustum_corners(1600, 900, fx, fy, cx, cy, depth, "opencv_rdf")
        draw_camera(ax, T_cam_to_ego_flu, corners, labels[cam],
                    CAM_COLOR.get(labels[cam], "black"))
    axes_setup(ax, f"nuScenes  (6 cams, ego frame->FLU; frustums @ {depth} m)")


# ------------------- main -------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--waymo_meta",
        default="data/waymo/colmap_input_5cam/"
                "segment-12879640240483815315_5852_605_5872_605_with_camera_labels/"
                "selection_meta.json")
    ap.add_argument("--nuscenes_scene",
        default="/mnt/public_datasets/nuscenes_10Hz/trainval/000")
    ap.add_argument("--out_dir", default="/mnt/zihanw/gsnet_nusc/viz")
    ap.add_argument("--depth", type=float, default=8.0,
                    help="frustum far-plane depth in metres")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    have_waymo = os.path.isfile(args.waymo_meta)
    have_nusc = os.path.isdir(args.nuscenes_scene)
    if not have_waymo:
        print(f"[warn] waymo meta not found: {args.waymo_meta}")
    if not have_nusc:
        print(f"[warn] nuscenes scene not found: {args.nuscenes_scene}")

    # Standalone Waymo
    if have_waymo:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        viz_waymo(ax, args.waymo_meta, args.depth)
        out = os.path.join(args.out_dir, "sensor_waymo.png")
        plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()
        print(f"wrote {out}")

    # Standalone nuScenes
    if have_nusc:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        viz_nuscenes(ax, args.nuscenes_scene, args.depth)
        out = os.path.join(args.out_dir, "sensor_nuscenes.png")
        plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()
        print(f"wrote {out}")

    # Side-by-side
    if have_waymo and have_nusc:
        fig = plt.figure(figsize=(18, 8))
        ax1 = fig.add_subplot(121, projection="3d")
        viz_waymo(ax1, args.waymo_meta, args.depth)
        ax2 = fig.add_subplot(122, projection="3d")
        viz_nuscenes(ax2, args.nuscenes_scene, args.depth)
        out = os.path.join(args.out_dir, "sensor_both.png")
        plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
