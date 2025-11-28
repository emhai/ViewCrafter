from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def load_poses_bounds(path):
    pb = np.load(path)  # (N,17)
    poses = pb[:, :-2].reshape(-1, 3, 5)  # (N,3,5)
    bounds = pb[:, -2:]                   # (N,2)
    return poses, bounds

def llff_3x5_to_c2w(poses_3x5):
    """Convert LLFF 3x5 to 4x4 c2w."""
    N = poses_3x5.shape[0]
    c2w = np.zeros((N, 4, 4))
    A = poses_3x5[:, :, :4]
    c2w[:, :3, 0] = A[:, :, 1]  # right
    c2w[:, :3, 1] = A[:, :, 0]  # up
    c2w[:, :3, 2] = A[:, :, 2]  # forward
    c2w[:, :3, 3] = A[:, :, 3]  # translation
    c2w[:, 3, 3] = 1.0
    return c2w

def autoscale_equal_3d(ax, pts, pad=0.1):
    """Equal aspect for 3D axes based on (N,3) points."""
    if len(pts) == 0:
        return
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    centers = (mins + maxs) * 0.5
    spans = (maxs - mins)
    radius = max(spans.max() * 0.5, 1e-6) * (1 + pad)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)

def visualize_poses_bounds(path):
    poses_3x5, _ = load_poses_bounds(str(path))
    c2w_all = llff_3x5_to_c2w(poses_3x5)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Camera centers & axes (IDs from poses_bounds.npy rows)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    centers = []
    for i, c2w in enumerate(c2w_all):
        C = c2w[:3, 3]
        R = c2w[:3, :3]
        centers.append(C)

        # Draw small orientation axes at each camera
        axes_len = 0.05 * max(1.0, np.linalg.norm(c2w_all[:, :3, 3]).mean() + 1e-6)
        # X (right), Y (up), Z (forward/back depending on pipeline)
        ax.quiver(C[0], C[1], C[2], R[0, 0], R[1, 0], R[2, 0], length=axes_len, normalize=True)  # X
        ax.quiver(C[0], C[1], C[2], R[0, 1], R[1, 1], R[2, 1], length=axes_len, normalize=True)  # Y
        ax.quiver(C[0], C[1], C[2], R[0, 2], R[1, 2], R[2, 2], length=axes_len, normalize=True)  # Z

        # Label with ID (row index)
        ax.text(C[0], C[1], C[2], f"{i}", fontsize=12)

    centers = np.stack(centers, axis=0)
    autoscale_equal_3d(ax, centers, pad=0.15)
    ax.view_init(elev=20, azim=-60)
    plt.show()

def main():
    path_to_dataset = Path("/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/4dgs_dataset/flame_salmon_1/poses_bounds.npy")
    visualize_poses_bounds(path_to_dataset)


if __name__ == '__main__':
    main()