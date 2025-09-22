from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

import torch.nn.functional as F

def save_masks(mask_list, save_dir, visualize=True, save=True):
    save_dir = Path(save_dir)
    save_dir.mkdir()

    for i, msk in enumerate(mask_list):
        if isinstance(msk, torch.Tensor):
            msk_np = msk.cpu().detach().numpy()
        else:
            msk_np = msk

        msk_img = (msk_np * 255).astype(np.uint8)

        if save:
            mask_img = Image.fromarray(msk_img)
            mask_img.save(save_dir / f"mask_{i}.png")

        if visualize:
            plt.imshow(msk_np, cmap='gray')
            plt.title(f"Mask {i}")
            plt.show()

def save_depth(depth_list, save_dir, visualize=True, save=True):
    save_dir = Path(save_dir)
    save_dir.mkdir()

    for i, dpt in enumerate(depth_list):
        dpt_np = dpt.cpu().detach().numpy()
        dpt_norm = ((dpt_np - dpt_np.min()) / (dpt_np.ptp() + 1e-8) * 255).astype(np.uint8)

        if save:
            depth_img = Image.fromarray(dpt_norm)
            depth_img.save(save_dir / f"depth_{i}.png")

        if visualize:
            plt.imshow(dpt_np, cmap='plasma')
            plt.title(f"Depth Map {i}")
            plt.colorbar()
            plt.show()

def visualize_pixel_masks(full_res_mask, image, path, title):

    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(image.numpy())
    axs[0].set_title("Original image")
    axs[0].axis("off")

    axs[1].imshow(full_res_mask.squeeze().numpy(), cmap='gray')
    axs[1].set_title(title)
    axs[1].axis("off")

    mask_resized = F.interpolate(full_res_mask, size=(image.shape[0], image.shape[1]), mode='nearest')  # shape: (1,1,768,1024)

    # Step 2 — squeeze to H,W
    mask_2d = mask_resized.squeeze() # shape: (768,1024)
    axs[2].imshow(image.numpy())  # original image
    axs[2].imshow(mask_2d, cmap='Reds', alpha=0.5)  # transparent red mask
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)

def visualize_masks_horizontal(masks, path, cmap=None):

    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()

    n = masks.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(n * 5, 5))

    for i in range(n):
        axes[i].imshow(masks[i], cmap=cmap)
        axes[i].axis("off")
        axes[i].set_title(f"Mask {i}")

    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


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


def extract_hwf(poses_3x5):
    H = poses_3x5[:, 0, 4]
    W = poses_3x5[:, 1, 4]
    F = poses_3x5[:, 2, 4]
    return H, W, F


def make_camera_rays(H, W, F):
    cx, cy = W * 0.5, H * 0.5
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float64)
    dirs = np.stack([
        (corners[:, 0] - cx) / F,
        (corners[:, 1] - cy) / F,
        np.ones(4)
    ], axis=1)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs


def transform_points(c2w, pts_cam):
    return (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]


def load_poses_bounds(path):
    pb = np.load(path)  # expected shape (N, 17) in LLFF
    poses = pb[:, :-2].reshape(-1, 3, 5)  # (N,3,5)
    bounds = pb[:, -2:]                   # (N,2) [near, far] (unused here)
    return poses, bounds

def llff_3x5_to_c2w(poses_3x5):
    """
    Convert LLFF stored pose (3x5) into 4x4 camera-to-world.
    LLFF packs columns as [y, x, z, t, hwf], so swap x/y back.
    """
    N = poses_3x5.shape[0]
    c2w = np.zeros((N, 4, 4), dtype=np.float64)
    A = poses_3x5[:, :, :4]  # (N,3,4)
    c2w[:, :3, 0] = A[:, :, 1]  # right (x)
    c2w[:, :3, 1] = A[:, :, 0]  # up (y)
    c2w[:, :3, 2] = A[:, :, 2]  # forward/back (z)
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
    poses_3x5, _ = load_poses_bounds(path)
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

def read_names(fs):
    node = fs.getNode("names")
    names = []
    if not node.empty():
        for i in range(int(node.size())):
            names.append(node.at(i).string())
    else:
        # fallback: try sequential ids until missing
        i = 0
        while True:
            key_rot = f"Rot_{i:02d}"
            if fs.getNode(key_rot).empty():
                break
            names.append(f"{i:02d}")
            i += 1
    return names

def read_mat(fs, key):
    node = fs.getNode(key)
    if node.empty():
        return None
    return node.mat()  # returns a numpy array

def visualize_extrinsics_yaml(path):
    # Load OpenCV YAML
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Failed to open {path}")

    names = read_names(fs)
    if not names:
        raise RuntimeError("Could not read 'names' or find any Rot_* entries.")

    # Collect camera centers and axes
    centers = []
    axes_world = []  # (R_c2w) for each cam
    kept_names = []

    for nm in names:
        # Many dumps include both R_* (Rodrigues vector) and Rot_* (3x3).
        # We prefer the explicit rotation matrix Rot_*; fall back to R_* if needed.
        Rot = read_mat(fs, f"Rot_{nm}")
        if Rot is None:
            rvec = read_mat(fs, f"R_{nm}")
            if rvec is None:
                print(f"Skipping {nm}: no Rot_{nm} or R_{nm}")
                continue
            Rot, _ = cv2.Rodrigues(rvec.reshape(3))

        T = read_mat(fs, f"T_{nm}")
        if T is None:
            print(f"Skipping {nm}: no T_{nm}")
            continue
        T = T.reshape(3, 1)

        # Interpret as world->camera: x_cam = R x_world + T
        # => camera center in world: C = -R^T T, camera orientation in world: R_c2w = R^T
        R = Rot
        Rt = R.T
        C = (-Rt @ T).reshape(3)
        R_c2w = Rt

        centers.append(C)
        axes_world.append(R_c2w)
        kept_names.append(nm)

    fs.release()

    centers = np.array(centers)  # (N,3)
    axes_world = np.array(axes_world)  # (N,3,3)

    # Plot
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Camera centers & axes (IDs from YAML 'names')")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Choose a reasonable axis length based on scene size
    if len(centers) > 0:
        span = np.ptp(centers, axis=0)
        axes_len = max(span.max() * 0.05, 0.05)
    else:
        axes_len = 0.1

    for i, (C, Rw, nm) in enumerate(zip(centers, axes_world, kept_names)):
        # Draw axes (columns of R_c2w are X, Y, Z in world)
        ax.quiver(C[0], C[1], C[2], Rw[0, 0], Rw[1, 0], Rw[2, 0], length=axes_len, normalize=True)  # X (right)
        ax.quiver(C[0], C[1], C[2], Rw[0, 1], Rw[1, 1], Rw[2, 1], length=axes_len, normalize=True)  # Y (up)
        ax.quiver(C[0], C[1], C[2], Rw[0, 2], Rw[1, 2], Rw[2, 2], length=axes_len, normalize=True)  # Z (forward)
        ax.text(C[0], C[1], C[2], nm,
                fontsize=6,
                color="black",
                zorder=10,  # ensure text is drawn last / on top
                clip_on=False)

    # Equal aspect
    if len(centers) > 0:
        mins = centers.min(axis=0)
        maxs = centers.max(axis=0)
        ctr = (mins + maxs) * 0.5
        size = (maxs - mins).max()
        size = max(size, 1e-6)
        pad = 0.15 * size
        ax.set_xlim(ctr[0] - size / 2 - pad, ctr[0] + size / 2 + pad)
        ax.set_ylim(ctr[1] - size / 2 - pad, ctr[1] + size / 2 + pad)
        ax.set_zlim(ctr[2] - size / 2 - pad, ctr[2] + size / 2 + pad)

    ax.view_init(elev=20, azim=-60)
    plt.show()


def main():
    path = "/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/SelfCap/corgi-release/optimized/extri.yml"
    visualize_extrinsics_yaml(path)

if __name__ == '__main__':
    main()