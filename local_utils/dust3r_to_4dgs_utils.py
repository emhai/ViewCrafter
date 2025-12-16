from pathlib import Path
import os
import struct
import numpy as np


def write_cameras_bin(path, cameras):
    """
    cameras: list of dicts:
      {id, model_id, width, height, params(np.ndarray)}
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam in cameras:
            f.write(struct.pack("<i", cam["id"]))
            f.write(struct.pack("<i", cam["model_id"]))  # e.g. 1=PINHOLE
            f.write(struct.pack("<Q", cam["width"]))
            f.write(struct.pack("<Q", cam["height"]))
            params = cam["params"].astype(np.float64).ravel()
            f.write(struct.pack("<Q", len(params)))
            f.write(struct.pack("<" + "d"*len(params), *params))

def write_images_bin(path, images):
    """
    images: list of dicts:
      {id, qvec(4), tvec(3), camera_id, name(str)}
      points2D can be empty (0 points) for initialization.
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for im in images:
            f.write(struct.pack("<i", im["id"]))
            q = im["qvec"].astype(np.float64).ravel()
            t = im["tvec"].astype(np.float64).ravel()
            f.write(struct.pack("<dddd", *q))
            f.write(struct.pack("<ddd", *t))
            f.write(struct.pack("<i", im["camera_id"]))
            f.write(im["name"].encode("utf-8") + b"\x00")

            # Write 2D points: we write zero, so no track/associations needed
            f.write(struct.pack("<Q", 0))  # num_points2D

def write_points3D_bin(path, points_xyz, points_rgb):
    """
    points_xyz: (M,3) float
    points_rgb: (M,3) uint8
    Minimal COLMAP points3D.bin: tracks can be empty.
    """
    M = points_xyz.shape[0]
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", M))
        for pid in range(M):
            X, Y, Z = points_xyz[pid].astype(np.float64)
            R, G, B = points_rgb[pid].astype(np.uint8)
            error = 0.0
            f.write(struct.pack("<Q", pid + 1))          # point3D_id (uint64)
            f.write(struct.pack("<ddd", X, Y, Z))        # xyz
            f.write(struct.pack("<BBB", int(R), int(G), int(B)))  # rgb
            f.write(struct.pack("<d", error))            # reproj error
            f.write(struct.pack("<Q", 0))                # track length = 0

# ----------------------------
# Helpers
# ----------------------------
def rotmat_to_qvec(R):
    """Convert rotation matrix to COLMAP qvec (w, x, y, z)."""
    # robust method
    K = np.array([
        [R[0,0]-R[1,1]-R[2,2], R[1,0]+R[0,1],       R[2,0]+R[0,2],       R[1,2]-R[2,1]],
        [R[1,0]+R[0,1],       R[1,1]-R[0,0]-R[2,2], R[2,1]+R[1,2],       R[2,0]-R[0,2]],
        [R[2,0]+R[0,2],       R[2,1]+R[1,2],       R[2,2]-R[0,0]-R[1,1], R[0,1]-R[1,0]],
        [R[1,2]-R[2,1],       R[2,0]-R[0,2],       R[0,1]-R[1,0],       R[0,0]+R[1,1]+R[2,2]]
    ], dtype=np.float64) / 3.0
    w, V = np.linalg.eigh(K)
    q = V[:, np.argmax(w)]
    # q is [x,y,z,w] in this construction; reorder to [w,x,y,z]
    qvec = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def write_ply(path, xyz, rgb):
    """ASCII PLY."""
    assert xyz.shape[0] == rgb.shape[0]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, rgb):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")

def compute_bounds_for_view(c2w, points_world, near_q=0.001, far_q=0.999):
    """Compute near/far from fused world points using robust quantiles in camera-z."""
    w2c = np.linalg.inv(c2w)
    R = w2c[:3,:3]
    t = w2c[:3,3]
    z = (points_world @ R.T + t)[..., 2]
    z = z[z > 1e-6]
    if z.size < 100:
        return 0.1, 10.0  # fallback
    near = float(np.quantile(z, near_q))
    far  = float(np.quantile(z, far_q))
    return near, far

def make_llff_pose_3x5(c2w, H, W, focal):
    """
    LLFF expects c2w with columns [down, right, backwards]. :contentReference[oaicite:4]{index=4}
    If your camera basis is (right=x, down=y, forward=z), then:
      down     = +y
      right    = +x
      backwards= -z
    So reorder columns to [y, x, -z].
    """
    R = c2w[:3,:3]
    t = c2w[:3,3:4]
    R_llff = np.stack([R[:,1], R[:,0], -R[:,2]], axis=1)
    pose_3x4 = np.concatenate([R_llff, t], axis=1)
    hwf = np.array([[H],[W],[focal]], dtype=np.float64)
    pose_3x5 = np.concatenate([pose_3x4, hwf], axis=1)
    return pose_3x5

# ----------------------------
# Main export
# ----------------------------
def export_for_4dgs(
    out_dir,
    c2w_list, K_list, img_hw_list,
    image_names,
    points_xyz, points_rgb
):
    os.makedirs(out_dir, exist_ok=True)
    sparse_dir = os.path.join(out_dir, "sparse_")
    os.makedirs(sparse_dir, exist_ok=True)

    # 1) cameras.bin (use one camera model per view if intrinsics differ)
    cameras = []
    for i, (K, (H,W)) in enumerate(zip(K_list, img_hw_list), start=1):
        fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
        cameras.append({
            "id": i,
            "model_id": 1,  # PINHOLE in COLMAP
            "width": int(W),
            "height": int(H),
            "params": np.array([fx, fy, cx, cy], dtype=np.float64),
        })
    write_cameras_bin(os.path.join(sparse_dir, "cameras.bin"), cameras)

    # 2) images.bin
    images = []
    for i, (c2w, name) in enumerate(zip(c2w_list, image_names), start=1):
        w2c = np.linalg.inv(c2w)
        R = w2c[:3,:3]
        t = w2c[:3,3]
        qvec = rotmat_to_qvec(R)
        images.append({
            "id": i,
            "qvec": qvec,
            "tvec": t.astype(np.float64),
            "camera_id": i,   # tie each image to its camera
            "name": name,
        })
    write_images_bin(os.path.join(sparse_dir, "images.bin"), images)

    # 3) points3D.bin (optional but recommended)
    write_points3D_bin(os.path.join(sparse_dir, "points3D.bin"),
                       points_xyz.astype(np.float64),
                       points_rgb.astype(np.uint8))

    # 4) points3D_multipleview.ply
    write_ply(os.path.join(out_dir, "points3D_multipleview.ply"),
              points_xyz.astype(np.float32),
              points_rgb.astype(np.uint8))

    # 5) poses_bounds_multipleview.npy (LLFF Nx17)
    N = len(c2w_list)
    poses_bounds = np.zeros((N, 17), dtype=np.float64)
    for i in range(N):
        H, W = img_hw_list[i]
        K = K_list[i]
        focal = float(0.5 * (K[0,0] + K[1,1]))  # average fx/fy
        pose_3x5 = make_llff_pose_3x5(c2w_list[i], H, W, focal)
        near, far = compute_bounds_for_view(c2w_list[i], points_xyz)
        row15 = pose_3x5.reshape(-1)  # 15
        poses_bounds[i, :15] = row15
        poses_bounds[i, 15] = near
        poses_bounds[i, 16] = far

    np.save(os.path.join(out_dir, "poses_bounds_multipleview.npy"), poses_bounds)

    print("Wrote:", out_dir)
    print(" - sparse_/cameras.bin, images.bin, points3D.bin")
    print(" - points3D_multipleview.ply")
    print(" - poses_bounds_multipleview.npy")

def main():
    path = Path("/media/emmahaidacher/Volume/DATASETS/INTERNET/espresso_short/4dgs_1_cam_downsampled")

if __name__ == "__main__":
    main()