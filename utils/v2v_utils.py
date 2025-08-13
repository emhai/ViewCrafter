import os
import shutil
import subprocess

import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
from skimage import data, filters
from torch.ao.nn.quantized.functional import threshold
from torch.utils.tensorboard.summary import video

from configs.v2v_config import *

import matplotlib.pyplot as plt

def save_masks(mask_list, save_dir, visualize=True, save=True):
    os.makedirs(save_dir)
    for i, msk in enumerate(mask_list):
        if isinstance(msk, torch.Tensor):
            msk_np = msk.cpu().detach().numpy()
        else:
            msk_np = msk

        msk_img = (msk_np * 255).astype(np.uint8)

        if save:
            mask_img = Image.fromarray(msk_img)
            mask_img.save(os.path.join(save_dir, f"mask_{i}.png"))

        if visualize:
            plt.imshow(msk_np, cmap='gray')
            plt.title(f"Mask {i}")
            plt.show()

def save_depth(depth_list, save_dir, visualize=True, save=True):
    os.makedirs(save_dir)

    for i, dpt in enumerate(depth_list):
        dpt_np = dpt.cpu().detach().numpy()
        dpt_norm = ((dpt_np - dpt_np.min()) / (dpt_np.ptp() + 1e-8) * 255).astype(np.uint8)

        if save:
            depth_img = Image.fromarray(dpt_norm)
            depth_img.save(os.path.join(save_dir, f"depth_{i}.png"))

        if visualize:
            plt.imshow(dpt_np, cmap='plasma')
            plt.title(f"Depth Map {i}")
            plt.colorbar()
            plt.show()

def extract_frames(video_path, frames_path):
    print(f"Extracting frames from {video_path}")

    outer_path = video_path.split("/")[0: -2]
    stdout_path = os.path.join("/", *outer_path, OUTPUT_LOG_FILE)

    ffmpeg_command = ['ffmpeg', '-i', os.path.join(video_path, video_path), f"{frames_path}/%05d.png"]
    with open(stdout_path, "a") as f:
        subprocess.run(ffmpeg_command, stdout=f, stderr=subprocess.STDOUT)

def create_folder_structure(folders):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print('Created folder:', folder)

def setup_structure(save_path, source_path):
    frames_path = os.path.join(save_path, CAMERA_FRAMES_DIR)
    all_frames_path = os.path.join(save_path, INPUTS_DIR)
    results_path = os.path.join(save_path, RESULTS_DIR)
    cameras_path = os.path.join(save_path, SEPERATED_CAMERAS_DIR)

    all_folders = [frames_path, all_frames_path, results_path, cameras_path]
    create_folder_structure(all_folders)

    # copy video folder
    video_path = os.path.join(save_path, ORIGINAL_VIDEOS_DIR)
    video_num = len(os.listdir(source_path))
    shutil.copytree(source_path, video_path)
    print(f"Copying {video_num} videos from {source_path} to {video_path}")

    # extract frames
    for video in os.listdir(video_path):
        video_name, video_ext = os.path.splitext(video)
        new_path = os.path.join(frames_path, video_name)
        os.makedirs(new_path)
        extract_frames(os.path.join(video_path, video), new_path)


    frame_folders = sorted(os.listdir(frames_path))
    folder_paths = [os.path.join(frames_path, folder) for folder in frame_folders]
    folder_files = [sorted(os.listdir(folder)) for folder in folder_paths]

    num_frames = len(folder_files[0])
    num_folders = len(folder_files)

    for frame_counter in range(num_frames):
        frame_counter_folder = os.path.join(all_frames_path, str(frame_counter))
        os.mkdir(frame_counter_folder)

        for i in range(num_folders):
            src_path = os.path.join(folder_paths[i], folder_files[i][frame_counter])
            dest_path = os.path.join(frame_counter_folder, f"{i}.png")
            shutil.copyfile(src_path, dest_path)

    guidance_image = folder_files[0][0]
    guidance_path = os.path.join(folder_paths[0], guidance_image)
    guidance_folder = os.path.join(save_path, GUIDANCE_DIR)
    os.mkdir(guidance_folder)
    shutil.copyfile(guidance_path, os.path.join(save_path, GUIDANCE_DIR, GUIDANCE_IMAGE))

    return save_path


def create_video(input_folder):

    name = os.path.basename(input_folder)
    outer_folder = os.path.dirname(input_folder)
    video_name = f"{name}.mp4"
    video_path = os.path.join(outer_folder, video_name)

    images = sorted(os.listdir(input_folder), key=lambda x: int(os.path.splitext(x)[0]))
    frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # You can also try 'avc1'
    fps = 30  # Adjust frame rate as needed
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Write images to video
    for image in images:
        img_path = os.path.join(input_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    # Release resources
    video.release()
    cv2.destroyAllWindows()


def separate_cameras(results_folder, cameras_folder):
    frame_types = [DIFFUSION_FRAMES, RENDER_FRAMES]

    for frame_number in os.listdir(results_folder):
        if not os.path.isdir(os.path.join(results_folder, frame_number)):
            continue


        for frame_type in frame_types:
            frame_folder = os.path.join(results_folder, frame_number, frame_type)
            for camera in os.listdir(frame_folder):
                file_name = os.path.join(frame_folder, camera)
                name, ext = os.path.splitext(camera)
                name = name.split("_")[1]

                name_folder = os.path.join(cameras_folder, frame_type, f"camera_{name}")
                #print(name_folder, "--", file_name)
                if not os.path.exists(name_folder):
                    os.makedirs(name_folder)

                shutil.copyfile(file_name, f"{name_folder}/{frame_number}.png")

    print("Creating Videos")
    for frame_type in frame_types:
        camera_files = [f for f in os.listdir(os.path.join(cameras_folder, frame_type))]
        for file in camera_files:
            create_video(os.path.join(cameras_folder, frame_type, file))


def visualize_comparison(img1, img2, diff_mask, static_mask, path):
    """Visualizes the initial image comparison and mask creation."""
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].imshow(img1.permute(1, 2, 0).cpu().numpy())
    axs[0].set_title("Current Image")
    axs[0].axis("off")

    axs[1].imshow(img2.permute(1, 2, 0).cpu().numpy())
    axs[1].set_title("Previous Image")
    axs[1].axis("off")

    axs[2].imshow(diff_mask.squeeze().cpu().numpy(), cmap='hot')
    axs[2].set_title("Pixel Difference (Sum)")
    axs[2].axis("off")

    axs[3].imshow(static_mask.squeeze().cpu().numpy(), cmap='gray')
    axs[3].set_title("Static Mask (Similar Pixels)")
    axs[3].axis("off")

    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)  # Close the figure to free up memory


def visualize_reprojection(full_res_mask, latent_mask, path):
    """Visualizes the final re-projected mask and its downsampled version."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # --- CHANGE IS HERE ---
    # 'gray_r' plots 0.0 as white and 1.0 as black.
    # This makes sparse points clearly visible on a white background.
    cmap_to_use = 'gray_r'

    axs[0].imshow(full_res_mask.squeeze().cpu().numpy(), cmap=cmap_to_use)
    axs[0].set_title("Full-Res Re-projected Mask (Sparse)")
    axs[0].axis("off")

    axs[1].imshow(latent_mask.squeeze().cpu().numpy(), cmap=cmap_to_use)
    axs[1].set_title(f"Downsampled Mask ({latent_mask.shape[-2]}x{latent_mask.shape[-1]})")
    axs[1].axis("off")

    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig) # Close the figure to free up memory


def visualize_point_cloud(pcd, masked_pcd, path):
    """Visualizes the original and masked 3D point clouds."""
    fig = plt.figure(figsize=(12, 6))

    # --- Original Point Cloud ---
    ax1 = fig.add_subplot(121, projection='3d')
    points = pcd.cpu().numpy()
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.1, c='blue', label='Original')
    ax1.set_title("Original Point Cloud")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend()

    # --- Masked Point Cloud ---
    ax2 = fig.add_subplot(122, projection='3d')
    masked_points = masked_pcd.cpu().numpy()
    ax2.scatter(masked_points[:, 0], masked_points[:, 1], masked_points[:, 2], s=0.1, c='red', label='Static')
    ax2.set_title("Masked (Static) Point Cloud")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def visualize_stacked_latent_masks(stacked_tensor, save_path=None):
    """Visualize tensor of shape (1, 1, 16, 72, 128)"""

    # Extract the actual data: (1, 1, 16, 72, 128) -> (16, 72, 128)
    masks_data = stacked_tensor.squeeze(0).squeeze(0)  # Remove batch and channel dims

    # Method 1: Show as grid (2 rows x 8 cols)
    fig, axes = plt.subplots(2, 8, figsize=(20, 6))
    axes = axes.flatten()

    for i in range(16):
        mask_np = masks_data[i].cpu().numpy()  # (72, 128)
        axes[i].imshow(mask_np, cmap='gray')
        axes[i].set_title(f'Frame {i + 1}', fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    plt.suptitle('16 Latent Masks (72x128 each)', y=1.02, fontsize=14)

    if save_path:
        plt.savefig(os.path.join(save_path, "latents.png"), dpi=150, bbox_inches='tight')



# --- REFACTORED MAIN FUNCTION ---



def load_easi3r_masks(input_path, h, w, t, save_path):
    easier_mask = Image.open(input_path).convert(
        "L")
    to_tensor = transforms.ToTensor()  # Converts to float tensor in range [0, 1]
    mask_tensor = to_tensor(easier_mask)
    mask_tensor = 1.0 - mask_tensor # invert to fit with ddim sampling blending
    mask_tensor = mask_tensor.unsqueeze(0)
    mask_latent = F.interpolate(
        mask_tensor,
        size=(h, w),
        mode='area'  # 'area' interpolation is robust for downsampling
    )
    masks_video = mask_latent.unsqueeze(2).repeat(1, 1, t, 1, 1)
    visualize_reprojection(mask_tensor, mask_latent, save_path)

    return masks_video


def create_masks(current_imgs, prev_imgs, h, w, t, output_dir, pcd, trajectory_cameras, threshold=0.1):
    os.makedirs(output_dir, exist_ok=True) # visualizations

    if not isinstance(current_imgs, list):
        current_imgs = [current_imgs]
        prev_imgs = [prev_imgs]

    assert len(current_imgs) == len(prev_imgs)

    all_masks = []
    mask_points_3d = []
    combined_mask = None
    for i in range(len(current_imgs)):
        image1 = current_imgs[i]
        image2 = prev_imgs[i]

        assert image1.max() <= 1.0 and image1.min() >= 0
        assert image2.max() <= 1.0 and image2.min() >= 0

        img1 = image1.permute(2, 0, 1).unsqueeze(0)  # bchw
        img2 = image2.permute(2, 0, 1).unsqueeze(0)

        abs_diff = torch.abs(img1 - img2)
        diff_mask_pixel_space = torch.sum(abs_diff, dim=1, keepdim=True)  # Shape: [1, 1, H, W]

        # Mask is 1.0 where pixels are SIMILAR, 0.0 where they are DIFFERENT
        mask_pixel_space = (diff_mask_pixel_space < threshold).float()

        vis_path_initial = os.path.join(output_dir, f"01_initial_comparison_{i}.png")
        visualize_comparison(img1.squeeze(0), img2.squeeze(0), diff_mask_pixel_space, mask_pixel_space, vis_path_initial)

        h2, w2 = mask_pixel_space.shape[2], mask_pixel_space.shape[3]
        mask_2d_half = F.interpolate(mask_pixel_space.float(), size=(h2 // 2, w2 // 2), mode='nearest')
        mask_2d_half = mask_2d_half.squeeze(0).squeeze(0).bool()
        mask_2d_half = ~mask_2d_half
        points = pcd[i].cpu()
        masked_points = points[mask_2d_half]
        mask_points_3d.append(masked_points)
        print(f"  Image Pair {i}: Found {len(masked_points)} static 3D points.")

    # Combine all static points from all image pairs
    combined_points = torch.cat(mask_points_3d, dim=0)
    print(f"  Total static points collected: {len(combined_points)}")

    vis_path_pcd = os.path.join(output_dir, "02_3d_point_clouds.png")
    # Assuming the first pcd is representative of the overall structure
    visualize_point_cloud(pcd[0], combined_points, vis_path_pcd)
    print(f"  Saved point cloud visualization to {vis_path_pcd}")

    H, W = trajectory_cameras.image_size[0].tolist()
    device = trajectory_cameras.device
    num_cameras = len(trajectory_cameras)
    combined_points = combined_points.to(device)

    # Project the combined static points onto all camera screens at once
    projected_points_screen = trajectory_cameras.transform_points_screen(combined_points)

    new_masks = []
    print(f"\n--- Step 2: Re-projecting {len(combined_points)} static points onto {num_cameras} camera views ---")
    for i in range(num_cameras):
        xy_coords = projected_points_screen[i, :, :2]
        z_depth = projected_points_screen[i, :, 2]

        # Filter for points visible in the camera frame
        valid_mask = (z_depth > 0) & (xy_coords[:, 0] >= 0) & (xy_coords[:, 0] < W) & (xy_coords[:, 1] >= 0) & (
                    xy_coords[:, 1] < H)
        valid_coords = xy_coords[valid_mask].long()

        # Create the full-resolution mask by "splatting" points
        full_res_mask = torch.zeros((int(H), int(W)), device=device, dtype=torch.float32)
        if valid_coords.shape[0] > 0:
            full_res_mask[valid_coords[:, 1], valid_coords[:, 0]] = 1.0

        # Prepare for downsampling
        finished_mask_bchw = full_res_mask.unsqueeze(0).unsqueeze(0)

        # Downsample using 'area' interpolation
        mask_latent = F.interpolate(
            finished_mask_bchw,
            size=(h, w),
            mode='area'
        )

        # [FIXED] The threshold was the main bug. 'area' mode averages pixel values.
        # If a 64x64 patch (downsampling from 512->8) contains just one white pixel,
        # the average is 1/4096. A threshold of > 0.99 is impossible.
        # A threshold of > 0.0 means "keep this latent pixel if ANY part of it was masked".
        mask_latent = (mask_latent > 0.0).float()
        mask_latent = 1 - mask_latent
        new_masks.append(mask_latent.cpu())

        vis_path_reproj = os.path.join(output_dir, f"03_reprojected_mask_cam_{i:02d}.png")
        visualize_reprojection(full_res_mask, mask_latent, vis_path_reproj)
        print(f"  Camera {i}: Re-projected {valid_coords.shape[0]} points. Visualization saved to {vis_path_reproj}")

    return new_masks

# https://learnopencv.com/simple-background-estimation-in-videos-using-opencv-c-python/
def estimate_background(video):
    frames = []
    cap = cv2.VideoCapture(video)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    background = np.median(frames, axis=0).astype(np.uint8)

    cv2.imwrite('/media/emmahaidacher/Volume/TESTS/bg.jpg', background)


def main():
    results_folder = "/home/emmahaidacher/Masterthesis/MasterThesis/good_results/espresso_fixedpose_2cams_60frames_mast3r_sameguidance_det-sampling_temp/results"
    cameras_folder = "/home/emmahaidacher/Masterthesis/MasterThesis/good_results/espresso_fixedpose_2cams_60frames_mast3r_sameguidance_det-sampling_temp/cameras"
    # input_vid = "/home/emmahaidacher/Masterthesis/MasterThesis/noisy_espresso_video/test.mp4"
    # output_folder = "/home/emmahaidacher/Masterthesis/MasterThesis/noisy_espresso_video/frames"
    # extract_frames(input_vid, output_folder)
    img1 = "/media/emmahaidacher/Volume/GOOD_RESULTS/espresso_1cam_16frames_pickle_deflick_reuse_latent_alpha8/camera_frames/0/00001.png"
    img2 = "/media/emmahaidacher/Volume/GOOD_RESULTS/espresso_1cam_16frames_pickle_deflick_reuse_latent_alpha8/camera_frames/0/00002.png"
    vid = "/media/emmahaidacher/Volume/DATASETS/INTERNET/espresso_short/1_video_short/0.mp4"
    estimate_background(vid)
    #separate_cameras(results_folder, cameras_folder)

if __name__ == "__main__":
    main()