import os
import shutil
import subprocess

import cv2
import numpy as np
import torch
from PIL import Image

from configs.v2v_config import OUTPUT_LOG_FILE, CAMERA_FRAMES_DIR, INPUTS_DIR, RESULTS_DIR, SEPERATED_CAMERAS_DIR, \
    ORIGINAL_VIDEOS_DIR, DIFFUSION_FRAMES, RENDER_FRAMES, MASKS_DIR, GUIDANCE_IMAGE, GUIDANCE_DIR

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

def check_images_for_diff(img1, img2):
    img1_gray = F.rgb_to_grayscale(img1)
    img2_gray = F.rgb_to_grayscale(img2)

    # --- 3. Calculate Difference using NumPy ---
    # Squeeze the channel dimension and convert to numpy for calculation
    arr1 = img1_resized.squeeze().numpy().astype(np.int16)
    arr2 = img2_resized.squeeze().numpy().astype(np.int16)

    # Calculate the absolute difference
    diff = np.abs(arr1 - arr2)

    # Create a binary mask based on the threshold
    mask = np.where(diff > threshold, 255, 0).astype(np.uint8)

    print(f"Successfully created difference mask from tensors. Found {np.sum(mask > 0)} different pixels.")
    return mask
    return thresh


def main():
    results_folder = "/home/emmahaidacher/Masterthesis/MasterThesis/good_results/espresso_fixedpose_2cams_60frames_mast3r_sameguidance_det-sampling_temp/results"
    cameras_folder = "/home/emmahaidacher/Masterthesis/MasterThesis/good_results/espresso_fixedpose_2cams_60frames_mast3r_sameguidance_det-sampling_temp/cameras"
    # input_vid = "/home/emmahaidacher/Masterthesis/MasterThesis/noisy_espresso_video/test.mp4"
    # output_folder = "/home/emmahaidacher/Masterthesis/MasterThesis/noisy_espresso_video/frames"
    # extract_frames(input_vid, output_folder)
    img1 = "/media/emmahaidacher/Volume/GOOD_RESULTS/espresso_1cam_16frames_pickle_deflick_reuse_latent_alpha8/camera_frames/0/00001.png"
    img2 = "/media/emmahaidacher/Volume/GOOD_RESULTS/espresso_1cam_16frames_pickle_deflick_reuse_latent_alpha8/camera_frames/0/00002.png"
    check_images_for_diff(img1, img2)
    #separate_cameras(results_folder, cameras_folder)

if __name__ == "__main__":
    main()