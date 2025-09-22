import os
import shutil
import subprocess

import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from pathlib import Path
from skimage import data, filters
from torch.ao.nn.quantized.functional import threshold
from torch.utils.tensorboard.summary import video

from configs.v2v_config import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip, ffmpeg_resize

import matplotlib.pyplot as plt
from utils.pvd_utils import save_pointcloud_with_normals, get_pc, center_crop_image
from utils.visualization_utils import visualize_pixel_masks


def extract_frames(video_path, frames_path, n_frames):
    print(f"Extracting frames from {video_path}")

    #  '-hide_banner', '-log_level', 'error'
    ffmpeg_command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(video_path), "-vf", f"select='between(n,0,{n_frames - 1})'",
    "-vsync", "0", f"{str(frames_path)}/%05d.png"]
    subprocess.run(ffmpeg_command)


def create_folder_structure(folders):
    for folder in folders:
        if not folder.exists():
            folder.mkdir()
            print('Created folder:', folder)

def setup_structure(save_path, source_path, num_frames):

    frames_path = save_path / CAMERA_FRAMES_DIR
    inputs_path = save_path / INPUTS_DIR
    results_path = save_path / RESULTS_DIR
    cameras_path = save_path / SEPERATED_CAMERAS_DIR
    video_path = save_path / ORIGINAL_VIDEOS_DIR

    all_folders = [frames_path, inputs_path, results_path, cameras_path, video_path]
    create_folder_structure(all_folders)

    # copy video folder
    for source_video in source_path.iterdir():
        cam = cv2.VideoCapture(str(source_video))
        fps = cam.get(cv2.CAP_PROP_FPS)
        w, h = cam.get(cv2.CAP_PROP_FRAME_WIDTH), cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

        temp_video_path = video_path / "temp.mp4"
        # tw, th = w // 2, h // 2   # resizing necessary for too high quality videos, otherwise CUDA OOM
        # print(f"Resizing from {w}x{h} to {tw}x{th}")
        target_video_path = video_path / source_video.name
        # target_num_frames =  (num_frames - 1) / fps
        # ffmpeg_extract_subclip(source_video, 0, target_num_frames, targetname=target_video_path)
        # ffmpeg_resize(temp_video_path, target_video_path, (tw, th))
        # temp_video_path.unlink()

        shutil.copy(source_video, target_video_path)

    print(f"Copying videos from {source_path} to {video_path}")

    # extract frames
    for video in video_path.iterdir():
        new_path = frames_path / video.stem
        new_path.mkdir()
        extract_frames(video, new_path, num_frames)

    frame_folders = sorted(frames_path.iterdir())
    frame_files = [sorted(files.iterdir()) for files in frame_folders]

    num_frames = len(frame_files[0])
    num_folders = len(frame_files)

    for frame_idx in range(num_frames):
        new_input_folder = inputs_path / str(frame_idx)
        new_input_folder.mkdir()

        for folder_idx in range(num_folders):
            src = frame_folders[folder_idx] / frame_files[folder_idx][frame_idx]
            dst = new_input_folder / f"{folder_idx}.png"
            shutil.copyfile(src, dst)

def create_video(input_folder):

    camera_name = input_folder.stem
    camera_dir = input_folder.parent
    video_name = f"{camera_name}.mp4"
    video_path = str(camera_dir / video_name)

    images = sorted(input_folder.iterdir(), key=lambda x: int(x.stem.split("_")[-1]))

    frame = cv2.imread(str(input_folder / images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 15
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Write images to video
    for image in images:
        img_path = str(input_folder / image)
        frame = cv2.imread(img_path)
        video.write(frame)

    # Release resources
    video.release()
    cv2.destroyAllWindows()

def separate_cameras(results_folder, all_cameras_folder, frame_type):

    for frame_number in results_folder.iterdir(): # 1 folder in results equal to 1 frame with n synthesized cameras
        if not frame_number.is_dir():
            continue

        frame_folder = frame_number / frame_type # diffusion_frames or render_frames
        for camera in frame_folder.iterdir(): # n synthesized cameras
            cam_name = int(camera.stem.split("_")[1]) # starts at frame_0001 -> 1 -> cam01
            frame_name = int(frame_number.stem) + 1 # starts at results 0 -> 1 -> frame_00001

            camera_folder_name = all_cameras_folder / frame_type / f"cam{cam_name:02}" # 4DGS standard
            #print(name_folder, "--", file_name)
            if not camera_folder_name.exists():
                camera_folder_name.mkdir(parents=True)

            dst = f"{str(camera_folder_name)}/frame_{frame_name:05}.jpg"

            shutil.copyfile(str(camera), dst) # 4DGS standard

    print("Creating Videos")
    for all_frames_folder in (all_cameras_folder / frame_type).iterdir():
        create_video(all_frames_folder)


def create_frame_diff_masks(current_imgs, prev_imgs, threshold=0.1, output_dir=None):
    # creates masks of shape (1, 1, H /2, W/2) # same dim as point cloud created by dust3r

    if not isinstance(current_imgs, list):
        current_imgs = [current_imgs]
        prev_imgs = [prev_imgs]

    assert len(current_imgs) == len(prev_imgs)

    all_masks = []

    for i in range(len(current_imgs)):
        image1 = current_imgs[i]
        image2 = prev_imgs[i]

        assert image1.max() <= 1.0 and image1.min() >= 0
        assert image2.max() <= 1.0 and image2.min() >= 0

        img1 = image1.permute(2, 0, 1).unsqueeze(0)  # bchw
        img2 = image2.permute(2, 0, 1).unsqueeze(0)

        abs_diff = torch.abs(img1 - img2)
        diff_mask_pixel_space = torch.sum(abs_diff, dim=1, keepdim=True)  # Shape: [1, 1, H, W]

        # Mask is 1.0 where pixels are similar, 0.0 where they are different
        # mask_pixel_space = (diff_mask_pixel_space < threshold).float()

        mask_pixel_space = (diff_mask_pixel_space > threshold).float()
        h2, w2 = mask_pixel_space.shape[2], mask_pixel_space.shape[3]
        mask_pixel_space_half = F.interpolate(mask_pixel_space.float(), size=(h2 // 2, w2 // 2),
                                              mode='nearest')  # get to same dim as pc
        print(mask_pixel_space_half.shape)

        if output_dir is not None:
            visualize_pixel_masks(mask_pixel_space_half, current_imgs[i], output_dir /f"pixel_diffs_{i}.png", "difference between first frame and current")

        all_masks.append(mask_pixel_space_half.bool().squeeze())

    return all_masks


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

    cv2.imwrite('/media/emmahaidacher/Volume/TESTS/bg.png', background)


def clean_empty_camera_folders():
    base_folder = Path("/media/emmahaidacher/Volume/GOOD_RESULTS/")
    for sub in base_folder.iterdir():
        if not sub.is_dir():
            continue
        cameras = sub / "cameras"
        if cameras.exists() and cameras.is_dir():
            if not any(cameras.iterdir()):  # cameras folder is empty
                print(f"Deleting {sub} (empty cameras folder)")
                shutil.rmtree(sub)
            else:
                print(f"Keeping {sub} (cameras has files)")
        else:
            items = list(sub.iterdir())
            if len(items) == 1 and items[0].is_file() and items[0].name == "args.json":
                print(f"Deleting {sub} (no cameras folder, only 1 item inside)")
                shutil.rmtree(sub)
            else:
                print(f"Keeping {sub} (no cameras folder, more than 1 item)")
            print(f"Skipping {sub} (no cameras folder)")

def main():
    # results_folder = Path("/media/emmahaidacher/Volume/TESTS/debug_test/results")
    # cameras_folder = Path("/media/emmahaidacher/Volume/TESTS/debug_test/cameras")
    # input_vid = "/home/emmahaidacher/Masterthesis/MasterThesis/noisy_espresso_video/test.mp4"
    # output_folder = "/home/emmahaidacher/Masterthesis/MasterThesis/noisy_espresso_video/frames"
    # extract_frames(input_vid, output_folder)
    # img1 = "/media/emmahaidacher/Volume/GOOD_RESULTS/espresso_1cam_16frames_pickle_deflick_reuse_latent_alpha8/camera_frames/0/00001.png"
    # img2 = "/media/emmahaidacher/Volume/GOOD_RESULTS/espresso_1cam_16frames_pickle_deflick_reuse_latent_alpha8/camera_frames/0/00002.png"
    # vid = "/media/emmahaidacher/Volume/DATASETS/INTERNET/espresso_short/1_video_short/0.mp4"
    # estimate_background(vid)
    # separate_cameras(results_folder, cameras_folder, DIFFUSION_FRAMES)
    clean_empty_camera_folders()

if __name__ == "__main__":
    main()