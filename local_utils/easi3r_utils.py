import os
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from moviepy.video.io.ffmpeg_tools import ffmpeg_resize
from torchvision.transforms import CenterCrop

from configs.v2v_config import *
from PIL import Image
import torchvision.transforms as transforms

from local_utils.v2v_utils import numeric_key
from local_utils.visualization_utils import visualize_pixel_masks

PATH_TO_EASI3R = Path("/home/emmahaidacher/Desktop/Easi3R")


def downsample_video(input_video, factor):

    output_folder = input_video.parent
    cam = cv2.VideoCapture(str(input_video))
    w, h = cam.get(cv2.CAP_PROP_FRAME_WIDTH), cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

    tw, th = w // factor, h // factor   # resizing necessary for too high quality videos, otherwise CUDA OOM
    print(f"Resizing from {w}x{h} to {tw}x{th}")
    target_video_path = output_folder / f"{input_video.stem}_{factor}.mp4"
    # for correct aspect ratio h=-2 https://stackoverflow.com/questions/8218363/maintaining-aspect-ratio-with-ffmpeg
    ffmpeg_resize(input_video, target_video_path, (tw, -2))


def load_easi3r_masks(input_paths, prev_input_paths, current_imgs, H=256, W=512, output_dir=None):

    # creates masks of shape (1, 1, H /2, W/2) # same dim as point cloud created by dust3r
    if not isinstance(current_imgs, list):
        current_imgs = [current_imgs]

    assert len(input_paths) == len(current_imgs)

    all_masks = []
    for i in range(len(input_paths)):

        easier_mask = Image.open(input_paths[i]).convert("L")
        prev_easier_mask = Image.open(prev_input_paths[i]).convert("L")

        crop = CenterCrop((H, W))
        cropped_mask = crop(easier_mask)
        prev_cropped_mask = crop(prev_easier_mask)

        to_tensor = transforms.ToTensor()  # Converts to float tensor in range [0, 1]
        mask_tensor = to_tensor(cropped_mask)
        prev_mask_tensor = to_tensor(prev_cropped_mask)

        combined_mask = (mask_tensor + prev_mask_tensor).clamp(0.0, 1.0)

        # mask_tensor = 1.0 - mask_tensor # invert to fit with ddim sampling blending
        combined_mask = combined_mask.unsqueeze(0)
        print(combined_mask.shape)

        if output_dir is not None:
            visualize_pixel_masks(combined_mask, current_imgs[i], output_dir / f"easi3r_mask_{i}.png", "easi3r mask for this frame")

        all_masks.append(combined_mask.bool().squeeze())

    return all_masks


def run_easi3r_from_video(input_video, output_dir, name, n_frames):

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["OPENBLAS_NUM_THREADS"] = "1"

    cmd = (f"conda run -n easi3r --no-capture-output python -u {PATH_TO_EASI3R}/demo.py "
           f"--weights {PATH_TO_EASI3R}/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth "
           f"--seq_name {name} "
           f"--input {input_video} "
           f"--output_dir {output_dir} "
           f"--sam2_mask_refine "
           f"--num_frames {n_frames} "
           f"--silent")

    # add --silent for non-verbose

    print(">> Running Easi3r")

    proc = subprocess.Popen(cmd, env=env, shell=True, cwd=PATH_TO_EASI3R)
    ret = proc.wait()
    if ret != 0:
        print(f"Easi3R failed with exit code {ret}")


def run_easi3r_from_viewcrafter(base_path, n_frames):

    original_videos_dir = base_path / ORIGINAL_VIDEOS_DIR

    easi3r_results_dir = base_path / EASI3R_RESULTS_DIR
    input_masks_dir = base_path / EASI3R_MASKS_DIR
    pickle_path = base_path / PICKLES_DIR

    easi3r_results_dir.mkdir()
    input_masks_dir.mkdir()
    pickle_path.mkdir()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["OPENBLAS_NUM_THREADS"] = "1"

    for video in original_videos_dir.iterdir():
        name = video.stem

        cmd = (f"conda run -n easi3r --no-capture-output python -u {PATH_TO_EASI3R}/demo.py "
               f"--weights {PATH_TO_EASI3R}/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth "
               f"--seq_name {name} "
               f"--input {video} "
               f"--output_dir {easi3r_results_dir} "
               f"--sam2_mask_refine " # this is important! do not delete
               f"--num_frames {n_frames} "
               f"--silent")

        # add --silent for non-verbose

        print(f">> Running Easi3r on {video}")

        proc = subprocess.Popen(cmd, env=env, shell=True, cwd=PATH_TO_EASI3R)
        ret = proc.wait()
        if ret != 0:
            print(f"Easi3R failed with exit code {ret}")

        pickle_src = easi3r_results_dir / name / "pickle.pkl"
        os.mkdir(pickle_path / name)
        pickle_dst = pickle_path / name / "pickle.pkl"
        if pickle_src.exists():
            shutil.copyfile(pickle_src, pickle_dst)
        else:
            print("pickle doesnt exist")

    easi3r_results = sorted(easi3r_results_dir.iterdir())

    dyn_mask_folders = [folder / "frames_dynamic_masks" for folder in easi3r_results]

    dyn_mask_files = [sorted(folder.iterdir(), key=numeric_key) for folder in dyn_mask_folders]

    num_frames = len(dyn_mask_files[0])
    num_folders = len(dyn_mask_files)

    for frame_idx in range(num_frames):
        new_mask_folder = input_masks_dir / str(frame_idx)
        new_mask_folder.mkdir()

        for folder_idx in range(num_folders):
            src = dyn_mask_folders[folder_idx] / dyn_mask_files[folder_idx][frame_idx]
            dst = new_mask_folder / f"{folder_idx}.png"
            shutil.copyfile(src, dst)

    print("done")

def main():
    path = Path("/media/emmahaidacher/Volume/DATASETS/INTERNET/espresso_short/4dgs_1_cam_downsampled")
    # video = path / "4.mp4"
    # res = [2, 4, 8]
    # for r in res:
    #     downsample_video(video, r)

    for video in path.iterdir():
        run_easi3r_from_video(video, path, video.stem, 60)

if __name__ == "__main__":
    main()