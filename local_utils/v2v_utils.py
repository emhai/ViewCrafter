import csv
import re
import shutil
import subprocess
import cv2
import numpy as np
from pathlib import Path
from configs.v2v_config import *


def numeric_key(path):
    # Extract the first integer found in the filename (e.g. 1, 2, 10, etc.)
    match = re.search(r'\d+', path.stem)
    return int(match.group()) if match else -1

def dir_empty(dir_path):
     path = Path(dir_path)
     has_next = next(path.iterdir(), None)
     if has_next is None:
             return True
     return False

def extract_frames(video_path, frames_path):
    # print(f"Extracting frames from {video_path}")

    #  '-hide_banner', '-log_level', 'error'
    ffmpeg_command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(video_path), f"{str(frames_path)}/%05d.png"]
    subprocess.run(ffmpeg_command)

def ffmpeg_side_by_side_vid(vid1, vid2, output_vid):
    ffmpeg_command = ["ffmpeg", "-y", "-i", str(vid1), "-i", str(vid2), "-filter_complex", "hstack=shortest=1", "-c:v", "libx264", str(output_vid)]
    subprocess.run(ffmpeg_command)

def ffmpeg_overlay_5050(vid1, vid2, output_vid):
    ffmpeg_command = [
        "ffmpeg", "-y",
        "-i", str(vid1),
        "-i", str(vid2),
        "-filter_complex",
        "[0:v][1:v]blend=all_mode=average:shortest=1",
        "-c:v", "libx264",
        str(output_vid),
    ]
    subprocess.run(ffmpeg_command)

def ffmpeg_4x4_video(input_folder, output_folder):

    video_files = sorted([
        p for p in input_folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".mp4"
    ])

    if len(video_files) != 16:
        print("Couldn't create video, not 16 input videos")
        return

    # -------- 4x4 GRID (all 16 videos) --------
    output_vid_4x4 = output_folder / "gen_4x4.mp4"
    cmd_4x4 = ["ffmpeg", "-y"]
    for vf in video_files:
        cmd_4x4 += ["-i", str(vf)]
    # xstack with a 4x4 grid
    filter_complex_4x4 = "xstack=grid=4x4:shortest=1:fill=black"
    cmd_4x4 += [
        "-filter_complex", filter_complex_4x4,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "medium",
        str(output_vid_4x4),
    ]
    subprocess.run(cmd_4x4)

    # -------- 2x2 GRID (indices 0, 5, 10, 15) --------
    indices_2x2 = (0, 5, 10, 15)
    output_vid_2x2 = output_folder / "gen_2x2.mp4"
    selected_files = [video_files[i] for i in indices_2x2]
    cmd_2x2 = ["ffmpeg", "-y"]
    for vf in selected_files:
        cmd_2x2 += ["-i", str(vf)]
    # 2x2 grid
    filter_complex_2x2 = "xstack=grid=2x2:shortest=1:fill=black"
    cmd_2x2 += [
        "-filter_complex", filter_complex_2x2,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "medium",
        str(output_vid_2x2),
    ]
    subprocess.run(cmd_2x2)

def create_folder_structure(folders):
    for folder in folders:
        if not folder.exists():
            folder.mkdir()
            # print('Created folder:', folder)

def setup_structure(save_path, source_path, gt_path):

    inputs_path = save_path / INPUTS_DIR
    results_path = save_path / RESULTS_DIR
    gen_videos_path = save_path / GENERATED_VIDEOS_DIR
    gen_frames_path = save_path / GENERATED_FRAMES_DIR
    og_videos_path = save_path / ORIGINAL_VIDEOS_DIR
    og_frames_path = save_path / ORIGINAL_FRAMES_DIR
    gt_videos_path = save_path / GROUND_TRUTH_VIDEOS_DIR
    gt_frames_path = save_path / GROUND_TRUTH_FRAMES_DIR
    rnd_videos_path = save_path / RENDERED_VIDEOS_DIR
    rnd_frames_path = save_path / RENDERED_FRAMES_DIR
    vis_results_path = save_path / VIS_RESULTS_DIR
    misc_path = save_path / MISC_DIR

    all_folders = [og_videos_path,
                   og_frames_path,
                   gen_videos_path,
                   gen_frames_path,
                   gt_videos_path,
                   gt_frames_path,
                   rnd_videos_path,
                   rnd_frames_path,
                   inputs_path,
                   results_path,
                   vis_results_path,
                   misc_path]

    create_folder_structure(all_folders)

    # copy video folder
    for og_video in source_path.iterdir():
        target_video_path = og_videos_path / og_video.name
        shutil.copy(og_video, target_video_path)
        frames_path = og_frames_path / og_video.stem
        frames_path.mkdir(exist_ok=True)
        extract_frames(og_video, frames_path)

    # copy ground truths folder
    if gt_path is not None:
        for gt_video in gt_path.iterdir():
            target_video_path = gt_videos_path / gt_video.name
            shutil.copy(gt_video, target_video_path)
            frames_path = gt_frames_path / gt_video.stem
            frames_path.mkdir(exist_ok=True)
            extract_frames(gt_video, frames_path)

    frame_folders = sorted(og_frames_path.iterdir())
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

    print("Folder Setup Completed!")

def create_video(input_folder, output_folder):

    camera_name = input_folder.stem
    video_name = f"{camera_name}.mp4"
    video_path = str(output_folder / video_name)

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

    video.release()
    cv2.destroyAllWindows()

def separate_cameras(base_dir, frame_type):

    results_folder = base_dir / RESULTS_DIR
    if frame_type == DIFFUSION_FRAMES:
        output_frames_folder = base_dir / GENERATED_FRAMES_DIR
        output_videos_folder = base_dir / GENERATED_VIDEOS_DIR
    elif frame_type == RENDER_FRAMES:
        output_frames_folder = base_dir / RENDERED_FRAMES_DIR
        output_videos_folder = base_dir / RENDERED_VIDEOS_DIR
    else:
        print("FAILED")
        return

    for frame_number in results_folder.iterdir(): # 1 folder in results equal to 1 frame with n synthesized cameras
        if not frame_number.is_dir():
            continue

        frame_folder = frame_number / frame_type # diffusion_frames or render_frames
        for camera in frame_folder.iterdir(): # n synthesized cameras
            cam_name = int(camera.stem.split("_")[1]) # starts at frame_0001 -> 1 -> cam01
            frame_name = int(frame_number.stem) + 1 # starts at results 0 -> 1 -> frame_00001

            camera_folder_name = output_frames_folder / f"cam{cam_name:02}" # 4DGS standard
            #print(name_folder, "--", file_name)
            if not camera_folder_name.exists():
                camera_folder_name.mkdir(parents=True)

            dst = f"{str(camera_folder_name)}/frame_{frame_name:05}.jpg"

            shutil.copyfile(str(camera), dst) # 4DGS standard

    print("Creating Videos")
    for all_frames_folder in output_frames_folder.iterdir():
        create_video(all_frames_folder, output_videos_folder)

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
        cameras = sub / GENERATED_VIDEOS_DIR
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

def print_diffusion_model(model, max_depth=3, prefix='', depth=0):
    if depth > max_depth:
        return
    print(f"{'  ' * depth}{prefix}{model.__class__.__name__}")
    for name, child in model.named_children():
        print_diffusion_model(child, max_depth, prefix=name + ': ', depth=depth + 1)

def main():
    results_folder = Path("/media/emmahaidacher/Volume/GOOD_RESULTS/20251020_1740_yoga_debug/results")
    cameras_folder = Path("/media/emmahaidacher/Volume/GOOD_RESULTS/20251020_1740_yoga_debug/cameras")
    # input_vid = "/home/emmahaidacher/Masterthesis/MasterThesis/noisy_espresso_video/test.mp4"
    # output_folder = "/home/emmahaidacher/Masterthesis/MasterThesis/noisy_espresso_video/frames"
    # extract_frames(input_vid, output_folder)
    # img1 = "/media/emmahaidacher/Volume/GOOD_RESULTS/espresso_1cam_16frames_pickle_deflick_reuse_latent_alpha8/camera_frames/0/00001.png"
    # img2 = "/media/emmahaidacher/Volume/GOOD_RESULTS/espresso_1cam_16frames_pickle_deflick_reuse_latent_alpha8/camera_frames/0/00002.png"
    # vid = "/media/emmahaidacher/Volume/DATASETS/INTERNET/espresso_short/1_video_short/0.mp4"
    # estimate_background(vid)
    # separate_cameras(results_folder, cameras_folder, DIFFUSION_FRAMES)
    # separate_cameras(results_folder, cameras_folder, RENDER_FRAMES)

    output_path = Path("/media/emmahaidacher/Volume/DATASETS/MODIFIED_DATASETS")
    input_path = Path("/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/SelfCap/bike-release/videos")
    # input_path = Path("/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/SelfCap/yoga3/")
    # output_path = Path("/media/emmahaidacher/Volume/TESTS/test_sep")
    # output_path.mkdir(exist_ok=True)
    # setup_structure(output_path, input_path, 16)
    #clean_empty_camera_folders()
    # clean_mask("")
    # vid_path_in = Path("/media/emmahaidacher/Volume/GOOD_RESULTS/results_26_11/10_cfg__ddim__latent_blending_salmon_3x15_near/generated_videos")
    # vid_path_out = Path("/media/emmahaidacher/Volume/GOOD_RESULTS/results_26_11/10_cfg__ddim__latent_blending_salmon_3x15_near/vis_results")
    # ffmpeg_4x4_video(vid_path_in, vid_path_out)
    separate_cameras(Path("/media/emmahaidacher/Volume/GOOD_RESULTS/20251209_1548_spinach_2_metrics"), DIFFUSION_FRAMES)




if __name__ == "__main__":
    main()