import shutil
import subprocess
from pathlib import Path
from configs.v2v_config import *
from configs.dataset_config_multi import *

def make_side_by_side(original_video, changed_video, output_video, compare_height=576):
    original_video = Path(original_video)
    changed_video = Path(changed_video)
    output_video = Path(output_video)

    filter_complex = (
        f"[0:v]setsar=1[orig];"
        f"[1:v]setsar=1,pad=iw:{2028}:0:({2028}-ih)/2[proc];"
        f"[orig][proc]hstack=shortest=1[v]"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(original_video),
        "-i", str(changed_video),
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-crf", "16",
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        str(output_video),
    ]

    subprocess.run(cmd, check=True)

def downsample_crop_cut_video(input_video, output_video, fps, start, duration, crop):
    if crop == "bottom":
        # center horizontally, stick to bottom vertically
        crop = (
            f"crop={TARGET_W}:{TARGET_H}:"
            f"(in_w-{TARGET_W})/2:in_h-{TARGET_H}"
        )
    else:
        # center horizontally + vertically
        crop = (
            f"crop={TARGET_W}:{TARGET_H}:"
            f"(in_w-{TARGET_W})/2:(in_h-{TARGET_H})/2"
        )

    vf = (
        f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase:"
        f"flags=lanczos,"
        f"{crop},"
        f"fps={fps}"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start),  # seek to start
        "-i", str(input_video),
        "-t", str(duration),  # clip duration
        "-vf", vf,
        "-c:v", "libx264",
        "-c:a", "copy",
        str(output_video)
    ]
    subprocess.run(cmd, check=True)

def create_modified_dataset(input_dir, output_dir, exp_name, starting_time, duration, fps, crop):

    new_folder = output_dir / f"{exp_name}_{duration}x{fps}"
    new_folder.mkdir(exist_ok=True)

    for file in Path(input_dir).iterdir():
        if file.suffix != ".mp4":
            shutil.copy(file, new_folder)
            continue
        new_video = new_folder / file.name
        downsample_crop_cut_video(file, new_video, fps, starting_time, duration, crop)
        #cut_video(new_video, new_video, starting_time, sec, fps)

    # make_side_by_side(file, new_video, new_folder / f"comparison_{file.stem}.mp4") # one comparison per folder?

    with open(new_folder / "README.txt", 'w') as output:
        output.write(f"amount of frames: {duration * fps}\n fps: {fps}\n starting_time: {starting_time}")

def generate_full_datasets(modified_path):
    for i, input_path in enumerate(paths):
        for setup in setup_duration:
            create_modified_dataset(input_path, modified_path, names[i], ss[i], setup[0], setup[1], crops[i])

def generate_filtered_datasets(full_modified_path, new_modified_path):
    for dataset in full_modified_path.iterdir():
        dataset_name = dataset.stem.split("_")[0]
        dataset_id = names.index(dataset_name)
        dataset_videos = video_names[dataset_id]
        print(dataset.name)
        new_path = new_modified_path / dataset.name
        new_path.mkdir(exist_ok=True)

        for i, video in enumerate(dataset_videos):
            filename = f"{i:04d}.mp4"
            shutil.copy(str(dataset / video), str(new_path / filename))



def generate_tuple_datasets(modified_path, new_path):

    for dataset in modified_path.iterdir():

        dataset_videos = sorted(dataset.iterdir()) # todo what if more than 10 videos, is it still sorted corectly?
        amount_vids = len(dataset_videos)
        amount_runs = amount_vids // 2

        for i in range(amount_runs):
            middle_index = amount_runs

            start = middle_index - i
            end = middle_index + i

            ground_truths = dataset_videos[start:end + 1]

            inputs = []
            left = start - 1
            right = end + 1

            inputs.append(dataset_videos[left])
            inputs.append(dataset_videos[right])

            distance = distances[i]

            new_folder = new_path / f"{dataset.name}_{distance}"
            new_folder.mkdir(exist_ok=True)

            gt_path = new_folder / "gt"
            gt_path.mkdir(exist_ok=True)

            input_path = new_folder / "input"
            input_path.mkdir(exist_ok=True)

            for video in ground_truths:
                shutil.copy(str(video), str(gt_path / video.name))

            for video in inputs:
                shutil.copy(str(video), str(input_path / video.name))

def generate_frames(modified_path, frames_path):

    for folder in modified_path.iterdir():
        name = folder.name.split("_")[0]
        frames_dir = frames_path / name
        if frames_dir.exists():
            continue
        frames_dir.mkdir(exist_ok=True)

        for video in folder.iterdir():
            out_png = frames_dir / (video.stem + ".png")

            cmd = [
                "ffmpeg",
                "-y",
                "-i", str(video),
                "-frames:v", "1",
                str(out_png),
            ]

            subprocess.run(cmd, check=True)

def main():
    full_modified_dataset_path = Path("/media/emmahaidacher/Volume/DATASETS/MODIFIED_DATASETS_4x15/full_modified_datasets")
    full_modified_dataset_path.mkdir(exist_ok=True)

    modified_dataset_path = Path("/media/emmahaidacher/Volume/DATASETS/MODIFIED_DATASETS_4x15/modified_datasets")
    modified_dataset_path.mkdir(exist_ok=True)

    frames_path = Path("/media/emmahaidacher/Volume/DATASETS/MODIFIED_DATASETS_4x15/frames")
    frames_path.mkdir(exist_ok=True)

    finished_dataset_path = Path("/media/emmahaidacher/Volume/DATASETS/MODIFIED_DATASETS_4x15/datasets")
    finished_dataset_path.mkdir(exist_ok=True)

    assert len(paths) == len(names) == len(video_names) == len(ss) == len(crops)

    generate_full_datasets(full_modified_dataset_path)
    generate_filtered_datasets(full_modified_dataset_path, modified_dataset_path)
    generate_frames(modified_dataset_path, frames_path)
    generate_tuple_datasets(modified_dataset_path, finished_dataset_path)

if __name__ == "__main__":
    main()