import os
import shutil
import subprocess
from pathlib import Path
from configs.v2v_config import *
from configs.dataset_config_multi import *
from configs.dataset_config_single import *

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

def generate_16x9_videos(input_video, output_video, fps, start, duration, crop_mode):

    w_expr = "if(gt(a\\,16/9)\\,ih*16/9\\,iw)"
    h_expr = "if(gt(a\\,16/9)\\,ih\\,iw*9/16)"

    x_expr = f"(iw-{w_expr})/2"

    if crop_mode == "bottom":
        y_expr = f"ih-{h_expr}"          # stick to bottom
    else:
        y_expr = f"(ih-{h_expr})/2"      # center vertically

    vf = (
        f"crop={w_expr}:{h_expr}:{x_expr}:{y_expr},"
        f"fps={fps}"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-i", str(input_video),
        "-t", str(duration),
        "-vf", vf,
        "-c:v", "libx264",
        "-c:a", "copy",
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

def generate_full_datasets(modified_path, setup_duration, paths, names, ss, crops):
    for i, input_path in enumerate(paths):
        create_modified_dataset(input_path, modified_path, names[i], ss[i], setup_duration[0], setup_duration[1], crops[i])

def generate_filtered_datasets(full_modified_path, new_modified_path, names, video_names):
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

def take_adjacent_gt(dataset, new_path):
    dataset_videos = sorted(dataset.iterdir())  # todo what if more than 10 videos, is it still sorted corectly?
    amount_vids = len(dataset_videos)

    for j in range(amount_vids - 1):
        distance = j
        gt = dataset_videos[j]
        inputs = [dataset_videos[j], dataset_videos[j + 1]]

        new_folder = new_path / f"{dataset.name}_{distance}"
        new_folder.mkdir(exist_ok=True)

        gt_path = new_folder / "gt"
        gt_path.mkdir(exist_ok=True)

        input_path = new_folder / "input"
        input_path.mkdir(exist_ok=True)

        shutil.copy(str(gt), str(gt_path / gt.name))

        for video in inputs:
            shutil.copy(str(video), str(input_path / video.name))

def keep_middle_gt(dataset, new_path):
    dataset_videos = sorted(dataset.iterdir())  # todo what if more than 10 videos, is it still sorted corectly?
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

        if amount_runs == 1:
            distance = ""
        else:
            distance = "_" + mult_distances[i]

        new_folder = new_path / f"{dataset.name}{distance}"
        new_folder.mkdir(exist_ok=True)

        gt_path = new_folder / "gt"
        gt_path.mkdir(exist_ok=True)

        input_path = new_folder / "input"
        input_path.mkdir(exist_ok=True)

        for video in ground_truths:
            shutil.copy(str(video), str(gt_path / video.name))

        for video in inputs:
            shutil.copy(str(video), str(input_path / video.name))



def create_traj_file(name, i,  path):

    index = single_names.index(name)

    traj_amount = str(single_trajs[index][i])
    text = f"0 {traj_amount}\n0 0\n0 0"
    modified = path / f"traj.txt"
    modified.write_text(text)

def single_input_single_gt(modified_path, new_path):


    for dataset in modified_path.iterdir():
        dataset_videos = sorted(dataset.iterdir())  # todo what if more than 10 videos, is it still sorted corectly?
        amount_vids = len(dataset_videos)
        for j in range(1, amount_vids):
            input = dataset_videos[0]
            gt = dataset_videos[j]
            if amount_vids == 2:
                distance_string = ""
            else:
                distance_string = "_" + single_distances[j - 1]
            new_folder = new_path / f"{dataset.name}{distance_string}"
            new_folder.mkdir(exist_ok=True)

            gt_path = new_folder / "gt"
            gt_path.mkdir(exist_ok=True)
            input_path = new_folder / "input"
            input_path.mkdir(exist_ok=True)

            shutil.copy(str(gt), str(gt_path / gt.name))
            shutil.copy(str(input), str(input_path / input.name))

            name = dataset.name.split("_")[0]
            create_traj_file(name, j-1, new_folder)

def generate_tuple_datasets(modified_path, new_path):

    for dataset in modified_path.iterdir():
        dataset_name = dataset.stem.split("_")[0]
        if dataset_name in ["salmon", "coffee", "spinach", "steak"]:
            keep_middle_gt(dataset, new_path)
        else:
            take_adjacent_gt(dataset, new_path)

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

def run(output_path, setup_duration, input_type="multiple"):

    output_path = output_path / f"datasets_{setup_duration[0]}x{setup_duration[1]}_{input_type}"

    full_modified_dataset_path = output_path / "full_datasets"
    full_modified_dataset_path.mkdir(exist_ok=True, parents=True)

    modified_dataset_path = output_path / "modified_datasets"
    modified_dataset_path.mkdir(exist_ok=True, parents=True)

    frames_path = output_path / "frames"
    frames_path.mkdir(exist_ok=True, parents=True)

    finished_dataset_path = output_path / "final_datasets"
    finished_dataset_path.mkdir(exist_ok=True, parents=True)

    if input_type == "multiple":
        paths, names, video_names, ss, crops = mult_paths, mult_names, mult_video_names, mult_ss, mult_crops
    else:
        paths, names, video_names, ss, crops = single_paths, single_names, single_video_names, single_ss, single_crops

    assert len(paths) == len(names) == len(video_names) == len(ss) == len(crops)

    if input_type == "single":
        assert len(single_trajs) == len(paths)

    generate_full_datasets(full_modified_dataset_path, setup_duration, paths, names, ss, crops)
    generate_filtered_datasets(full_modified_dataset_path, modified_dataset_path, names, video_names)

    generate_frames(modified_dataset_path, frames_path)

    if input_type == "multiple":
        generate_tuple_datasets(modified_dataset_path, finished_dataset_path)
    else:
        single_input_single_gt(modified_dataset_path, finished_dataset_path)

def create_robo_dataset():

    setup_duration = (4, 15)
    input_type = "multiple"

    output_path = Path("/media/emmahaidacher/STORAGE8TB/DATASETS/MODIFIED") / f"datasets_{setup_duration[0]}x{setup_duration[1]}_{input_type}_robosapiens"
    output_path.mkdir(exist_ok=True)
    robosapiens_dir = Path("/media/emmahaidacher/STORAGE8TB/DATASETS/OWN/ROBOSAPIENS")
    finished_dataset_path = output_path / "final_datasets"
    finished_dataset_path.mkdir(exist_ok=True)
    full_folder = output_path / "full_datasets"
    full_folder.mkdir(exist_ok=True)

    for distance in robosapiens_dir.iterdir():
        videos = distance / "shortened"
        if not videos.is_dir():
            continue

        new_folder = full_folder / f"robosapiens{distance.name}_{setup_duration[0]}x{setup_duration[1]}"
        new_folder.mkdir(exist_ok=True)

        for file in Path(videos).iterdir():
            if file.suffix != ".mp4":
                continue
            new_video = new_folder / file.name
            downsample_crop_cut_video(file, new_video, setup_duration[1], 0, setup_duration[0], "center")

    for distance in full_folder.iterdir():
        keep_middle_gt(distance, finished_dataset_path)


def main():

    # for i in Path("/media/emmahaidacher/Volume/GOOD_RESULTS/salmon_4dgs_ups").iterdir():
    #     temp_name = i.stem.split("_")[-1]
    #     os.rename(i, str(i.parent / f"temp_{temp_name}"))
    setup_duration = [3, 15]
    output_path = Path("/media/emmahaidacher/STORAGE8TB/DATASETS/MODIFIED")
    # run(output_path, setup_duration, "multiple")
    run(output_path, setup_duration, "single")

    #single_input_single_gt(Path("/media/emmahaidacher/Volume/DATASETS/datasets_4x15_single/modified_datasets"),
                    #       Path("/media/emmahaidacher/Volume/DATASETS/datasets_4x15_single/datasets"))

    # create_robo_dataset()
if __name__ == "__main__":
    main()