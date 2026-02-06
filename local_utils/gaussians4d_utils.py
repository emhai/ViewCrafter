import os
import shutil
from pathlib import Path
import subprocess
import shlex

from PIL import Image

from configs.v2v_config import GENERATED_FRAMES_DIR

PATH_TO_4DGS = Path("/home/emmahaidacher/Desktop/4DGaussians")


def run_command(cmd):
    cmd = shlex.split(cmd)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=PATH_TO_4DGS)

    print(">> returncode:", proc.returncode)
    if proc.stdout:
        print(">> stdout:\n", proc.stdout)
    if proc.stderr:
        print(">> stderr:\n", proc.stderr)

    return proc.returncode == 0


def run_4dgs(exp_name):
    # From https://github.com/hustvl/4DGaussians
    # For multipleviews scenes: If you want to train your own dataset of multipleviews scenes, you can orginize your dataset as follows:
    # ├── data
    # |   | multipleview
    # │     | (your dataset name)
    # │   	  | cam01
    # |     		  ├── frame_00001.jpg
    # │     		  ├── frame_00002.jpg
    # │     		  ├── ...
    # │   	  | cam02
    # │     		  ├── frame_00001.jpg
    # │     		  ├── frame_00002.jpg
    # │     		  ├── ...
    # │   	  | ...

    assert (PATH_TO_4DGS / "data" / "multipleview" / exp_name).exists()

    print(">> 4DGS START")

    # DATA PREPARATION
    print(">> DATA PREPARATION")
    cmd = f'conda run -n Gaussians4D bash multipleviewprogress.sh {exp_name}'
    if not run_command(cmd): return

    # TRAINING
    print(">> TRAINING")
    cmd = f'conda run -n Gaussians4D python train.py -s  data/multipleview/{exp_name} --port 6017 --expname "multipleview/{exp_name}" --configs arguments/multipleview/default.py'
    if not run_command(cmd): return

    # RENDERING
    print(">> RENDERING")
    cmd = f'conda run -n Gaussians4D python render.py --model_path "output/multipleview/{exp_name}"  --skip_train --configs arguments/multipleview/default.py'
    if not run_command(cmd): return

    # EVALUATION
    print(">> EVALUATION")
    cmd = f'conda run -n Gaussians4D python metrics.py --model_path "output/multipleview/{exp_name}"'
    if not run_command(cmd): return

    print(">> 4DGS END")


def create_4dgs_shell(exp_name, output_file):
    commands = [
        f'bash multipleviewprogress.sh {exp_name}',
        f'python train.py -s data/multipleview/{exp_name} --port 6017 --expname "multipleview/{exp_name}" --configs arguments/multipleview/default.py',
        f'python render.py --model_path "output/multipleview/{exp_name}" --skip_train --configs arguments/multipleview/default.py',
        f'python metrics.py --model_path "output/multipleview/{exp_name}"',
    ]

    shell_path = Path(output_file)
    with shell_path.open("w") as f:
        f.write("#!/bin/bash\n\n")
        for cmd in commands:
            f.write(cmd + "\n")

    shell_path.chmod(0o755)


def setup_4dgs_from_viewcrafter(frames_dir, exp_name):
    output_folder = PATH_TO_4DGS / "data" / "multipleview" / exp_name
    output_folder.mkdir(exist_ok=True)

    for folder in frames_dir.iterdir():
        if folder.is_dir():
            shutil.copytree(str(folder), str(output_folder / folder.name))


def setup_4dgs_from_videos(video_folder, exp_name):
    output_folder = PATH_TO_4DGS / "data" / "multipleview" / exp_name
    output_folder.mkdir()

    for i, video in enumerate(sorted(video_folder.iterdir()), start=1):
        target_path = output_folder / f"cam{i:02}"
        print(f"Extracting frames from {video}")
        target_path.mkdir()

        cmd = f'ffmpeg  -i {str(video)} -start_number 1 -q:v 1 -qmin 1 -qmax 1 {str(target_path)}/frame_%05d.jpg'
        run_command(cmd)


def from_png_to_jpg(folder):
    folder = Path(folder)
    for p in folder.rglob("*.png"):
        img = Image.open(p).convert("RGB")
        out_path = p.with_suffix(".jpg")
        img.save(out_path, "JPEG")
        p.unlink()


def rename_frames(folder):
    folder = Path(folder)

    files = sorted(
        (p for p in folder.rglob("frame_*.jpg")),
        key=lambda p: int(p.stem.split("_")[-1]),
        reverse=True,
    )

    for p in files:
        num = int(p.stem.split("_")[-1])
        p.rename(p.with_name(f"frame_{num + 1:05}.jpg"))


def rename_frames_from_number(folder):
    folder = Path(folder)

    files = sorted(
        (p for p in folder.rglob("000*.jpg")),
        key=lambda p: int(p.stem),
        reverse=True,
    )

    for p in files:
        num = int(p.stem)
        p.rename(p.with_name(f"frame_{num:05}.jpg"))


def test_4dgs_after_complete_run(base_dir, exp_name):
    #setup_4dgs_from_viewcrafter(base_dir, exp_name)
    #
    # original_exp_name = "original_" + exp_name
    # setup_4dgs_from_videos(self.base_dir / ORIGINAL_VIDEOS_DIR, original_exp_name)
    #
    # torch.cuda.synchronize()  # finish kernels
    # torch.cuda.empty_cache()  # release cached blocks to the driver
    # torch.cuda.ipc_collect()  # clean IPC memory
    # del self.diffusion  # todo, works?

    # run_4dgs(exp_name)
    # run_4dgs(original_exp_name)
    pass



def main():
    setup_4dgs_from_videos(Path("/mnt/data/DATASETS/MODIFIED/4D_Gaussian_ground_truth/original_ds_coffee_2"), "original_ds_coffee_2")
    # setup_4dgs_from_videos(Path("/mnt/data/DATASETS/MODIFIED/4D_Gaussian_ground_truth/original_ds_coffee_3"), "original_ds_coffee_3")
    # setup_4dgs_from_videos(Path("/mnt/data/DATASETS/MODIFIED/4D_Gaussian_ground_truth/original_ds_salmon_2"), "original_ds_salmon_2")
    # setup_4dgs_from_videos(Path("/mnt/data/DATASETS/MODIFIED/4D_Gaussian_ground_truth/original_ds_salmon_3"), "original_ds_salmon_3")

if __name__ == '__main__':
    main()
