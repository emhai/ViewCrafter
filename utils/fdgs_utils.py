import os
import shutil
from pathlib import Path
import subprocess
import shlex

from PIL import Image

from configs.v2v_config import DIFFUSION_FRAMES

PATH_TO_4DGS = Path("/home/emmahaidacher/Desktop/4DGaussians")


def run_command(cmd):
    cmd = shlex.split(cmd)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=PATH_TO_4DGS)

    print(">> returncode:", proc.returncode)
    if proc.stdout:
        print(">> stdout:\n", proc.stdout)
    if proc.stderr:
        print(">> stderr:\n", proc.stderr)

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
    run_command(cmd)

    # TRAINING
    print(">> TRAINING")
    cmd = f'conda run -n Gaussians4D python train.py -s  data/multipleview/{exp_name} --port 6017 --expname "multipleview/{exp_name}" --configs arguments/multipleview/default.py'
    run_command(cmd)

    # RENDERING
    print(">> RENDERING")
    cmd = f'conda run -n Gaussians4D python render.py --model_path "output/multipleview/{exp_name}"  --skip_train --configs arguments/multipleview/default.py'
    run_command(cmd)

    # EVALUATION
    print(">> EVALUATION")
    cmd = f'conda run -n Gaussians4D python metrics.py --model_path "output/multipleview/{exp_name}"'
    run_command(cmd)

    print(">> 4DGS END")


def setup_4dgs_from_viewcrafter(cameras_path, exp_name):

    output_folder = PATH_TO_4DGS / "data" / "multipleview" / exp_name
    output_folder.mkdir()

    diffusion_folder = cameras_path / DIFFUSION_FRAMES
    for folder in diffusion_folder.iterdir():
        if folder.is_dir():
            shutil.copytree(str(folder), str(output_folder / folder.name))

    
def setup_4dgs_from_videos(video_folder, exp_name):

    output_folder = PATH_TO_4DGS / "data" / "multipleview" / exp_name
    output_folder.mkdir()

    for i, video in enumerate(sorted(video_folder.iterdir()), start=1):

        target_path = output_folder / f"cam{i:02}"
        print(f"Extracting frames from {video}")
        target_path.mkdir()

        cmd = f'ffmpeg -start_number 1 -i {str(video)} {str(target_path)}/frame_%05d.jpg'
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

def main():
    path = Path("/media/emmahaidacher/Volume/GOOD_RESULTS/20250917_1637_espresso_single_4dgs/cameras/")
    # setup_4dgs_from_videos(path, "test")

    # from_png_to_jpg(path)
    # rename_frames(path)
    setup_4dgs_from_viewcrafter(path, "test2")
    run_4dgs("test2")

if __name__ == '__main__':
    main()