import shutil
import subprocess
from pathlib import Path

from fontTools.unicodedata import script

from configs.v2v_config import *
from configs.dataset_config import *
import time
import csv

from local_utils.metric_utils import rerun_metrics


def run_one_category_all_datasets():
    script = Path(PATH_TO_REPO / "master_scripts/multiple/9_cfg__ddim__latent_blending.sh")
    out_dir = Path(PATH_TO_GOOD_RESULTS / "9_ALL_RESULTS")
    dataset_path = Path(PATH_TO_DATASETS / "near_datasets")

    if not out_dir.exists():
        out_dir.mkdir()

    img_dir = dataset_path / "input"
    gt_dir = dataset_path / "gt"
    no_frames = 60
    for dataset in dataset_path.iterdir():
        script_name = script.name.split(".")[0]
        exp_name = dataset.name + "_" + script_name
        text = script.read_text()

        replacements = {
            "IMAGE_DIR": str(img_dir),
            "GT_DIR": str(gt_dir),
            "OUT_DIR": str(out_dir),
            "EXP_NAME": exp_name,
            "NUMBER_FRAMES": str(no_frames),
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        modified = script.parent / f"temp_{script.name}"
        new_text = f"cd {PATH_TO_REPO}\n" + text
        modified.write_text(new_text)
        modified.chmod(0o755)
        try:
            subprocess.run(["bash", str(modified)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Script {modified} failed with return code {e.returncode}")


def run_one_dataset_all_categories():
    scripts_dir = Path( PATH_TO_REPO / "master_scripts/multiple")
    out_dir = Path(PATH_TO_GOOD_RESULTS / "COFFEE_RESULTS_ALL")
    dataset_path = Path(PATH_TO_DATASETS / "near_datasets/coffee_4x15_near")

    if not out_dir.exists():
        out_dir.mkdir()

    img_dir = dataset_path / "input"
    gt_dir = dataset_path / "gt"
    no_frames = 60
    for script in scripts_dir.iterdir():
        script_name = script.name.split(".")[0]
        exp_name = dataset_path.name + "_" + script_name
        text = script.read_text()

        replacements = {
            "IMAGE_DIR": str(img_dir),
            "GT_DIR": str(gt_dir),
            "OUT_DIR": str(out_dir),
            "EXP_NAME": exp_name,
            "NUMBER_FRAMES": str(no_frames),
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        modified = scripts_dir / f"temp_{script.name}"
        new_text = f"cd {PATH_TO_REPO}\n" + text
        modified.write_text(new_text)
        modified.chmod(0o755)
        try:
            subprocess.run(["bash", str(modified)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Script {modified} failed with return code {e.returncode}")

def run_all(datasets_path, out_dir, scripts_to_run):
    scripts_dir = Path(PATH_TO_REPO) / "master_scripts" / "multiple"
    if not out_dir.exists():
        out_dir.mkdir()

    for dataset in datasets_path.iterdir():
        img_dir = dataset / "input"
        gt_dir = dataset / "gt"
        no_frames = 45
        for script in scripts_dir.iterdir():

            script_name = int(script.name.split("_")[0])
            if script_name not in scripts_to_run:
                continue

            exp_name = f"{script.stem}_{dataset.name}"
            text = script.read_text()

            replacements = {
                "IMAGE_DIR": str(img_dir),
                "GT_DIR": str(gt_dir),
                "OUT_DIR": str(out_dir),
                "EXP_NAME": exp_name,
                "NUMBER_FRAMES": str(no_frames),
            }
            for old, new in replacements.items():
                text = text.replace(old, new)

            modified = scripts_dir / f"temp_{script.name}"
            new_text = f"cd {PATH_TO_REPO}\n" + text
            modified.write_text(new_text)
            modified.chmod(0o755)
            try:
                subprocess.run(["bash", str(modified)], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Script {modified} failed with return code {e.returncode}")
            finally:
                try:
                    modified.unlink()  # or modified.unlink(missing_ok=True) on Python 3.8+
                except FileNotFoundError:
                    pass

def run_one(dataset_path, out_dir, script_to_run):
    if not out_dir.exists():
        out_dir.mkdir()

    img_dir = dataset_path / "input"
    gt_dir = dataset_path / "gt"
    no_frames = 45

    exp_name = f"{script_to_run.stem}_{dataset_path.name}"
    text = script_to_run.read_text()
    replacements = {
        "IMAGE_DIR": str(img_dir),
        "GT_DIR": str(gt_dir),
        "OUT_DIR": str(out_dir),
        "EXP_NAME": exp_name,
        "NUMBER_FRAMES": str(no_frames),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    modified = script_to_run.parent / f"temp_{script_to_run.name}"
    new_text = f"cd {PATH_TO_REPO}\n" + text
    modified.write_text(new_text)
    modified.chmod(0o755)
    try:
        subprocess.run(["bash", str(modified)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Script {modified} failed with return code {e.returncode}")
    finally:
        try:
            modified.unlink()  # or modified.unlink(missing_ok=True) on Python 3.8+
        except FileNotFoundError:
            pass

def main():

    # out_dir = Path(PATH_TO_GOOD_RESULTS) / "results_26_11"
    # datasets_path = Path(PATH_TO_DATASETS) / "near_middle_coffee_yoga_goats"
    # to_run_coffee = [4, 9, 10]
    # run_all(datasets_path, out_dir, to_run_coffee)

    to_run_all = [9, 10]
    datasets_path = Path("/media/emmahaidacher/Volume/DATASETS/MODIFIED_DATASETS_3x15/datasets")

    out_dir = Path("/media/emmahaidacher/Volume/GOOD_RESULTS/results_26_11")
    run_one(Path("/media/emmahaidacher/Volume/DATASETS/MODIFIED_DATASETS_3x15/datasets/harp_3x15_near"),
            out_dir, Path("/home/emmahaidacher/Desktop/ViewCrafterFork/ViewCrafter/master_scripts/multiple/8_cfg__ddim.sh"))
    #run_all(datasets_path, out_dir, to_run_all)
    datasets_path = Path("/media/emmahaidacher/Volume/DATASETS/MODIFIED_DATASETS_3x15/datasets")
    inputs = ["steak_3x15_middle", "steak_3x15_near", "welder_3x15_near", "yoga_3x15_near", "harp_3x15_near"]
    scripts = ["/home/emmahaidacher/Desktop/ViewCrafterFork/ViewCrafter/master_scripts/multiple/9_cfg__latent_blending.sh",
               "/home/emmahaidacher/Desktop/ViewCrafterFork/ViewCrafter/master_scripts/multiple/10_cfg__ddim__latent_blending.sh"]

    # for i in inputs:
    #     for s in scripts:
    #         run_one(datasets_path / i, out_dir, Path(s))
#
    # rerun_metrics(Path("/media/emmahaidacher/Volume/GOOD_RESULTS/results_20_11"))

if __name__ == "__main__":
    main()