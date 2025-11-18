import shutil
import subprocess
from pathlib import Path
from configs.v2v_config import *
from configs.dataset_config import *
import time
import csv


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

def run_all():
    scripts_dir = Path(PATH_TO_REPO / "master_scripts/multiple")
    out_dir = Path(PATH_TO_GOOD_RESULTS / "NEAR_RESULTS_ALL")
    if not out_dir.exists():
        out_dir.mkdir()
    datasets_path = Path(PATH_TO_DATASETS / "near_datasets")

    for dataset in datasets_path.iterdir():
        img_dir = dataset / "input"
        gt_dir = dataset / "gt"
        no_frames = 60
        for script in scripts_dir.iterdir():
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

            modified = scripts_dir / f"temp_{script.name}"
            modified.write_text(text)
            modified.chmod(0o755)
            subprocess.run(["bash", str(modified)], check=True)


def main():
    run_one_dataset_all_categories()

if __name__ == "__main__":
    main()