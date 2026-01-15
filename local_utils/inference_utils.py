import shutil
import subprocess
from pathlib import Path

from configs.v2v_config import *

def run_one_category_all_datasets(script, dataset_path, out_dir, no_frames):

    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    for dataset in dataset_path.iterdir():
        img_dir = dataset / "input"
        gt_dir = dataset / "gt"
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

        if "TRAJ_DIR" in text:
            text = text.replace("TRAJ_DIR", str(dataset / "traj.txt"))

        modified = script.parent / f"temp_{script.name}"
        new_text = f"cd {PATH_TO_REPO}\n" + text
        modified.write_text(new_text)
        modified.chmod(0o755)
        try:
            subprocess.run(["bash", str(modified)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Script {modified} failed with return code {e.returncode}")


def run_one_dataset_all_categories(dataset_path, out_dir):
    scripts_dir = Path(PATH_TO_REPO) / "master_scripts" / "multiple_short"

    if not out_dir.exists():
        out_dir.mkdir()

    img_dir = dataset_path / "input"
    gt_dir = dataset_path / "gt"
    no_frames = 60
    for script in scripts_dir.iterdir():
        exp_name = f"{script.stem}_{dataset_path.name}"
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

def run_all(datasets_path, out_dir, scripts_to_run, no_frames):
    scripts_dir = Path(PATH_TO_REPO) / "master_scripts" / "multiple"
    if not out_dir.exists():
        out_dir.mkdir()

    for dataset in datasets_path.iterdir():
        img_dir = dataset / "input"
        gt_dir = dataset / "gt"
        for script in sorted(scripts_dir.iterdir(), reverse=True):

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

def run_one(dataset_path, out_dir, script_to_run, no_frames):
    if not out_dir.exists():
        out_dir.mkdir()

    img_dir = dataset_path / "input"
    gt_dir = dataset_path / "gt"

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

    out_dir = Path("/mnt/data/RESULTS/1401/robosapiens")
    dataset_path = Path("/mnt/data/DATASETS/MODIFIED/datasets_4x15_multiple_robosapiens/datasets_without_3")
    script = Path("/home/emmahaidacher/Desktop/ViewCrafterFork/ViewCrafter/master_scripts/multiple_important/0_vanilla.sh")
    run_one_category_all_datasets(script, dataset_path, out_dir, 60)

    dataset_path = Path("/mnt/data/DATASETS/MODIFIED/datasets_4x15_multiple_robosapiens/final_datasets")
    script = Path("/home/emmahaidacher/Desktop/ViewCrafterFork/ViewCrafter/master_scripts/multiple_important/8_cfg__ddim.sh")
    run_one_category_all_datasets(script, dataset_path, out_dir, 60)

    out_dir = Path("/mnt/data/RESULTS/1401/single")
    dataset_path = Path("/mnt/data/DATASETS/MODIFIED/datasets_3x15_single/final_datasets")
    script = Path("/home/emmahaidacher/Desktop/ViewCrafterFork/ViewCrafter/master_scripts/single_important/10_cfg__ddim__latent_blending.sh")
    run_one_category_all_datasets(script, dataset_path, out_dir, 45)

    out_dir = Path("/mnt/data/RESULTS/1401/multiple")
    script = Path("/home/emmahaidacher/Desktop/ViewCrafterFork/ViewCrafter/master_scripts/multiple_important/0_vanilla.sh")
    dataset_path = Path("/mnt/data/DATASETS/MODIFIED/datasets_3x15_multiple/working_datasets")
    run_one_category_all_datasets(script, dataset_path, out_dir, 45)


if __name__ == "__main__":
    main()