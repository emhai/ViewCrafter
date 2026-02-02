import shutil
import subprocess
from pathlib import Path

from configs.v2v_config import *

def run_one_category_all_datasets(script, dataset_path, out_dir, no_frames):

    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    for dataset in dataset_path.iterdir():
        print("running", dataset)
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


def run_one_dataset_all_categories(scripts_dir, dataset_path, out_dir, no_frames):

    if not out_dir.exists():
        out_dir.mkdir()

    img_dir = dataset_path / "input"
    gt_dir = dataset_path / "gt"
    for script in scripts_dir.iterdir():

        if str(script.stem).split("_")[0] == "temp":
            print("running", script)
            continue

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

        if "TRAJ_DIR" in text:
            text = text.replace("TRAJ_DIR", str(dataset_path / "traj.txt"))

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

def run_one(script_to_run, dataset_path, out_dir,  no_frames):
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


def exists_all(in_dir, script):
    exist = in_dir.exists() and script.exists()
    if not exist:
        print(f"Script {script} or in_dir {in_dir} does not exist")
    return exist

def main():

    # COFFEE, ALL
    # 6 * 45 + 4 * 20 = 6h
    out_dir = Path(PATH_TO_RESULTS) / "all_coffee_single"
    dataset_path = Path(PATH_TO_DATASETS) / "MODIFIED" / "datasets_3x15_single" / "important_datasets" / "coffee_3x15"
    script = Path(PATH_TO_REPO) / "master_scripts" / "single_important_few"

    if exists_all(dataset_path, script):
        run_one_dataset_all_categories(script, dataset_path, out_dir, 45)

    # SPINACH, ALL
    # 6 * 45 + 4 * 20 = 6h
    out_dir = Path(PATH_TO_RESULTS) / "all_spinach_single"
    dataset_path = Path(PATH_TO_DATASETS) / "MODIFIED" / "datasets_3x15_single" / "important_datasets" / "spinach_3x15"
    script = Path(PATH_TO_REPO) / "master_scripts" / "single_important_few"

    if exists_all(dataset_path, script):
        run_one_dataset_all_categories(script, dataset_path, out_dir, 45)

    # ALLES ZUSAMMEN: 56h
    # ANFANG 19.01. 20:00
    # ENDE   22.01. 04:00

if __name__ == "__main__":
    main()