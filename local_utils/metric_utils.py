import csv
import json
import shutil
import warnings
from pathlib import Path

from skimage.metrics import structural_similarity
from math import log10, sqrt
import cv2
import numpy as np
from pytorch_fid.fid_score import calculate_fid_given_paths

import lpips
from cdfvd import fvd
from fvmd import fvmd

from configs.v2v_config import *
from local_utils.v2v_utils import ffmpeg_side_by_side_vid
from vbench import VBench

def init_results_file(base_dir):
    results_file = base_dir / RESULTS_CSV_FILE

    if not results_file.exists():
        with results_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "exp_name",
                    "GT_video",
                    "GEN_video",
                    "PSNR_mse ↑",
                    "PSNR_avg ↑"
                    "SSIM ↑",
                    "LPIPS_alex ↓",
                    "LPIPS_vgg ↓",
                ]
            )
    return results_file

def init_tot_results_file(base_dir):
    results_file = base_dir / TOT_RESULTS_CSV_FILE

    if not results_file.exists():
        with results_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "exp_name",
                    "PSNR_mse ↑",
                    "PSNR_avg ↑"
                    "SSIM ↑",
                    "LPIPS_alex ↓",
                    "LPIPS_vgg ↓",
                    "FID ↓",
                    "FVD ↓",
                    "FVMD ↓",
                    "subject_consistency ↑",
                    "background_consistency ↑",
                    "temporal_flickering ↑",
                    "motion_smoothness ↑",
                    "dynamic_degree ↑",
                    "aesthetic_quality ↑",
                    "imaging_quality ↑"
                ]
            )
    return results_file

""" 
==============================================
=================== IMAGES ===================
==============================================
"""
# https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
def calc_psnr(original_path, synthesized_path):

    original = cv2.imread(str(original_path)).astype(np.float32)
    synthesized = cv2.imread(str(synthesized_path)).astype(np.float32)

    mse = np.mean((original - synthesized) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal therefore PSNR has no importance.
        return 100.0 # no inf, if averaged inf
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

# https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
# "incorrect" version but used in academia
def calc_psnr_video(original_path, synthesized_path):
    gt_frames = sorted(original_path.iterdir())
    synthesized_frames = sorted(synthesized_path.iterdir())
    gt_frames, synthesized_frames = slice_to_same_size(gt_frames, synthesized_frames)

    psnr_list = []

    for p_ref, p_test in zip(gt_frames, synthesized_frames):
        psnr = calc_psnr(p_ref, p_test)
        psnr_list.append(psnr)

    psnr_avg = np.mean(psnr_list)
    return psnr_avg

# https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
# "correct" implementation but not used in academia
def calc_psnr_mse_video(original_path, synthesized_path):

    gt_frames = sorted(original_path.iterdir())
    synthesized_frames = sorted(synthesized_path.iterdir())
    gt_frames, synthesized_frames = slice_to_same_size(gt_frames, synthesized_frames)

    assert len(gt_frames) == len(synthesized_frames), "Both dirs must have same number of frames."

    mse_list = []

    for p_ref, p_test in zip(gt_frames, synthesized_frames):
        img_ref = cv2.imread(str(p_ref)).astype(np.float32)
        img_test = cv2.imread(str(p_test)).astype(np.float32)

        assert img_ref is not None and img_test is not None, "Failed to read one of the frames."
        assert img_ref.shape == img_test.shape, "Frame shapes must match."

        mse_frame = np.mean((img_ref - img_test) ** 2)
        mse_list.append(mse_frame)

    mse_avg = np.mean(mse_list)

    if mse_avg == 0:
        return 100.0

    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse_avg))
    return psnr

# https://stackoverflow.com/questions/71567315/how-to-get-the-ssim-comparison-score-between-two-images
def calc_ssim(original_path, synthesized_path):
    original = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
    synthesized = cv2.imread(str(synthesized_path), cv2.IMREAD_GRAYSCALE)
    assert synthesized.shape == original.shape
    assert synthesized.max() <= 255 and synthesized.min() >= 0
    assert original.max() <= 255 and original.min() >= 0

    # Compute SSIM between two images
    (score, diff) = structural_similarity(original, synthesized, full=True)

    # additional_SSIM(diff, original, synthesized)
    return score

def calc_ssim_video(original_path, synthesized_path):

    gt_frames = sorted(original_path.iterdir())
    synthesized_frames = sorted(synthesized_path.iterdir())
    gt_frames, synthesized_frames = slice_to_same_size(gt_frames, synthesized_frames)

    ssim_list = []

    for p_ref, p_test in zip(gt_frames, synthesized_frames):

        ssim = calc_ssim(p_ref, p_test)
        ssim_list.append(ssim)

    ssim_avg = np.mean(ssim_list)
    return ssim_avg

# https://github.com/richzhang/PerceptualSimilarity/blob/master/test_network.py
def calc_lpips(original_path, synthesized_path, loss_fn):

    original = lpips.im2tensor(lpips.load_image(str(original_path))).cuda()
    synthesized = lpips.im2tensor(lpips.load_image(str(synthesized_path))).cuda()

    d = loss_fn.forward(original, synthesized)

    return d.mean().cpu().detach().numpy()


def calc_lpips_video(original_path, synthesized_path, loss_fn):

    gt_frames = sorted(original_path.iterdir())
    synthesized_frames = sorted(synthesized_path.iterdir())
    gt_frames, synthesized_frames = slice_to_same_size(gt_frames, synthesized_frames)

    lpips_list = []

    for p_ref, p_test in zip(gt_frames, synthesized_frames):

        lpips = calc_lpips(p_ref, p_test, loss_fn)
        lpips_list.append(lpips)

    lpips_avg = np.mean(lpips_list)
    return lpips_avg

# https://github.com/mseitzer/pytorch-fid
def calc_fid(original_path, synthesized_path, temp_path):
    temp_dir = temp_path / "temp"
    temp_gt_frames_dir = temp_dir / "gt_frames"
    temp_syn_frames_dir = temp_dir / "syn_frames"

    temp_gt_frames_dir.mkdir(exist_ok=True, parents=True)
    temp_syn_frames_dir.mkdir(exist_ok=True, parents=True)

    for video_dir in synthesized_path.iterdir():
        if not video_dir.is_dir():
            continue

        for frame in video_dir.iterdir():
            new_name = f"{video_dir.name}_{frame.name}"
            shutil.copyfile(str(frame), str(temp_syn_frames_dir / new_name))

    for video_dir in original_path.iterdir():
        if not video_dir.is_dir():
            continue

        for frame in video_dir.iterdir():
            new_name = f"{video_dir.name}_{frame.name}"
            shutil.copyfile(str(frame), str(temp_gt_frames_dir / new_name))

    # requires full paths
    paths = [str(temp_gt_frames_dir), str(temp_syn_frames_dir)]
    fid_value = calculate_fid_given_paths(
        paths,
        batch_size=32,
        device="cuda", #todo cuda:0?
        dims=2048,
    )

    # shutil.rmtree(temp_dir)

    return fid_value

""" 
==============================================
=================== VIDEOS ===================
==============================================
"""

# https://content-debiased-fvd.github.io/documentation/
# https://github.com/songweige/content-debiased-fvd
def calc_fvd(original_path, synthesized_path):
    evaluator = fvd.cdfvd('i3d', ckpt_path=None, device='cuda')

    evaluator.compute_real_stats(evaluator.load_videos(str(original_path), data_type="video_folder"))
    evaluator.compute_fake_stats(evaluator.load_videos(str(synthesized_path),  data_type="video_folder"))
    score = evaluator.compute_fvd_from_stats()
    print(score)

    return score

def calc_fvmd(original_path, synthesized_path, log_path):
    #Folder/
    #|-- Clip1/
    #|   |-- Frame1.png/jpg
    #|   |-- Frame2.png/jpg
    #|   |-- ...
    #|
    #|-- Clip2/
    #|   |-- Frame1.png/jpg
    #|   |-- Frame2.png/jpg
    #|   |-- ...
    #|
    #|-- ...
    fvmd_value = fvmd(log_dir=str(log_path),
    gen_path=str(synthesized_path),
    gt_path=str(original_path))

    return fvmd_value

def calc_vbench(base_dir, synthesized_path):

    prompts = ['subject_consistency', 'background_consistency', 'temporal_flickering',
               'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality']

    my_VBench = VBench(device="cuda", full_info_dir=None, output_path=base_dir / MISC_DIR)
    my_VBench.evaluate(videos_path=synthesized_path, name="vbench", mode="custom_input", dimension_list=prompts)

    json_path = base_dir / MISC_DIR / "vbench_eval_results.json"

    with json_path.open("r") as f:
        data = json.load(f)

    metrics = []
    for metric_name, value_list in data.items():
        result = f"{value_list[0]:.3f}"
        metrics.append(result)

    return metrics


# https://content-debiased-fvd.github.io/documentation/
# https://github.com/songweige/content-debiased-fvd
def KVD(original_path, synthesized_path):
    pass
    # todo, laut chatgpt only nice to have


""" 
==============================================
=================== UTILS ====================
==============================================
"""

def slice_to_same_size(list1, list2):
    num_pairs = min(len(list1), len(list2))
    list1 = list1[:num_pairs]
    list2 = list2[:num_pairs]
    return list1, list2

def evaluate_frame_dirs(gt_dir, cand_dir, max_frames=None, stride=1):
    gt_frames = sorted(gt_dir.iterdir(), key=lambda p: p.name)
    cand_frames = sorted(cand_dir.iterdir(), key=lambda p: p.name)

    gt_frames, cand_frames = slice_to_same_size(gt_frames, cand_frames)

    assert gt_frames and cand_frames
    assert len(gt_frames) == len(cand_frames)

    num_pairs = len(gt_frames)
    psnr_list = []
    ssim_list = []

    for i in range(0, num_pairs, stride):
        if max_frames is not None and len(psnr_list) >= max_frames:
            break

        gt_path = gt_frames[i]
        cand_path = cand_frames[i]

        psnr_list.append(calc_psnr(gt_path, cand_path))
        ssim_list.append(calc_ssim(gt_path, cand_path))

    psnr_list = np.array(psnr_list, dtype=np.float32)
    ssim_list = np.array(ssim_list, dtype=np.float32)

    return {
        "candidate_dir": str(cand_dir),
        "num_frames": int(len(psnr_list)),
        "psnr_mean": float(psnr_list.mean()),
        "psnr_std": float(psnr_list.std()),
        "ssim_mean": float(ssim_list.mean()),
        "ssim_std": float(ssim_list.std()),
    }

def pick_best_candidate(gt_frames_dir, candidate_frame_dirs, max_frames=None, stride=1):

    results = []
    for candidate in candidate_frame_dirs.iterdir():
        metrics = evaluate_frame_dirs(
            gt_frames_dir,
            candidate,
            max_frames=max_frames,
            stride=stride,
        )
        results.append(metrics)

    # sort best-first by SSIM then PSNR
    results.sort(key=lambda m: (m["ssim_mean"], m["psnr_mean"]), reverse=True)
    return results


def run_metrics(base_dir):
    results_file = init_results_file(base_dir)
    total_results_file = init_tot_results_file(base_dir)

    loss_fn_alex = lpips.LPIPS(net="alex", spatial=True, verbose=False).cuda()
    loss_fn_vgg = lpips.LPIPS(net="vgg", spatial=True, verbose=False).cuda()

    total_psnr_avg = []
    total_psnr_mse = []
    total_ssim = []
    total_lpips_alex = []
    total_lpips_vgg = []

    for ground_truth_frame in (base_dir / GROUND_TRUTH_FRAMES_DIR).iterdir():

        ranking_rendered = pick_best_candidate(
            ground_truth_frame,
            base_dir / RENDERED_FRAMES_DIR,
            max_frames=11,
            stride=6,
        )

        # for debugging / comparison only
        ranking_generated = pick_best_candidate(
            ground_truth_frame,
            base_dir / GENERATED_FRAMES_DIR,
            max_frames=11,
            stride=6,
        )

        best_rendered = ranking_rendered[0]
        best_rendered_name = Path(best_rendered["candidate_dir"]).name

        best_generated = ranking_generated[0]
        best_generated_name = Path(best_generated["candidate_dir"]).name

        print(f"\nBest RENDERED candidate (frames) for gt {ground_truth_frame.name}: {best_rendered['candidate_dir']}")
        print(f"Best GENERATED candidate (frames) for gt {ground_truth_frame.name}: {best_generated['candidate_dir']}")

        if best_rendered_name == best_generated_name:
            print("MATCH")

        best_candidate = base_dir / GENERATED_FRAMES_DIR / best_rendered_name
        gt_name = ground_truth_frame.name

        # image-level metrics (always GT vs GENERATED on the rendered-selected camera)
        psnr_mse = calc_psnr_mse_video(ground_truth_frame, best_candidate)
        psnr_avg = calc_psnr_video(ground_truth_frame, best_candidate)
        ssim = calc_ssim_video(ground_truth_frame, best_candidate)
        lpips_alex = calc_lpips_video(ground_truth_frame, best_candidate, loss_fn_alex)
        lpips_vgg = calc_lpips_video(ground_truth_frame, best_candidate, loss_fn_vgg)

        total_psnr_avg.append(psnr_avg)
        total_psnr_mse.append(psnr_mse)
        total_ssim.append(ssim)
        total_lpips_alex.append(lpips_alex)
        total_lpips_vgg.append(lpips_vgg)

        with results_file.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                base_dir.name,
                ground_truth_frame.name,
                best_candidate.name,
                f"{psnr_mse:.3f}",
                f"{psnr_avg:.3f}",
                f"{ssim:.3f}",
                f"{lpips_alex:.3f}",
                f"{lpips_vgg:.3f}",
            ])

        gt_video = base_dir / GROUND_TRUTH_VIDEOS_DIR / f"{gt_name}.mp4"
        gen_video = base_dir / GENERATED_VIDEOS_DIR / f"{best_candidate.name}.mp4"
        ren_video = base_dir / RENDERED_VIDEOS_DIR / f"{best_candidate.name}.mp4"
        ffmpeg_side_by_side_vid(gt_video, gen_video, base_dir / VIS_RESULTS_DIR / f"gen_{gt_name}_{best_candidate.name}.mp4")
        ffmpeg_side_by_side_vid(gt_video, ren_video, base_dir / VIS_RESULTS_DIR / f"ren_{gt_name}_{best_candidate.name}.mp4")

    fid = calc_fid(base_dir / GROUND_TRUTH_FRAMES_DIR, base_dir / GENERATED_FRAMES_DIR, base_dir / MISC_DIR)
    fvd = calc_fvd(base_dir / GROUND_TRUTH_VIDEOS_DIR, base_dir / GENERATED_VIDEOS_DIR)
    fvmd = calc_fvmd(base_dir / GROUND_TRUTH_FRAMES_DIR, base_dir / GENERATED_FRAMES_DIR, base_dir / MISC_DIR)
    vbench = calc_vbench(base_dir, base_dir / GENERATED_VIDEOS_DIR)

    with total_results_file.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            base_dir.name,
            f"{np.mean(total_psnr_mse):.3f}",
            f"{np.mean(total_psnr_avg):.3f}",
            f"{np.mean(total_ssim):.3f}",
            f"{np.mean(total_lpips_alex):.3f}",
            f"{np.mean(total_lpips_vgg):.3f}",
            f"{np.mean(fid):.3f}",
            f"{fvd:.3f}",
            f"{fvmd:.3f}",
            *vbench
        ])


def run(original_path, synthesized_path):
    pass

def rerun_metrics(results_dir):

    for result in results_dir.iterdir():
        result_file = result / RESULTS_CSV_FILE
        tot_result_file = result / TOT_RESULTS_CSV_FILE
        vis_folder = result / VIS_RESULTS_DIR

        if result_file.exists():
            result_file.unlink()
        if tot_result_file.exists():
            tot_result_file.unlink()
        if vis_folder.exists():
            shutil.rmtree(vis_folder)
            vis_folder.mkdir()

        run_metrics(result)

def combine_files(results_dir):

    output_path = results_dir / "combined_results.csv"

    with output_path.open("w", newline="") as out_f:
        writer = None

        for result in sorted(results_dir.iterdir()):
            if not result.is_dir():
                continue

            tot_result_file = result / TOT_RESULTS_CSV_FILE
            assert tot_result_file.exists()

            with tot_result_file.open("r", newline="") as in_f:
                reader = csv.reader(in_f)
                header = next(reader)

                if writer is None:
                    writer = csv.writer(out_f)
                    writer.writerow(header)

                for row in reader:
                    writer.writerow(row)


def main():

    original_path = ""
    synthesized_path = ""

    # run(original_path, synthesized_path)

    base_path = Path("/media/emmahaidacher/Volume/GOOD_RESULTS/test_metrix")
    rerun_path = Path("/media/emmahaidacher/Volume/GOOD_RESULTS/all_results_with_bad")
    # run_metrics(base_path)
    # rerun_metrics(rerun_path)
    combine_files(rerun_path)

if __name__ == "__main__":
    main()