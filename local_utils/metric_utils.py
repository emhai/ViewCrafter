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
                    "PSNR ↑",
                    "SSIM ↑",
                    "LPIPS ↓",
                    "FID ↓"
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
                    "PSNR ↑",
                    "SSIM ↑",
                    "LPIPS ↓",
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
        return float('inf') # return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

# https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
def calc_psnr_video(original_path, synthesized_path):

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
        return float("inf")

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
def calc_lpips(original_path, synthesized_path):

    spatial = True         # Return a spatial map of perceptual distance.
    loss_fn = lpips.LPIPS(net='alex', spatial=spatial, verbose=False).cuda()  # Can also set net = 'squeeze' or 'vgg'

    original = lpips.im2tensor(lpips.load_image(str(original_path)))
    original = original.cuda()
    synthesized = lpips.im2tensor(lpips.load_image(str(synthesized_path)))
    synthesized = synthesized.cuda()

    d = loss_fn.forward(original, synthesized)

    if not spatial:
        return d.cpu().detach().numpy()
    else:
        return d.mean().cpu().detach().numpy() # todo necessary?
        # The mean distance is approximately the same as the non-spatial distance
        # Visualize a spatially-varying distance map between ex_p0 and ex_ref
        # pylab.imshow(d[0, 0, ...].data.cpu().numpy())
        # pylab.show()

def calc_lpips_video(original_path, synthesized_path):

    gt_frames = sorted(original_path.iterdir())
    synthesized_frames = sorted(synthesized_path.iterdir())
    gt_frames, synthesized_frames = slice_to_same_size(gt_frames, synthesized_frames)

    lpips_list = []

    for p_ref, p_test in zip(gt_frames, synthesized_frames):

        lpips = calc_lpips(p_ref, p_test)
        lpips_list.append(lpips)

    lpips_avg = np.mean(lpips_list)
    return lpips_avg

# https://github.com/mseitzer/pytorch-fid
def calc_fid(original_path, synthesized_path):
    # requires full paths
    paths = [str(original_path), str(synthesized_path)]
    fid_value = calculate_fid_given_paths(
        paths,
        batch_size=32,
        device="cuda", #todo cuda:0?
        dims=2048,
    )
    return fid_value

""" 
==============================================
=================== VIDEOS ===================
==============================================
"""

# https://content-debiased-fvd.github.io/documentation/
# https://github.com/songweige/content-debiased-fvd
def calc_fvd(original_path, synthesized_path):
    evaluator = fvd.cdfvd('videomae', ckpt_path=None, device='cuda')

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

    total_psnr = []
    total_ssim = []
    total_lpips = []
    total_fid = []
    for ground_truth_frame in (base_dir / GROUND_TRUTH_FRAMES_DIR).iterdir():

        ranking_generated = pick_best_candidate(
            ground_truth_frame,
            base_dir / GENERATED_FRAMES_DIR,
            max_frames=11,
            stride=6,
        )

        ranking_rendered = pick_best_candidate(
            ground_truth_frame,
            base_dir / RENDERED_FRAMES_DIR,
            max_frames=11,
            stride=6,
        )

        print("Generated candidates:")
        for r in ranking_generated:
            print(
                r["candidate_dir"],
                "frames:", r["num_frames"],
                "PSNR:", r["psnr_mean"],
                "SSIM:", r["ssim_mean"],
            )

        print("\nRendered candidates:")
        for r in ranking_rendered:
            print(
                r["candidate_dir"],
                "frames:", r["num_frames"],
                "PSNR:", r["psnr_mean"],
                "SSIM:", r["ssim_mean"],
            )

        best_generated = ranking_generated[0]
        best_rendered = ranking_rendered[0]

        # Choose the better one (here: prioritize PSNR, then SSIM)
        def score(r):
            return (r["psnr_mean"], r["ssim_mean"])

        best_overall = max([best_generated, best_rendered], key=score)

        print(f"\nBest candidates (frames) for gt {ground_truth_frame.name}: {best_generated['candidate_dir']} and {best_rendered['candidate_dir']}")
        print("WINNER:", "GENERATED" if best_overall is best_generated else "RENDERED")

        best = best_overall['candidate_dir']
        print(f"Best candidate (frames) for gt {ground_truth_frame.name}: {best}")

        gt_name = ground_truth_frame.name
        best_candidate = Path(best)

        psnr = calc_psnr_video(ground_truth_frame, best_candidate)
        ssim = calc_ssim_video(ground_truth_frame, best_candidate)
        lpips = calc_lpips_video(ground_truth_frame, best_candidate)
        fid = calc_fid(ground_truth_frame, best_candidate)

        total_psnr.append(psnr)
        total_ssim.append(ssim)
        total_lpips.append(lpips)
        total_fid.append(fid)

        with results_file.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                base_dir.name,
                ground_truth_frame.name,
                best_candidate.name,
                f"{psnr:.3f}",
                f"{ssim:.3f}",
                f"{lpips:.3f}",
                f"{fid:.3f}"
            ])

        gt_video = base_dir / GROUND_TRUTH_VIDEOS_DIR / f"{gt_name}.mp4"
        gen_video = base_dir / GENERATED_VIDEOS_DIR / f"{best_candidate.name}.mp4"
        ffmpeg_side_by_side_vid(gt_video, gen_video, base_dir / VIS_RESULTS_DIR / f"{gt_name}_{best_candidate.name}.mp4")

    # calculate only once over all videos (distribution)
    fvd = calc_fvd(base_dir / GROUND_TRUTH_VIDEOS_DIR, base_dir / GENERATED_VIDEOS_DIR)
    fvmd = calc_fvmd(base_dir / GROUND_TRUTH_FRAMES_DIR, base_dir / GENERATED_FRAMES_DIR, base_dir / MISC_DIR)
    vbench = calc_vbench(base_dir, base_dir / GENERATED_VIDEOS_DIR)

    with total_results_file.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            base_dir.name,
            f"{np.mean(total_psnr):.3f}",
            f"{np.mean(total_ssim):.3f}",
            f"{np.mean(total_lpips):.3f}",
            f"{np.mean(total_fid):.3f}",
            f"{fvd:.3f}",
            f"{fvmd:.3f}",
            *vbench
        ])

def run(original_path, synthesized_path):
    warnings.filterwarnings("ignore", category=UserWarning) # in torchvision "Arguments other than a weight enum ... deprecated"
    warnings.filterwarnings("ignore", category=FutureWarning) # in lpips "You are using torch.load with weights_onl=False ... deprecated"

    lpips = calc_lpips(original_path, synthesized_path)
    lpips = lpips.item()
    psnr= calc_psnr(original_path, synthesized_path)
    ssim = calc_ssim(original_path, synthesized_path)
    # print(f"For files {original_name}, {synthesized_name}: PSNR: {psnr:.3f}, SSIM: {ssim:.3f}, LPIPS: {lpips:.3f}")
    return lpips, psnr, ssim

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

        for result in results_dir.iterdir():
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
    rerun_path = Path("/media/emmahaidacher/Volume/GOOD_RESULTS/results_18_11")
    # run_metrics(base_path)
    rerun_metrics(rerun_path)
    combine_files(rerun_path)

if __name__ == "__main__":
    main()