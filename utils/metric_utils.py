import csv
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
from utils.v2v_utils import init_results_file, ffmpeg_side_by_side_vid

""" 
==============================================
=================== IMAGES ===================
==============================================
"""
# https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
def PSNR(original_path, synthesized_path):

    original = cv2.imread(str(original_path)).astype(np.float32)
    synthesized = cv2.imread(str(synthesized_path)).astype(np.float32)

    mse = np.mean((original - synthesized) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal therefore PSNR has no importance.
        return float('inf') # return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

# https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
def PSNR_video(original_path, synthesized_path):

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
def SSIM(original_path, synthesized_path):
    original = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
    synthesized = cv2.imread(str(synthesized_path), cv2.IMREAD_GRAYSCALE)
    assert synthesized.shape == original.shape
    assert synthesized.max() <= 255 and synthesized.min() >= 0
    assert original.max() <= 255 and original.min() >= 0

    # Compute SSIM between two images
    (score, diff) = structural_similarity(original, synthesized, full=True)

    # additional_SSIM(diff, original, synthesized)
    return score

def SSIM_video(original_path, synthesized_path):

    gt_frames = sorted(original_path.iterdir())
    synthesized_frames = sorted(synthesized_path.iterdir())
    gt_frames, synthesized_frames = slice_to_same_size(gt_frames, synthesized_frames)

    ssim_list = []

    for p_ref, p_test in zip(gt_frames, synthesized_frames):

        ssim = SSIM(p_ref, p_test)
        ssim_list.append(ssim)

    ssim_avg = np.mean(ssim_list)
    return ssim_avg

# https://github.com/richzhang/PerceptualSimilarity/blob/master/test_network.py
def LPIPS(original_path, synthesized_path):

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

def LPIPS_video(original_path, synthesized_path):

    gt_frames = sorted(original_path.iterdir())
    synthesized_frames = sorted(synthesized_path.iterdir())
    gt_frames, synthesized_frames = slice_to_same_size(gt_frames, synthesized_frames)

    lpips_list = []

    for p_ref, p_test in zip(gt_frames, synthesized_frames):

        lpips = LPIPS(p_ref, p_test)
        lpips_list.append(lpips)

    lpips_avg = np.mean(lpips_list)
    return lpips_avg


# https://github.com/mseitzer/pytorch-fid
def FID(original_path, synthesized_path):
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
def FVD(original_path, synthesized_path):
    evaluator = fvd.cdfvd('videomae', ckpt_path=None, device='cuda')

    evaluator.compute_real_stats(evaluator.load_videos(str(original_path)))
    evaluator.compute_fake_stats(evaluator.load_videos(str(synthesized_path)))
    score = evaluator.compute_fvd_from_stats()

    return score

def FVMD(original_path, synthesized_path, log_path):
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
    psnrs = []
    ssims = []

    for i in range(0, num_pairs, stride):
        if max_frames is not None and len(psnrs) >= max_frames:
            break

        gt_path = gt_frames[i]
        cand_path = cand_frames[i]

        psnrs.append(PSNR(gt_path, cand_path))
        ssims.append(SSIM(gt_path, cand_path))


    psnrs = np.array(psnrs, dtype=np.float32)
    ssims = np.array(ssims, dtype=np.float32)

    return {
        "candidate_dir": str(cand_dir),
        "num_frames": int(len(psnrs)),
        "psnr_mean": float(psnrs.mean()),
        "psnr_std": float(psnrs.std()),
        "ssim_mean": float(ssims.mean()),
        "ssim_std": float(ssims.std()),
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
    gt_frames_dir = base_dir / GROUND_TRUTH_FRAMES_DIR
    gen_frames_dir = base_dir / GENERATED_FRAMES_DIR
    gt_videos_dir = base_dir / GROUND_TRUTH_VIDEOS_DIR
    gen_videos_dir = base_dir / GENERATED_VIDEOS_DIR
    vis_results_dir = base_dir / VIS_RESULTS_DIR

    results_file = init_results_file(base_dir)

    total_psnr = []
    total_ssim = []
    total_lpips = []
    total_fid = []
    for ground_truth_frame in gt_frames_dir.iterdir():

        ranking = pick_best_candidate(
            ground_truth_frame,
            gen_frames_dir,
            max_frames=10,
            stride=10,
        )

        for r in ranking:
            print(
                r["candidate_dir"],
                "frames:", r["num_frames"],
                "PSNR:", r["psnr_mean"],
                "SSIM:", r["ssim_mean"],
            )

        best = ranking[0]['candidate_dir']
        print(f"Best candidate (frames) for gt {ground_truth_frame.name}: {best}")

        PSNR_vid = PSNR_video(ground_truth_frame, Path(best))
        SSIM_vid = SSIM_video(ground_truth_frame, Path(best))
        LPIPS_vid = LPIPS_video(ground_truth_frame, Path(best))
        FID_vid = FID(ground_truth_frame, Path(best))

        total_psnr.append(PSNR_vid)
        total_ssim.append(SSIM_vid)
        total_lpips.append(LPIPS_vid)
        total_fid.append(FID_vid)

        with results_file.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                base_dir.name,
                ground_truth_frame.name,
                Path(best).name,
                f"{PSNR_vid:.3f}",
                f"{SSIM_vid:.3f}",
                f"{LPIPS_vid:.3f}",
                f"{FID_vid:.3f}"
            ])

        gt_name = ground_truth_frame.name
        gen_name = Path(best).name
        gt_video = gt_videos_dir / f"{gt_name}.mp4"
        gen_video = gen_videos_dir / f"{gen_name}.mp4"
        ffmpeg_side_by_side_vid(gt_video, gen_video, vis_results_dir / f"{gt_name}_{gen_name}.mp4")

    with results_file.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            base_dir.name,
            "AVERAGE",
            "",
            f"{np.mean(total_psnr):.3f}",
            f"{np.mean(total_ssim):.3f}",
            f"{np.mean(total_lpips):.3f}",
            f"{np.mean(total_fid):.3f}"
        ])

def run(original_path, synthesized_path):
    warnings.filterwarnings("ignore", category=UserWarning) # in torchvision "Arguments other than a weight enum ... deprecated"
    warnings.filterwarnings("ignore", category=FutureWarning) # in lpips "You are using torch.load with weights_onl=False ... deprecated"

    lpips = LPIPS(original_path, synthesized_path)
    lpips = lpips.item()
    psnr= PSNR(original_path, synthesized_path)
    ssim = SSIM(original_path, synthesized_path)
    # print(f"For files {original_name}, {synthesized_name}: PSNR: {psnr:.3f}, SSIM: {ssim:.3f}, LPIPS: {lpips:.3f}")
    return lpips, psnr, ssim

def main():

    original_path = ""
    synthesized_path = ""

    # run(original_path, synthesized_path)

    base_path = Path("/media/emmahaidacher/Volume/GOOD_RESULTS/20251118_1507_spinach_2_metrics")
    run_metrics(base_path)

if __name__ == "__main__":
    main()