import argparse
import os
import warnings

from skimage.metrics import structural_similarity
from math import log10, sqrt
import cv2
import numpy as np
from pytorch_fid.fid_score import calculate_fid_given_paths

import lpips
from cdfvd import fvd
from fvmd import fvmd

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

    frames_ref = sorted(original_path.iterdir())
    frames_test = sorted(synthesized_path.iterdir())
    assert len(frames_ref) == len(frames_test), "Both dirs must have same number of frames."

    mse_list = []

    for p_ref, p_test in zip(frames_ref, frames_test):
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
        return d
    else:
        return d.mean()
        # The mean distance is approximately the same as the non-spatial distance
        # Visualize a spatially-varying distance map between ex_p0 and ex_ref
        # pylab.imshow(d[0, 0, ...].data.cpu().numpy())
        # pylab.show()

# https://github.com/mseitzer/pytorch-fid
def FID(original_path, synthesized_path):
    # requires full paths
    paths = [str(original_path), str(synthesized_path)]
    fid_value = calculate_fid_given_paths(
        paths,
        batch_size=50,
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

    run(original_path, synthesized_path)


if __name__ == "__main__":
    main()