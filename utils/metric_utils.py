import argparse
import os
import warnings

from skimage.metrics import structural_similarity
from math import log10, sqrt
import cv2
import numpy as np

import lpips

# https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
def PSNR(original_path, synthesized_path):

    original = cv2.imread(original_path).astype(np.float32)
    synthesized = cv2.imread(synthesized_path).astype(np.float32)

    mse = np.mean((original - synthesized) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal therefore PSNR has no importance.
        return float('inf') # return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

# https://stackoverflow.com/questions/71567315/how-to-get-the-ssim-comparison-score-between-two-images
def SSIM(original_path, synthesized_path):

    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    synthesized = cv2.imread(synthesized_path, cv2.IMREAD_GRAYSCALE)
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
    loss_fn = lpips.LPIPS(net='alex', spatial=spatial, verbose=False)  # Can also set net = 'squeeze' or 'vgg'

    original = lpips.im2tensor(lpips.load_image(original_path))
    synthesized = lpips.im2tensor(lpips.load_image(synthesized_path))

    d = loss_fn.forward(original, synthesized)

    if not spatial:
        return d
    else:
        return d.mean()
        # The mean distance is approximately the same as the non-spatial distance
        # Visualize a spatially-varying distance map between ex_p0 and ex_ref
        # pylab.imshow(d[0, 0, ...].data.cpu().numpy())
        # pylab.show()

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