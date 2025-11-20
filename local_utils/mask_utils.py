from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from local_utils.visualization_utils import visualize_pixel_masks


def create_frame_diff_masks(current_imgs, prev_imgs, threshold=0.1, output_dir=None):
    # creates masks of shape (1, 1, H /2, W/2) # same dim as point cloud created by dust3r

    if not isinstance(current_imgs, list):
        current_imgs = [current_imgs]
        prev_imgs = [prev_imgs]

    assert len(current_imgs) == len(prev_imgs)

    all_masks = []

    for i in range(len(current_imgs)):
        image1 = current_imgs[i]
        image2 = prev_imgs[i]

        assert image1.max() <= 1.0 and image1.min() >= 0
        assert image2.max() <= 1.0 and image2.min() >= 0

        img1 = image1.permute(2, 0, 1).unsqueeze(0)  # bchw
        img2 = image2.permute(2, 0, 1).unsqueeze(0)

        abs_diff = torch.abs(img1 - img2)
        diff_mask_pixel_space = torch.sum(abs_diff, dim=1, keepdim=True)  # Shape: [1, 1, H, W]

        # Mask is 1.0 where pixels are similar, 0.0 where they are different
        # mask_pixel_space = (diff_mask_pixel_space < threshold).float()

        mask_pixel_space = (diff_mask_pixel_space > threshold).float()
        h2, w2 = mask_pixel_space.shape[2], mask_pixel_space.shape[3]
        mask_pixel_space_half = F.interpolate(mask_pixel_space.float(), size=(h2 // 2, w2 // 2),
                                              mode='nearest')  # get to same dim as pc
        print(mask_pixel_space_half.shape)

        if output_dir is not None:
            visualize_pixel_masks(mask_pixel_space_half, current_imgs[i], output_dir /f"pixel_diffs_{i}.png", "difference between first frame and current")

        all_masks.append(mask_pixel_space_half.bool().squeeze())

    return all_masks

def clean_mask(input_mask):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    closing = cv2.morphologyEx(input_mask, cv2.MORPH_CLOSE, kernel)
    dilation = cv2.dilate(closing, kernel, iterations=1)
    filled = cv2.medianBlur(dilation, 7)


    return filled

def main():

    clean_mask("")

if __name__ == "__main__":
    main()