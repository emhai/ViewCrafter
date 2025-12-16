from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from configs.v2v_config import EASI3R_MASKS_DIR, EASI3R_RESULTS_DIR, VIS_RESULTS_DIR
from local_utils.v2v_utils import numeric_key
from local_utils.visualization_utils import visualize_pixel_masks
from torchvision.transforms import CenterCrop
import torchvision.transforms as transforms


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

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))

    closing = cv2.morphologyEx(input_mask, cv2.MORPH_CLOSE, kernel)
    dilation = cv2.dilate(closing, kernel, iterations=2)
    filled = cv2.medianBlur(dilation, 9)

    return filled

def rendered_mask_to_binary(rendered_mask):

    threshold = 1e-6
    return (rendered_mask.abs() > threshold).any(dim=-1)

def binary_mask_to_latent(binary_mask, noise_shape):

    _, _, n, h, w = noise_shape
    binary_mask = binary_mask.float()
    binary_mask = binary_mask.unsqueeze(0).unsqueeze(0)

    mask_latent = F.interpolate(
        binary_mask,
        size=(n, h, w),
        mode='nearest'
    )

    return mask_latent

def create_bg_mask(base_dir):
    mask_dir = base_dir / EASI3R_RESULTS_DIR
    all_folders = list(mask_dir.glob("*"))
    amount = len(list(all_folders[0].glob("*")))
    all_masks = []
    for i in range(amount):
        for folder in sorted(mask_dir.iterdir()):
            all_masks.append(list(folder.glob("*"))[i])

        print(all_masks)
        all_masks = []

def compute_bg_mask_last_n_frames(base_dir, current_frame=None, n_last_frames=None, H=288, W=512):

    easier_results = base_dir / EASI3R_RESULTS_DIR
    if current_frame is not None:
        assert n_last_frames is not None
        m, n = max(current_frame - n_last_frames, 0), current_frame
    else:
        m, n = None, None

    all_masks = []
    for folder in easier_results.iterdir():
        dyn_mask_folder = folder / "frames_dynamic_masks"

        dyn_mask_folder = sorted(list(dyn_mask_folder.iterdir()), key=numeric_key)[m:n]
        combined_mask = None
        for mask in dyn_mask_folder:
            easier_mask = Image.open(str(mask)).convert("L")
            crop = CenterCrop((H, W))
            cropped_mask = crop(easier_mask)

            to_tensor = transforms.ToTensor()  # Converts to float tensor in range [0, 1]
            mask_tensor = to_tensor(cropped_mask)

            if combined_mask is None:
                combined_mask = mask_tensor
            else:
                combined_mask += mask_tensor
        combined_mask = combined_mask.clamp(0, 1)
        combined_mask = combined_mask.squeeze()
        combined_mask = combined_mask > 0
        all_masks.append(combined_mask)

        mask_np = (combined_mask.numpy() * 255.0).astype(np.uint8)
        img = Image.fromarray(mask_np, mode="L")
        img.save(base_dir / VIS_RESULTS_DIR / f"bg_mask_{str(folder.stem)}.png")


    return all_masks


def main():
    compute_bg_mask_last_n_frames(Path("/media/emmahaidacher/Volume/GOOD_RESULTS/20251014_1554_yoga_debug"), 10, 12)

if __name__ == "__main__":
    main()