from pathlib import Path

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

import torch.nn.functional as F

def save_masks(mask_list, save_dir, visualize=True, save=True):
    save_dir = Path(save_dir)
    save_dir.mkdir()

    for i, msk in enumerate(mask_list):
        if isinstance(msk, torch.Tensor):
            msk_np = msk.cpu().detach().numpy()
        else:
            msk_np = msk

        msk_img = (msk_np * 255).astype(np.uint8)

        if save:
            mask_img = Image.fromarray(msk_img)
            mask_img.save(save_dir / f"mask_{i}.png")

        if visualize:
            plt.imshow(msk_np, cmap='gray')
            plt.title(f"Mask {i}")
            plt.show()

def save_depth(depth_list, save_dir, visualize=True, save=True):
    save_dir = Path(save_dir)
    save_dir.mkdir()

    for i, dpt in enumerate(depth_list):
        dpt_np = dpt.cpu().detach().numpy()
        dpt_norm = ((dpt_np - dpt_np.min()) / (dpt_np.ptp() + 1e-8) * 255).astype(np.uint8)

        if save:
            depth_img = Image.fromarray(dpt_norm)
            depth_img.save(save_dir / f"depth_{i}.png")

        if visualize:
            plt.imshow(dpt_np, cmap='plasma')
            plt.title(f"Depth Map {i}")
            plt.colorbar()
            plt.show()

def visualize_pixel_masks(full_res_mask, image, path, title):

    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(image.numpy())
    axs[0].set_title("Original image")
    axs[0].axis("off")

    axs[1].imshow(full_res_mask.squeeze().numpy(), cmap='gray')
    axs[1].set_title(title)
    axs[1].axis("off")

    mask_resized = F.interpolate(full_res_mask, size=(image.shape[0], image.shape[1]), mode='nearest')  # shape: (1,1,768,1024)

    # Step 2 — squeeze to H,W
    mask_2d = mask_resized.squeeze() # shape: (768,1024)
    axs[2].imshow(image.numpy())  # original image
    axs[2].imshow(mask_2d, cmap='Reds', alpha=0.5)  # transparent red mask
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)

def visualize_masks_horizontal(masks, path, cmap=None):

    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()

    n = masks.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(n * 5, 5))

    for i in range(n):
        axes[i].imshow(masks[i], cmap=cmap)
        axes[i].axis("off")
        axes[i].set_title(f"Mask {i}")

    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
