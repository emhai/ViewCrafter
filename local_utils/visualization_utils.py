import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms.functional as TF

from configs.v2v_config import LATENTS_DIR, MASKS_DIR, EASI3R_MASKS_DIR


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

def visualize_latents(base_dir, intermediates, model, prefix="x_inter"):
    out_dir = base_dir / LATENTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    gif_frames = []

    for i, x in enumerate(intermediates):
        with torch.no_grad():
            pixel = model.decode_first_stage(x)  # [B, C, T, H, W] or [B, C, H, W]

        pixel = pixel[0].detach().cpu().float()  # -> [C, T, H, W] or [C, H, W]

        if pixel.ndim == 4:
            # [C, T, H, W] -> pick middle frame
            C, T, H, W = pixel.shape
            t_mid = T // 2
            frame = pixel[:, t_mid]  # [C, H, W]
        else:
            frame = pixel  # [C, H, W]

        f_min = frame.min()
        f_max = frame.max()
        denom = (f_max - f_min).clamp(min=1e-8)
        frame_norm = (frame - f_min) / denom  # in [0, 1]
        save_image(frame_norm, out_dir / f"{prefix}_{i:03d}.png")

        pil_img = TF.to_pil_image(frame)
        gif_frames.append(pil_img)


    rgb_images = [img.convert('RGB') for img in gif_frames]
    gif_path = out_dir / f"{prefix}_latents.gif"
    rgb_images[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=4,
        loop=0,
    )
    for img in gif_frames:
            img.close()



def main():
    pass

    
if __name__ == '__main__':
    main()