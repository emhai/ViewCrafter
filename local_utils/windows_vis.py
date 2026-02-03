from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import os
import cv2


def crop_with_red_border(path):
    # Always reload the ORIGINAL image
    img = Image.open(str(path)).convert("RGB")

    box = (340, 115, 660, 295)

    img_boxed = img.copy()
    draw = ImageDraw.Draw(img_boxed)
    draw.rectangle(box, outline=(255, 0, 0), width=4)

    # ---- 2) Crop and save the same region ----
    crop = img.crop(box)
    # Overwrites the file
    img_boxed.save(str(path.parent / f"{path.stem}_w_box{path.suffix}"))
    crop.save(str(path.parent / f"{path.stem}_crop{path.suffix}"))


def save_image_difference(
    img_path_1,
    img_path_2,
    out_path,
    mode="rgb",
    amplify=1.0
):

    if mode == "gray":
        img1 = Image.open(str(img_path_1)).convert("L")
        img2 = Image.open(str(img_path_2)).convert("L")
    else:
        img1 = Image.open(str(img_path_1)).convert("RGB")
        img2 = Image.open(str(img_path_2)).convert("RGB")

    if img1.size != img2.size:
        raise ValueError("Images must have the same resolution")

    a = np.asarray(img1, dtype=np.int16)
    b = np.asarray(img2, dtype=np.int16)

    diff = np.abs(a - b) * amplify
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    Image.fromarray(diff).save(str(out_path))


def make_slit_scan(
    image_dir,
    output_path,
    slice_x=None,
    slice_width=4,
    ext=(".png", ".jpg", ".jpeg")
):
    """
    Create a slit-scan image by extracting a vertical slice from each frame.

    Args:
        image_dir (str): Folder containing frames (e.g. 45 images)
        output_path (str): Path to save slit-scan image
        slice_x (int): X position of slice (default: center)
        slice_width (int): Width of slice in pixels
        ext (tuple): Valid image extensions
    """

    # --- load frames ---
    files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(ext))
    assert len(files) > 0, "No images found"

    frames = []
    for f in files:
        img = cv2.imread(os.path.join(image_dir, f))
        assert img is not None, f"Failed to load {f}"
        frames.append(img)

    h, w, c = frames[0].shape

    # --- choose slice position ---
    if slice_x is None:
        slice_x = w // 2  # center slice by default

    x0 = max(0, slice_x - slice_width // 2)
    x1 = min(w, x0 + slice_width)

    # --- extract and stack slices ---
    slices = []
    for img in frames:
        slit = img[:, x0:x1, :]   # [H, slice_width, 3]
        slices.append(slit)

    slit_scan = np.concatenate(slices, axis=1)  # time → x-axis

    # --- save ---
    cv2.imwrite(output_path, slit_scan)
    print(f"Saved slit-scan to {output_path}")

import os
import cv2
import numpy as np

def make_horizontal_slit_scan(
    image_dir,
    output_path,
    slice_y=None,
    slice_height=4,
    ext=(".png", ".jpg", ".jpeg")
):

    # --- load frames ---
    files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(ext))
    assert len(files) > 0, "No images found"

    frames = []
    for f in files:
        img = cv2.imread(os.path.join(image_dir, f))
        assert img is not None, f"Failed to load {f}"
        frames.append(img)

    h, w, c = frames[0].shape
    print(h, w, c)

    # --- choose stripe position ---
    if slice_y is None:
        slice_y = h // 2

    y0 = max(0, slice_y - slice_height // 2)
    y1 = min(h, y0 + slice_height)

    # --- extract and stack stripes ---
    stripes = []
    for img in frames:
        stripe = img[y0:y1, :, :]   # [slice_height, W, 3]
        stripes.append(stripe)

    slit_scan = np.concatenate(stripes, axis=0)  # time → y-axis

    # --- ensure valid type ---
    slit_scan = np.clip(slit_scan, 0, 255).astype(np.uint8)

    # --- save ---
    assert output_path.lower().endswith((".png", ".jpg", ".jpeg"))
    cv2.imwrite(output_path, slit_scan)

    print(f"Saved horizontal slit-scan to {output_path}")

def stitch_with_mask():

    img_a_path = "Z:\ORGA\\figures\\from_viewcrafter\\salmon_all\\cfg_ddim\\cam09\\frame_00001.jpg"
    img_b_path = "Z:\ORGA\\figures\\from_viewcrafter\\salmon_all\\cfg_ddim\\cam09\\frame_00015.jpg"
    mask_path = "Z:\\ORGA\\figures\\from_viewcrafter\\\salmon_all\\bg_mask_0001.png"

    img_a = cv2.imread(img_a_path)
    img_b = cv2.imread(img_b_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    H, W = img_a.shape[:2]
    m = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # ensure {0,1}
    m = (m > 0).astype(img_a.dtype)

    res = img_a * m[..., None] + img_b * (1 - m[..., None])

    cv2.imwrite("upsampled_mask.png", m)
    cv2.imwrite("stitched.png", res)
def main():
    #path = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cam_08_cfg_1.jpg")
    #crop_with_red_border(path)
#
    #path = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cam_08_cfg_29.jpg")
    #crop_with_red_border(path)
#
    #path = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cam_08_ddim_cfg_29.jpg")
    #crop_with_red_border(path)
#
    #path1 = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cam_08_cfg_1_crop.jpg")
    #path2 = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cam_08_cfg_29_crop.jpg")
    #out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cfg_diff_29.jpg")
    #save_image_difference(path1, path2, out)
#
    #path1 = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cam_08_cfg_1_crop.jpg")
    #path2 = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cam_08_ddim_cfg_29_crop.jpg")
    #out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cfg_ddim_diff_29.jpg")
    # slice_h = 12
    # dir = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\all_modifications\\generated_frames\\cam08")
    # out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\all_modifications_horizontal_slice.png")
    # make_horizontal_slit_scan(str(dir), str(out), slice_height=slice_h)
#
    # dir = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\only_ddim\\generated_frames\\cam08")
    # out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\only_ddim_horizontal_slice.png")
    # make_horizontal_slit_scan(str(dir), str(out), slice_height=slice_h)
#
    # dir = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\only_cfg\\generated_frames\\cam08")
    # out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\only_cfg_horizontal_slice.png")
    # make_horizontal_slit_scan(str(dir), str(out), slice_height=slice_h)
#
    # dir = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\ddim_plus_cfg\\generated_frames\\cam08")
    # out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\ddim_plus_cfg_horizontal_slice.png")
    # make_horizontal_slit_scan(str(dir), str(out), slice_height=slice_h)
#
    # dir = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\vanilla\\generated_frames\\cam08")
    # out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\vanilla_horizontal_slice.png")
    # make_horizontal_slit_scan(str(dir), str(out), slice_height=slice_h)
#
    # dir = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\original_frames\\0001")
    # out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\original_horizontal_slice.png")
    # make_horizontal_slit_scan(str(dir), str(out), slice_height=slice_h)

    #save_image_difference(path1, path2, out)
    stitch_with_mask()
if __name__ == "__main__":
    main()
