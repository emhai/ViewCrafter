from pathlib import Path

from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import os
import cv2
import numpy as np


def crop_with_red_border(path):
    img = Image.open(str(path)).convert("RGB")
    box = (340, 115, 660, 295)
    img_boxed = img.copy()
    draw = ImageDraw.Draw(img_boxed)
    draw.rectangle(box, outline=(255, 0, 0), width=4)
    crop = img.crop(box)
    img_boxed.save(str(path.parent / f"{path.stem}_w_box{path.suffix}"))
    crop.save(str(path.parent / f"{path.stem}_crop{path.suffix}"))


def save_image_difference(img_path_1, img_path_2, out_path, mode="rgb", amplify=1.0):
    if mode == "gray":
        img1 = Image.open(str(img_path_1)).convert("L")
        img2 = Image.open(str(img_path_2)).convert("L")
    else:
        img1 = Image.open(str(img_path_1)).convert("RGB")
        img2 = Image.open(str(img_path_2)).convert("RGB")

    a = np.asarray(img1, dtype=np.int16)
    b = np.asarray(img2, dtype=np.int16)

    diff = np.abs(a - b) * amplify
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    Image.fromarray(diff).save(str(out_path))


def make_horizontal_slit_scan(image_dir, output_path, slice_y=None, slice_height=4, ext=(".png", ".jpg", ".jpeg")):
    files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(ext))
    assert len(files) > 0, "No images found"

    frames = []
    for f in files:
        img = cv2.imread(os.path.join(image_dir, f))
        assert img is not None, f"Failed to load {f}"
        frames.append(img)

    h, w, c = frames[0].shape
    print(h, w, c)

    if slice_y is None:
        slice_y = h // 2

    y0 = max(0, slice_y - slice_height // 2)
    y1 = min(h, y0 + slice_height)

    stripes = []
    for img in frames:
        stripe = img[y0:y1, :, :]
        stripes.append(stripe)

    slit_scan = np.concatenate(stripes, axis=0)  # time → y-axis

    slit_scan = np.clip(slit_scan, 0, 255).astype(np.uint8)

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


def plot_metric_ablation(x, y1, y2, title, ylabel, out_path):
    plt.figure()
    plt.plot(x, y1, marker="o", label="Coffee Near")
    plt.plot(x, y2, marker="o", label="Salmon Near")
    plt.xlabel("Ablation configuration (0–4)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(str(out_path))


def ablation_multi_helper():
    x = [0, 1, 2, 3, 4]

    # FVD (↓)
    fvd_coffee = [1681, 1734, 1004, 871, 746]
    fvd_salmon = [2546, 2401, 2043, 1652, 1497]

    # FVMD (↓)
    fvmd_coffee = [816, 822, 456, 469, 239]
    fvmd_salmon = [835, 679, 519, 452, 365]

    # LPIPS (↓)
    lpips_coffee = [0.275, 0.272, 0.290, 0.269, 0.268]
    lpips_salmon = [0.273, 0.257, 0.279, 0.257, 0.264]

    # Temporal Flickering (↑)
    tf_coffee = [96.4, 96.8, 98.5, 99.0, 99.5]
    tf_salmon = [96.3, 96.7, 97.9, 98.6, 99.2]

    out_path = "Z:\\ORGA\\figures\\graphs\\multi_ablation"
    plot_metric_ablation(x, fvd_coffee, fvd_salmon, "Ablation: FVD across configurations", "FVD (↓)", f"{out_path}\\fvd.png")
    plot_metric_ablation(x, fvmd_coffee, fvmd_salmon, "Ablation: FVMD across configurations", "FVMD (↓)", f"{out_path}\\fvmd.png")
    plot_metric_ablation(x, lpips_coffee, lpips_salmon, "Ablation: LPIPS across configurations", "LPIPS (↓)", f"{out_path}\\lpips.png")
    plot_metric_ablation(x, tf_coffee, tf_salmon, "Ablation: VBench Temporal Flickering", "Temporal Flickering (%) (↑)", f"{out_path}\\temp_flicker.png")


def slice_helper():
    slice_h = 12
    dir = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\all_modifications\\generated_frames\\cam08")
    out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\all_modifications_horizontal_slice.png")
    make_horizontal_slit_scan(str(dir), str(out), slice_height=slice_h)
    dir = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\only_ddim\\generated_frames\\cam08")
    out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\only_ddim_horizontal_slice.png")
    make_horizontal_slit_scan(str(dir), str(out), slice_height=slice_h)
    dir = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\only_cfg\\generated_frames\\cam08")
    out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\only_cfg_horizontal_slice.png")
    make_horizontal_slit_scan(str(dir), str(out), slice_height=slice_h)
    dir = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\ddim_plus_cfg\\generated_frames\\cam08")
    out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\ddim_plus_cfg_horizontal_slice.png")
    make_horizontal_slit_scan(str(dir), str(out), slice_height=slice_h)
    dir = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\vanilla\\generated_frames\\cam08")
    out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\vanilla_horizontal_slice.png")
    make_horizontal_slit_scan(str(dir), str(out), slice_height=slice_h)
    dir = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\original_frames\\0001")
    out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\original_horizontal_slice.png")
    make_horizontal_slit_scan(str(dir), str(out), slice_height=slice_h)


def crop_red_box_cfg_helper():
    path = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cam_08_cfg_1.jpg")
    crop_with_red_border(path)
    path = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cam_08_cfg_29.jpg")
    crop_with_red_border(path)
    path = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cam_08_ddim_cfg_29.jpg")
    crop_with_red_border(path)


def image_diff_cfg_helper():
    path1 = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cam_08_cfg_1_crop.jpg")
    path2 = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cam_08_cfg_29_crop.jpg")
    out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cfg_diff_29.jpg")
    save_image_difference(path1, path2, out)
    path1 = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cam_08_cfg_1_crop.jpg")
    path2 = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cam_08_ddim_cfg_29_crop.jpg")
    out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cfg_ddim_diff_29.jpg")
    save_image_difference(path1, path2, out)


def main():
    ablation_multi_helper()


if __name__ == "__main__":
    main()
