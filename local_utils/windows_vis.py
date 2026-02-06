from pathlib import Path

from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import os
import cv2
import numpy as np

from configs.v2v_config import PATH_TO_RESULTS


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


import os
import cv2
import numpy as np

def make_horizontal_slit_scan(
    image_dir,
    output_path,
    slice_y=None,
    slice_height=4,
    ext=(".png", ".jpg", ".jpeg"),
        frame=0
):
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

    # ---- visualize slice on first frame ----
    vis = frames[frame].copy()
    cv2.rectangle(
        vis,
        (0, y0),
        (w - 1, y1 - 1),
        (0, 0, 255),  # red (BGR)
        thickness=2,
    )

    vis_path = os.path.splitext(output_path)[0] + "_slice_vis.png"
    cv2.imwrite(vis_path, vis)

    # ---- build slit-scan ----
    stripes = []
    for img in frames:
        stripe = img[y0:y1, :, :]
        stripes.append(stripe)

    slit_scan = np.concatenate(stripes, axis=0)  # time → y-axis
    slit_scan = np.clip(slit_scan, 0, 255).astype(np.uint8)

    assert output_path.lower().endswith((".png", ".jpg", ".jpeg"))
    cv2.imwrite(output_path, slit_scan)

    print(f"Saved horizontal slit-scan to {output_path}")
    print(f"Saved slice visualization to {vis_path}")



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


def plot_metric_robo(x, yB, yN, yP, title, ylabel, out_path):
    plt.figure()
    plt.plot(x, yB, marker="o", label="B")
    plt.plot(x, yN, marker="o", label="V")
    plt.plot(x, yP, marker="o", label="P")
    plt.xlabel("Distance between input videos (cm)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x)
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.savefig(out_path)

def robo_helper():
    # x-axis
    x = [28, 33, 38, 43, 48, 53, 58]

    psnr_B = [17.99, 17.23, 17.25, 17.25, 16.31, 15.96, 15.88]
    psnr_N = [18.45, 17.50, 17.50, 17.41, 17.29, 16.19, 16.86]
    psnr_P = [18.48, 17.54, 17.54, 17.44, 17.29, 16.19, 16.83]

    ssim_B = [0.783, 0.772, 0.772, 0.772, 0.757, 0.749, 0.750]
    ssim_N = [0.795, 0.783, 0.784, 0.782, 0.777, 0.756, 0.774]
    ssim_P = [0.793, 0.782, 0.782, 0.780, 0.773, 0.753, 0.770]

    lpips_B = [0.266, 0.298, 0.303, 0.303, 0.367, 0.392, 0.386]
    lpips_N = [0.261, 0.285, 0.291, 0.308, 0.330, 0.385, 0.338]
    lpips_P = [0.254, 0.283, 0.283, 0.299, 0.322, 0.375, 0.333]

    fid_B = [58.9, 72.9, 72.2, 72.2, 95.1, 104.9, 105.2]
    fid_N = [66.4, 80.4, 83.2, 94.4, 107.9, 127.7, 125.7]
    fid_P = [65.3, 81.4, 81.4, 96.5, 103.7, 126.5, 124.8]

    fvd_B = [1185, 1213, 1150, 1150, 1292, 1413, 1549]
    fvd_N = [948, 860, 804, 847, 854, 1009, 1053]
    fvd_P = [992, 826, 826, 868, 886, 1054, 1067]

    fvmd_B = [1630, 2156, 2114, 2114, 5541, 7864, 6479]
    fvmd_N = [1134, 1403, 1183, 1427, 1529, 1946, 1931]
    fvmd_P = [1239, 1617, 1617, 1754, 1968, 2173, 2155]

    subj_B = [95.1, 94.0, 94.2, 94.1, 92.9, 92.0, 92.2]
    subj_N = [95.6, 95.1, 95.3, 95.3, 94.4, 94.4, 94.3]
    subj_P = [95.8, 95.4, 95.5, 95.6, 94.7, 94.6, 94.5]

    bg_B = [95.2, 93.1, 94.4, 94.1, 93.4, 93.4, 93.2]
    bg_N = [95.1, 94.5, 94.6, 94.7, 93.8, 93.3, 94.3]
    bg_P = [95.6, 95.1, 94.9, 95.1, 94.0, 93.8, 94.6]

    tf_B = [96.7, 96.4, 96.6, 96.4, 95.7, 95.3, 95.4]
    tf_N = [98.5, 98.4, 98.5, 98.5, 98.4, 98.4, 98.4]
    tf_P = [98.6, 98.6, 98.6, 98.6, 98.6, 98.6, 98.6]

    ms_B = [97.6, 97.3, 97.5, 97.4, 96.6, 96.4, 96.2]
    ms_N = [99.1, 99.0, 99.0, 99.0, 98.9, 98.9, 98.9]
    ms_P = [99.1, 99.1, 99.1, 99.1, 99.0, 99.0, 99.0]

    aes_B = [65.7, 66.3, 65.2, 65.0, 65.0, 63.0, 62.9]
    aes_N = [66.4, 65.8, 65.1, 62.3, 62.9, 60.7, 62.5]
    aes_P = [65.7, 65.3, 63.9, 62.2, 62.0, 60.2, 62.0]

    iq_B = [71.5, 72.4, 72.0, 71.8, 70.8, 71.6, 71.5]
    iq_N = [66.9, 68.9, 68.8, 67.7, 66.7, 66.7, 67.1]
    iq_P = [67.2, 69.2, 68.3, 68.0, 66.5, 66.5, 66.8]

    # plot_metric_robo(x, psnr_B, psnr_N, psnr_P, "Robot: PSNR vs distance", "PSNR (↑)")
    # plot_metric_robo(x, ssim_B, ssim_N, ssim_P, "Robot: SSIM vs distance", "SSIM (↑)")
    # plot_metric_robo(x, fid_B, fid_N, fid_P, "Robot: FID vs distance", "FID (↓)")
    # plot_metric_robo(x, subj_B, subj_N, subj_P, "Robot: VBench Subject Consistency vs distance","Subject Consistency (%) (↑)")
    # plot_metric_robo(x, bg_B, bg_N, bg_P, "Robot: VBench Background Consistency vs distance", "Background Consistency (%) (↑)")
    # plot_metric_robo(x, ms_B, ms_N, ms_P, "Robot: VBench Motion Smoothness vs distance", "Motion Smoothness (%) (↑)")
    # plot_metric_robo(x, aes_B, aes_N, aes_P, "Robot: VBench Aesthetic Quality vs distance", "Aesthetic Quality (%) (↑)")
    # plot_metric_robo(x, iq_B, iq_N, iq_P, "Robot: VBench Imaging Quality vs distance", "Imaging Quality (%) (↑)")

    out_path = "Z:\\ORGA\\figures\\graphs\\robot"

    plot_metric_robo(x, lpips_B, lpips_N, lpips_P, "Robot: LPIPS across distances (↓)", "LPIPS", f"{out_path}\\lpips.png")
    plot_metric_robo(x, fvd_B, fvd_N, fvd_P, "Robot: FVD across distances (↓)", "FVD", f"{out_path}\\fvd.png")
    plot_metric_robo(x, fvmd_B, fvmd_N, fvmd_P, "Robot: FVMD across distances (↓)", "FVMD",f"{out_path}\\fvmd.png")
    plot_metric_robo(x, tf_B, tf_N, tf_P, "Robot: VBench Temporal Flickering across distances (↓)", "Temporal Flickering (%)", f"{out_path}\\temp_flicker.png")


def plot_metric_ablation(x, y1, y2, title, ylabel, out_path):
    plt.figure()
    plt.plot(x, y1, marker="o", label="Coffee Near")
    plt.plot(x, y2, marker="o", label="Salmon Near")
    plt.xlabel("Configuration (0)–(4)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.2)
    plt.xticks(x)
    plt.legend()
    plt.savefig(str(out_path))

def ablation_single_helper():
    x = [0, 1, 2, 3, 4]

    # FVD (↓)
    fvd_coffee = [1642, 1857, 1710, 1077, 1027]
    fvd_spinach = [1085, 1039, 1283, 922, 934]

    # FVMD (↓)
    fvmd_coffee = [836, 832, 833, 290, 235]
    fvmd_spinach = [686, 690, 722, 559, 369]

    # LPIPS (↓)
    lpips_coffee = [0.508, 0.515, 0.518, 0.520, 0.521]
    lpips_spinach = [0.494, 0.448, 0.462, 0.445, 0.439]

    # Temporal Flickering (↑)
    tf_coffee = [96.4, 97.1, 96.6, 99.4, 99.7]
    tf_spinach = [97.9, 98.0, 97.8, 99.3, 99.5]

    out_path = Path(PATH_TO_RESULTS) / "toorga"
    plot_metric_ablation(x, fvd_coffee, fvd_spinach, "Ablation: FVD across single-video configurations (↓)", "FVD",
                         f"{out_path}/fvd_single.png")
    plot_metric_ablation(x, fvmd_coffee, fvmd_spinach, "Ablation: FVMD across single-video configurations (↓)", "FVMD",
                         f"{out_path}/fvmd_single.png")
    plot_metric_ablation(x, lpips_coffee, lpips_spinach, "Ablation: LPIPS across single-video configurations (↓)", "LPIPS",
                         f"{out_path}/lpips_single.png")
    plot_metric_ablation(x, tf_coffee, tf_spinach, "Ablation: VBench Temporal Flickering across single-video configurations (↑)",
                         "Temporal Flickering (%)", f"{out_path}/temp_flicker_single.png")


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

    out_path = Path(PATH_TO_RESULTS) / "toorga"
    plot_metric_ablation(x, fvd_coffee, fvd_salmon, "Ablation: FVD across multi-video configurations (↓)", "FVD", f"{out_path}/fvd_multi.png")
    plot_metric_ablation(x, fvmd_coffee, fvmd_salmon, "Ablation: FVMD across multi-video configurations (↓)", "FVMD", f"{out_path}/fvmd_multi.png")
    plot_metric_ablation(x, lpips_coffee, lpips_salmon, "Ablation: LPIPS across multi-video configurations (↓)", "LPIPS", f"{out_path}/lpips_multi.png")
    plot_metric_ablation(x, tf_coffee, tf_salmon, "Ablation: VBench Temporal Flickering across multi-video configurations (↑)", "Temporal Flickering (%)", f"{out_path}/temp_flicker_multi.png")


def slice_helper():
    slice_h = 12
    # slice_y = None
    # dir = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\all_modifications\\generated_frames\\cam09")
    # out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\slices_cam09\\all_modifications_horizontal_slice.png")
    # make_horizontal_slit_scan(str(dir), str(out), slice_height=slice_h, slice_y=slice_y)
    # dir = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\only_ddim\\generated_frames\\cam09")
    # out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\slices_cam09\\only_ddim_horizontal_slice.png")
    # make_horizontal_slit_scan(str(dir), str(out), slice_height=slice_h, slice_y=slice_y)
    # dir = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\only_cfg\\generated_frames\\cam09")
    # out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\slices_cam09\\only_cfg_horizontal_slice.png")
    # make_horizontal_slit_scan(str(dir), str(out), slice_height=slice_h, slice_y=slice_y)
    # dir = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\ddim_plus_cfg\\generated_frames\\cam09")
    # out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\slices_cam09\\ddim_plus_cfg_horizontal_slice.png")
    # make_horizontal_slit_scan(str(dir), str(out), slice_height=slice_h, slice_y=slice_y)
    # dir = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\vanilla\\generated_frames\\cam09")
    # out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\slices_cam09\\vanilla_horizontal_slice.png")
    # make_horizontal_slit_scan(str(dir), str(out), slice_height=slice_h, slice_y=slice_y)
    # dir = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\gt_frames\\0002")
    # out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\slices_cam09\\gt_horizontal_slice.png")
    # make_horizontal_slit_scan(str(dir), str(out), slice_height=slice_h, slice_y=278)
    # dir = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\rendered_frames\\cam09")
    # out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\\\rendered_slice.png")
    # make_horizontal_slit_scan(str(dir), str(out), slice_height=slice_h, slice_y=slice_y)

    slice_h = 12
    slice_y = None
    # dir = Path("Z:\\ORGA\\figures\\from_viewcrafter\\salmon_all\\all\\cam09")
    # out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\salmon_all\\slices_cam09\\all_modifications_horizontal_slice.png")
    # make_horizontal_slit_scan(str(dir), str(out), slice_height=slice_h, slice_y=slice_y)
    # dir = Path("Z:\\ORGA\\figures\\from_viewcrafter\\salmon_all\\ddim\\cam09")
    # out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\salmon_all\\slices_cam09\\only_ddim_horizontal_slice.png")
    # make_horizontal_slit_scan(str(dir), str(out), slice_height=slice_h, slice_y=slice_y)
    # dir = Path("Z:\\ORGA\\figures\\from_viewcrafter\\salmon_all\\cfg\\cam09")
    # out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\salmon_all\\slices_cam09\\only_cfg_horizontal_slice.png")
    # make_horizontal_slit_scan(str(dir), str(out), slice_height=slice_h, slice_y=slice_y)
    # dir = Path("Z:\\ORGA\\figures\\from_viewcrafter\\salmon_all\\cfg_ddim\\cam09")
    # out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\salmon_all\\slices_cam09\\ddim_plus_cfg_horizontal_slice.png")
    # make_horizontal_slit_scan(str(dir), str(out), slice_height=slice_h, slice_y=slice_y)
    # dir = Path("Z:\\ORGA\\figures\\from_viewcrafter\\salmon_all\\vanilla\\cam09")
    # out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\salmon_all\\slices_cam09\\vanilla_horizontal_slice.png")
    # make_horizontal_slit_scan(str(dir), str(out), slice_height=slice_h, slice_y=slice_y)
    # dir = Path("Z:\\ORGA\\figures\\from_viewcrafter\\salmon_all\\gt_frames\\0002")
    # out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\salmon_all\\slices_cam09\\gt_horizontal_slice.png")
    # make_horizontal_slit_scan(str(dir), str(out), slice_height=slice_h, slice_y=278)
    # dir = Path("Z:\\ORGA\\figures\\from_viewcrafter\\salmon_all\\cam09")
    # out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\salmon_all\\\\rendered_slice.png")
    # make_horizontal_slit_scan(str(dir), str(out), slice_height=slice_h, slice_y=slice_y)

        #mul_path = Path(PATH_TO_RESULTS) / "all_spinach_single"
        #slice_y = 400
        #frame = 10
        #new_dir = mul_path / f"slices_cam16"
        #new_dir.mkdir(exist_ok=True)
#   #
        #exp_no_l = ["0", "3" ,"7", "8", "10"]
        #exp_l = ["vanilla", "ddim", "cfg", "cfg__ddim", "cfg__ddim__latent_blending"]
        #names  = ["1_vanilla", "2_ddim", "3_cfg", "4_cfg_ddim", "5_all"]
#   #
        #for exp_no, exp, name in zip(exp_no_l, exp_l, names):
        #    path1 = mul_path / f"{exp_no}_{exp}_spinach_3x15" / "generated_frames" / "cam16"
        #    out = new_dir / f"{slice_y}_{name}.png"
        #    make_horizontal_slit_scan(str(path1), str(out), slice_height=slice_h, slice_y=slice_y, frame=frame)
#   #
        #path1 = mul_path  / "0_vanilla_spinach_3x15" / "gt_frames" / "0001"
        #out = new_dir / f"{slice_y}_gt.png"
        #make_horizontal_slit_scan(str(path1), str(out), slice_height=slice_h, slice_y=slice_y + 10, frame=frame)

    # path1 = mul_path / "0_vanilla_welder_3x15_1" / "gt_frames" / "0001"
    # out = new_dir / f"{slice_y}_gt.png"
    # make_horizontal_slit_scan(str(path1), str(out), slice_height=slice_h, slice_y=slice_y - 10)
#
    mul_path = Path("/mnt/data/RESULTS/20260206_1339_long_run_300_frames")
    slice_y = 400
#
    new_dir = mul_path / f"slice"
    new_dir.mkdir(exist_ok=True)
#
    path1 = mul_path / "generated_frames" / "cam09"
    out = new_dir / f"{slice_y}_generated.png"
    make_horizontal_slit_scan(str(path1), str(out), slice_height=slice_h, slice_y=slice_y)
#
    path1 = mul_path / "gt_frames" / "0002"
    out = new_dir / f"{slice_y}_gt.png"
    make_horizontal_slit_scan(str(path1), str(out), slice_height=slice_h, slice_y=slice_y - 10)

    # path1 = mul_path / f"0_vanilla_{exp_name}_4x15" / "gt_frames" / "1"
    # out = new_dir / f"{slice_y}_gt.png"
    # make_horizontal_slit_scan(str(path1), str(out), slice_height=slice_h, slice_y=slice_y + 20)


def crop_red_box_cfg_helper():
    path = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cam_08_cfg_1.jpg")
    crop_with_red_border(path)
    path = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cam_08_cfg_29.jpg")
    crop_with_red_border(path)
    path = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cam_08_ddim_cfg_29.jpg")
    crop_with_red_border(path)


def image_diff_cfg_helper():
    # path1 = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cam_08_cfg_1_crop.jpg")
    # path2 = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cam_08_cfg_29_crop.jpg")
    # out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cfg_diff_29.jpg")
    # save_image_difference(path1, path2, out)
    # path1 = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cam_08_cfg_1_crop.jpg")
    # path2 = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cam_08_ddim_cfg_29_crop.jpg")
    # out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\ddim_comparison\\cfg_ddim_diff_29.jpg")
    # save_image_difference(path1, path2, out)

    path1 = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\slices\\gt_horizontal_slice.png")
    # path2 = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\slices\\all_modifications_horizontal_slice.png")
    # out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\slices\\diff_gt_all.png")
    # save_image_difference(path1, path2, out)
    # path2 = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\slices\\ddim_plus_cfg_horizontal_slice.png")
    # out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\slices\\diff_gt_cfg_ddim.png")
    # save_image_difference(path1, path2, out)
    # path2 = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\slices\\only_cfg_horizontal_slice.png")
    # out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\slices\\diff_gt_cfg.png")
    # save_image_difference(path1, path2, out)
    # path2 = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\slices\\only_ddim_horizontal_slice.png")
    # out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\slices\\diff_gt_ddim.png")
    # save_image_difference(path1, path2, out)
    path2 = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\slices\\vanilla_horizontal_slice.png")
    out = Path("Z:\\ORGA\\figures\\from_viewcrafter\\coffee_all\\slices\\diff_gt_vanilla.png")
    save_image_difference(path1, path2, out)


def main():
    slice_helper()
    # image_diff_cfg_helper()
     #ablation_multi_helper()
     #ablation_single_helper()

if __name__ == "__main__":
    main()
