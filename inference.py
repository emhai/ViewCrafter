import csv
import shutil
from pathlib import Path

from configs.v2v_config import *
from local_utils.gaussians4d_utils import setup_4dgs_from_viewcrafter, run_4dgs, PATH_TO_4DGS
from local_utils.metric_utils import run_metrics
from local_utils.timer_utils import RunTimer
from local_utils.upsample_utils import upsample_folder_realesrgan, upsample_folder_cv2
from local_utils.v2v_utils import separate_cameras, ffmpeg_nxn_video
from viewcrafter import ViewCrafter
import os
from configs.infer_config import get_parser
from local_utils.pvd_utils import *
from datetime import datetime
import json


if __name__=="__main__":
    parser = get_parser() # infer config.py
    opts = parser.parse_args()
    if opts.exp_name == None:
        prefix = datetime.now().strftime("%Y%m%d_%H%M")
        opts.exp_name = f'{prefix}_{os.path.splitext(os.path.basename(opts.image_dir))[0]}'

    opts.save_dir = os.path.join(opts.out_dir, opts.exp_name)

    if os.path.exists(opts.save_dir):
        prefix = datetime.now().strftime("%Y%m%d_%H%M")
        opts.exp_name = f'{prefix}_{opts.exp_name}'
        opts.save_dir = os.path.join(opts.out_dir, opts.exp_name)

    os.makedirs(opts.save_dir,exist_ok=True)# todo exist not okay

    args_dict = vars(opts)
    with open(os.path.join(opts.save_dir, ARGS_FILE), 'w') as f:
        json.dump(args_dict, f, indent=4)

    timer = RunTimer()
    with timer.time("total"):
        pvd = ViewCrafter(opts, timer)

        if opts.mode == 'single_view_target':
            pvd.nvs_single_view()

        elif opts.mode == 'single_view_txt':
            pvd.nvs_single_view()

        elif opts.mode == 'single_view_eval':
            pvd.nvs_single_view_eval()

        elif opts.mode == 'sparse_view_interp':
            pvd.nvs_sparse_view_interp()

        elif opts.mode == 'multi_video_interp':
            pvd.run_video_interp("multi")

        elif opts.mode == 'single_video_interp':
            pvd.run_video_interp("single")
            shutil.copyfile(opts.traj_txt, os.path.join(opts.save_dir, 'traj.txt'))

        else:
            raise KeyError(f"Invalid Mode: {opts.mode}")

    print("Deleting ViewCrafter object")
    base_dir = pvd.base_dir
    del pvd

    print("Cleaning GPU up")
    torch.cuda.synchronize()  # finish kernels
    torch.cuda.empty_cache()  # release cached blocks to the driver
    torch.cuda.ipc_collect()  # clean IPC memory

    separate_cameras(base_dir, DIFFUSION_FRAMES)
    separate_cameras(base_dir, RENDER_FRAMES)

    if opts.gt_dir is not None:
        with timer.time("metrics"):
            run_metrics(base_dir)

    with timer.time("visualize"):
        ffmpeg_nxn_video(base_dir / GENERATED_VIDEOS_DIR, base_dir / VIS_RESULTS_DIR)

    with timer.time("upsample"):
        upsample_folder_realesrgan(base_dir / GENERATED_FRAMES_DIR, 2)
        upsample_folder_cv2(base_dir / GENERATED_FRAMES_DIR, 2, "cubic")
        upsample_folder_cv2(base_dir / GENERATED_FRAMES_DIR, 2, "lanczos")
        upsample_folder_cv2(base_dir / GENERATED_FRAMES_DIR, 2, "linear")

    with timer.time("4dgs"):
        exp_name = f"{opts.exp_name}_lanczos"
        setup_4dgs_from_viewcrafter(base_dir / f"{GENERATED_FRAMES_DIR}_upsampled_lanczos", exp_name)
        run_4dgs(exp_name)
        # /home/emmahaidacher/Desktop/4DGaussians/output/multipleview/yoga_mul_lanczos/video/ours_14000/video_rgb.mp4
        path_to_4dgs_video = PATH_TO_4DGS / "output" / "multipleview" / exp_name / "video" / "ours_14000" / "video_rgb.mp4"
        if path_to_4dgs_video.exists():
            shutil.copyfile(path_to_4dgs_video, base_dir / VIS_RESULTS_DIR / "4dgs_render_result.mp4")


    # todo, time everything? 4dgs, visualize
    # Create timings file
    timer.finish_timings(base_dir, opts.exp_name, opts.n_frames)
