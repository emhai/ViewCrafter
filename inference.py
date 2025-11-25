import csv
from pathlib import Path

from configs.v2v_config import ARGS_FILE, RESULTS_CSV_FILE
from local_utils.timer import RunTimer
from local_utils.v2v_utils import init_timings_file
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

        else:
            raise KeyError(f"Invalid Mode: {opts.mode}")

    timings_file = init_timings_file(pvd.base_dir)
    times = timer.as_dict()

    t_dust3r = times.get("dust3r", 0.0)
    t_easi3r = times.get("easi3r", 0.0)
    t_ddim = times.get("ddim", 0.0)
    t_total = times.get("total", 0.0)
    t_metrics = times.get("metrics", 0.0)
    t_visualize = times.get("visualize", 0.0)
    t_diffusion = times.get("diffusion", 0.0)

    t_total = t_total - t_visualize # todo remove all visualizations?
    t_diffusion = t_diffusion - t_visualize

    t_misc = t_total - (t_diffusion + t_easi3r + t_ddim + t_dust3r + t_metrics)

    with timings_file.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            opts.exp_name,
            opts.n_frames,
            f"{t_total:.3f}",
            f"{t_easi3r:.3f}",
            f"{t_ddim:.3f}",
            f"{t_dust3r:.3f}",
            f"{t_diffusion:.3f}",
            f"{t_metrics:.3f}",
            f"{t_misc:.3f}",
        ])

