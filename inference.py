from viewcrafter import ViewCrafter
import os
from configs.infer_config import get_parser
from utils.pvd_utils import *
from datetime import datetime
import json


if __name__=="__main__":
    parser = get_parser() # infer config.py
    opts = parser.parse_args()
    if opts.exp_name == None:
        prefix = datetime.now().strftime("%Y%m%d_%H%M")
        opts.exp_name = f'{prefix}_{os.path.splitext(os.path.basename(opts.image_dir))[0]}'
    opts.save_dir = os.path.join(opts.out_dir,opts.exp_name)
    os.makedirs(opts.save_dir,exist_ok=True)

    args_dict = vars(opts)
    with open(os.path.join(opts.save_dir, 'args.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)

    pvd = ViewCrafter(opts)

    if opts.mode == 'single_view_target':
        pvd.nvs_single_view()

    elif opts.mode == 'single_view_txt':
        pvd.nvs_single_view()

    elif opts.mode == 'single_view_eval':
        pvd.nvs_single_view_eval()

    elif opts.mode == 'sparse_view_interp':
        pvd.nvs_sparse_view_interp()

    elif opts.mode == 'multi_video_interp':
        pvd.run_multi_video_interp()

    elif opts.mode == 'single_video_interp':
        pvd.run_single_video_interp()


    else:
        raise KeyError(f"Invalid Mode: {opts.mode}")
