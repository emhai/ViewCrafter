from enum import Enum

PATH_TO_REPO = "/home/emmahaidacher/Desktop/ViewCrafterFork/ViewCrafter/"
PATH_TO_GOOD_RESULTS = "/media/emmahaidacher/Volume/GOOD_RESULTS"
PATH_TO_DATASETS = "/media/emmahaidacher/Volume/DATASETS/MODIFIED_DATASETS"

OUTPUT_LOG_FILE = "output.log"                  # file to log all output not necessary in terminal
RESULTS_CSV_FILE = "results.csv"                # file for all results, metrics ...
TIMINGS_CSV_FILE = "timings.csv"                # file for all timings
ARGS_FILE = "args.json"                         # file for all input parameters

INPUTS_DIR = "inputs"                           # input for each viewcrafter iteration (1 frame from each video)
RESULTS_DIR = "results"                         # output of each viewcrafter iteration (interpolation between frame of each video)
ORIGINAL_VIDEOS_DIR = "original_videos"         # copy of original videos
ORIGINAL_FRAMES_DIR = "original_frames"         # for each original video, folder with all frames
GROUND_TRUTH_VIDEOS_DIR = "gt_videos"           # copy of ground truth videos
GROUND_TRUTH_FRAMES_DIR = "gt_frames"           # for each ground truth video, folder with all frames
RENDERED_VIDEOS_DIR = "rendered_videos"         # rendered videos
RENDERED_FRAMES_DIR = "rendered_frames"

EASI3R_RESULTS_DIR = "easi3r_results"           # results from easi3r run
EASI3R_MASKS_DIR = "easi3r_masks"               # only masks seperated to folders from easi3r runs
PICKLES_DIR = "easi3r_pickles"                  # resulting pickles from easi3r runs

VIS_RESULTS_DIR = "vis_results"

GENERATED_VIDEOS_DIR = "generated_videos"       # newly generated videos of all positions interpolated between original video
GENERATED_FRAMES_DIR = "generated_frames"       # for each generated video, folder with all frames

DIFFUSION_FRAMES = "diffusion_frames"           # in generated_videos, stitched together diffusion frames
RENDER_FRAMES = "render_frames"                 # in generated_videos, stitched together render frames

DEPTHS_DIR = "depths"                           # for depth images as estimated per dust3r
MASKS_DIR = "masks"                             # masks folder
LATENTS_DIR = "latents"                         # latents folder

class MaskType(Enum):
    COMP_WITH_PREV = 1
    COMP_WITH_FIRST = 2
    EASI3R_PREV = 3
    EASI3R_FIRST = 4

class MSAType(Enum):
    PIX_2_VIDEO = 1
    MASACTRL = 2
    ALL = 3

TARGET_W, TARGET_H = 1024, 576
TARGET_AR = TARGET_W / TARGET_H  # 16:9
