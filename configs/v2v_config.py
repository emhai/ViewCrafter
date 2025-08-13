
OUTPUT_LOG_FILE = "output.log"          # file to log all output not necessary in terminal
CAMERA_FRAMES_DIR = "camera_frames"     # for each original video, folder with all frames
INPUTS_DIR = "inputs"                   # input for each viewcrafter iteration (1 frame from each video)
RESULTS_DIR = "results"                 # output of each viewcrafter iteration (interpolation between frame of each video)
ORIGINAL_VIDEOS_DIR = "original_videos" # copy of original videos

SEPERATED_CAMERAS_DIR = "cameras"       # newly generated videos of all positions interpolated between original video
DIFFUSION_FRAMES = "diffusion_frames"   # in cameras, stitched together diffusion frames
RENDER_FRAMES = "render_frames"         # in cameras, stitched together render frames

DEPTHS_DIR = "depths"                   # for depth images as estimated per dust3r
MASKS_DIR = "masks"                     # masks folder

PREDICTED_CAMERA_POSES_FILE = "test_camera_poses.pt" # todo
PREDICTED_FOCALS_FILE = "test_focals_poses.pt"

GUIDANCE_DIR = "guidance"
GUIDANCE_IMAGE = "guidance.png"
