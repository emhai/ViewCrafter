import os
import subprocess


def create_4dgs_setup(video_folder):

    base_path = os.path.dirname(video_folder)
    for i, video in enumerate(sorted(os.listdir(video_folder))):

        video_path = os.path.join(video_folder, video)
        target_path = os.path.join(video_folder, f"cam{i + 1:02}")
        print(f"Extracting frames from {video_path}")
        os.mkdir(target_path)

        ffmpeg_command = ['ffmpeg', '-i', video_path, f"{target_path}/frame_%05d.jpg"]
        proc = subprocess.Popen(ffmpeg_command)
        ret = proc.wait()

def main():
    path = "/home/emmahaidacher/Desktop/4DGaussians/data/multipleview/espresso_3_cams"
    create_4dgs_setup(path)

if __name__ == '__main__':
    main()