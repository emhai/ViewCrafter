single_paths = ["/media/emmahaidacher/STORAGE8TB/DATASETS/INTERNET/NEURAL_3D_VIDEO/coffee_martini",
                "/media/emmahaidacher/STORAGE8TB/DATASETS/INTERNET/NEURAL_3D_VIDEO/cook_spinach/",
                "/media/emmahaidacher/STORAGE8TB/DATASETS/INTERNET/NEURAL_3D_VIDEO/flame_steak/",
                "/media/emmahaidacher/STORAGE8TB/DATASETS/INTERNET/NEURAL_3D_VIDEO/flame_salmon_1/",
                "/media/emmahaidacher/STORAGE8TB/DATASETS/INTERNET/SELFCAP/yoga-release/videos",
                "/media/emmahaidacher/STORAGE8TB/DATASETS/INTERNET/GOOGLE_LIGHTFIELD/06_Goats",
                "/media/emmahaidacher/STORAGE8TB/DATASETS/INTERNET/GOOGLE_LIGHTFIELD/01_Welder",
                "/media/emmahaidacher/STORAGE8TB/DATASETS/INTERNET/GOOGLE_LIGHTFIELD/02_Flames",
                "/media/emmahaidacher/STORAGE8TB/DATASETS/INTERNET/GOOGLE_LIGHTFIELD/07_Car",
                ]

single_names = ["coffee", "spinach", "steak", "salmon", "yoga", "goats", "welder", "flames", "car"]

single_crops = ["bottom", "bottom", "bottom", "bottom",  "bottom", "center", "center", "center", "center"]

single_ss = [2, 5, 1, 11, 15, 0, 10, 4, 2] # starting second

single_video_names = [["cam05.mp4", "cam00.mp4"],                   # coffee
                      ["cam00.mp4", "cam05.mp4"],                   # spinach
                      ["cam00.mp4", "cam06.mp4"],                   # steak
                      ["cam05.mp4", "cam00.mp4"],                   # salmon
                      ["0005.mp4", "0006.mp4"],                     # yoga
                      ["camera_0001.mp4", "camera_0003.mp4"],       # goats
                      ["camera_0001.mp4", "camera_0006.mp4"],       # welder
                      ["camera_0006.mp4", "camera_0001.mp4"],       # flames
                      ["camera_0006.mp4", "camera_0001.mp4"]        # car
                      ]

#               -coffee-    -spinach-   -steak-     -salmon-    -yoga-  -goats- -welder-    -flames-    -car-
single_trajs = [[-5],       [6],        [-6],       [-6],       [-8],   [-4],   [4],        [-4],       [-6]]

single_distances = ["near", "middle", "far", "v_far", "vv_far"]
