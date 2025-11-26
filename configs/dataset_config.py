paths = ["/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/4dgs_dataset/coffee_martini",
         "/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/4dgs_dataset/cook_spinach/",
         "/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/4dgs_dataset/flame_steak/",
         "/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/SelfCap/yoga-release/videos",
         "/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/SelfCap/bike-release/videos",
         "/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/google_lightfield/06_Goats_rect",
         "/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/google_lightfield/01_Welder_rect",
         "/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/google_lightfield/02_Flames_rect",
         "/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/google_lightfield/07_Car_rect",
         ]

names = ["coffee", "spinach", "steak", "yoga", "bike", "goats", "welder", "flames", "car"]

crops = ["bottom", "bottom", "bottom", "bottom", "center", "center", "center", "center", "center"]
ss = [2, 5, 1, 15, 76, 0, 10, 4, 2] # starting second
video_names = [["cam07.mp4", "cam06.mp4", "cam00.mp4", "cam05.mp4", "cam04.mp4"],                               # coffee
               ["cam07.mp4", "cam06.mp4", "cam00.mp4", "cam05.mp4", "cam04.mp4"],                               # spinach
               ["cam08.mp4", "cam06.mp4", "cam00.mp4", "cam05.mp4", "cam03.mp4"],                               # steak
               ["0006.mp4", "0005.mp4", "0010.mp4"],                                                            # yoga
               ["0014.mp4", "0012.mp4", "0011.mp4"],                                                            # bike
               ["camera_0006.mp4", "camera_0001.mp4", "camera_0003.mp4"],                                       # goats
               ["camera_0006.mp4", "camera_0001.mp4", "camera_0003.mp4"],                                       # welder
               ["camera_0006.mp4", "camera_0001.mp4", "camera_0003.mp4"],                                       # flames
               ["camera_0006.mp4", "camera_0001.mp4", "camera_0003.mp4"],                                       # car
               ]

distances = ["near", "middle", "far"]
setup_duration = [[3, 15]] # [[2, 30], [2, 15], [4, 15]]