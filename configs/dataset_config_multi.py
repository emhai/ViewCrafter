paths = ["/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/4dgs_dataset/coffee_martini",
         "/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/4dgs_dataset/cook_spinach/",
         "/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/4dgs_dataset/flame_steak/",
         "/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/4dgs_dataset/flame_salmon_1/",
         "/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/SelfCap/yoga-release/videos",
         "/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/SelfCap/bike-release/videos",
         "/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/google_lightfield/06_Goats_rect",
         "/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/google_lightfield/01_Welder_rect",
         "/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/google_lightfield/02_Flames_rect",
         "/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/google_lightfield/07_Car_rect",
         "/media/emmahaidacher/Volume/DATASETS/OWN_DATASETS/harp/synched1/only_videos"
         ]

names = ["coffee", "spinach", "steak", "salmon", "yoga", "bike", "goats", "welder", "flames", "car", "harp"]

crops = ["bottom", "bottom", "bottom", "bottom",  "bottom", "center", "center", "center", "center", "center", "center"]
ss = [2, 5, 1, 11, 15, 76, 0, 10, 4, 2, 15] # starting second
video_names = [["cam07.mp4", "cam06.mp4", "cam00.mp4", "cam05.mp4", "cam04.mp4"],                               # coffee
               ["cam07.mp4", "cam06.mp4", "cam00.mp4", "cam05.mp4", "cam04.mp4"],                               # spinach
               ["cam08.mp4", "cam06.mp4", "cam00.mp4", "cam05.mp4", "cam03.mp4"],                               # steak
               ["cam07.mp4", "cam06.mp4", "cam00.mp4", "cam05.mp4", "cam04.mp4"],                               # salmon
               ["0006.mp4", "0005.mp4", "0010.mp4"],                                                            # yoga
               ["0014.mp4", "0012.mp4", "0011.mp4"],                                                            # bike
               ["camera_0006.mp4", "camera_0001.mp4", "camera_0003.mp4"],                                       # goats
               ["camera_0006.mp4", "camera_0001.mp4", "camera_0003.mp4"],                                       # welder
               ["camera_0006.mp4", "camera_0001.mp4", "camera_0003.mp4"],                                       # flames
               ["camera_0006.mp4", "camera_0001.mp4", "camera_0003.mp4"],                                       # car
               ["3.mp4", "4.mp4", "5.mp4"]                                                                      # harp
               ]

distances = ["near", "middle", "far"]
setup_duration = [[4, 15]] # [[2, 30], [2, 15], [4, 15]]