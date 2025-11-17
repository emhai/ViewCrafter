paths = ["/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/4dgs_dataset/coffee_martini",
         "/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/4dgs_dataset/cook_spinach/",
         "/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/4dgs_dataset/flame_steak/",
         "/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/SelfCap/yoga-release/videos",
         "/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/SelfCap/corgi-release/videos",
         "/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/SelfCap/bike-release/videos",
         "/media/emmahaidacher/Volume/DATASETS/INTERNET_DATASETS/06_Goats"]

names = ["coffee", "spinach", "steak", "yoga", "corgi", "bike", "goats"]

crops = ["bottom", "bottom", "bottom", "bottom", "center", "center", "center"]
ss = [2, 5, 1, 15, 22, 76, 0] # starting second
video_names = [["cam08.mp4", "cam07.mp4", "cam06.mp4", "cam00.mp4", "cam05.mp4", "cam04.mp4", "cam02.mp4"],     # coffee
               ["cam08.mp4", "cam07.mp4", "cam06.mp4", "cam00.mp4", "cam05.mp4", "cam04.mp4", "cam03.mp4"],     # spinach
               ["cam08.mp4", "cam07.mp4", "cam06.mp4", "cam00.mp4", "cam05.mp4", "cam04.mp4", "cam03.mp4"],     # steak
               ["0002.mp4", "0004.mp4", "0006.mp4", "0005.mp4", "0010.mp4", "0012.mp4", "0014.mp4"],            # yoga
               ["0019.mp4", "0008.mp4", "0015.mp4", "0013.mp4", "0011.mp4", "0009.mp4", "0017.mp4"],            # corgi
               ["0018.mp4", "0016.mp4", "0014.mp4", "0012.mp4", "0011.mp4", "0008.mp4", "0005.mp4"],            # bike
               ["camera_0016.mp4", "camera_0006.mp4", "camera_0001.mp4", "camera_0003.mp4", "camera_0010.mp4"]] # goats

distances = ["near", "middle", "far"]
setup_duration = [[4, 15]] # [[2, 30], [2, 15], [4, 15]]