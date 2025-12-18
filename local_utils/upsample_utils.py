import glob
import sys
from pathlib import Path

import cv2


import os
from realesrgan import RealESRGANer
from basicsr.utils.download_util import load_file_from_url
from basicsr.archs.rrdbnet_arch import RRDBNet

def upsample_folder(in_folder, scale=2, tile=0, tile_pad=10, pre_pad=0):

    if scale == 2:
        model_name = 'RealESRGAN_x2plus'
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif scale == 4:
        model_name = 'RealESRGAN_x4plus'
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']

    model_path = os.path.join('weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            # model_path will be updated
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=None,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=True,
        gpu_id=0)

    out_folder = in_folder.parent / f"{str(in_folder.stem)}_upsampled"
    out_folder.mkdir(exist_ok=True)

    for folder in in_folder.iterdir():
        paths = sorted(list(folder.rglob('*')))
        new_folder = out_folder / folder.name
        new_folder.mkdir(exist_ok=True)

        for idx, path in enumerate(paths):
            print('Testing', str(path))
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            output, _ = upsampler.enhance(img, outscale=netscale)
            save_path = new_folder / path.name
            cv2.imwrite(str(save_path), output)


def main():

    in_dir = Path("/media/emmahaidacher/Volume/GOOD_RESULTS/salmon_4dgs_ups")
    upsample_folder(in_dir, 2)


if __name__ == "__main__":
    main()