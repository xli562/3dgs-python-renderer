from utils.gaussian_utils import *
from utils.plotting_utils import *
from utils.graphics_utils import *
from utils.camera_utils import *

import os, argparse
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

ON_LINUX = os.name == 'posix'
PLY_PATH = r'D:\MyCodes\MyPythonCodes\3dgs\3dgs-acceleration\output\chair\point_cloud\iteration_7000\point_cloud.ply' if not ON_LINUX else '/home/xl562/chair_point_cloud.ply'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D Gaussian Renderer')
    parser.add_argument('-i', type=int, required=False, default=1000, help='sample size from original point cloud, default=1000, complete sampling: -1')
    parser.add_argument('--show', type=bool, required=False, default=False, help='show rendered image with matplotlib')
    parser.add_argument('--store', type=bool, required=False, default=True, help='store rendered image with timestamp in ./output')
    args = parser.parse_args()
    sample_size = args.i
    show_img = args.show
    store_img = args.store

    print('Loading gaussians ...')
    model:GaussianData = load_gau_from_ply(PLY_PATH, sample_size)
    gaussian_objects = []
    for (pos, scale, rot, opacity, sh) in tqdm(zip(model.xyz, model.scale, model.rot, model.opacity, model.sh), smoothing=0.6):
        gaussian_objects.append(Gaussian(pos, scale, rot, opacity, sh))

    h, w = 720, 1080
    camera = Camera(h, w, position=(0.94, 1.29, -2.65), target=(0.0, 0.66, 0.07))
    # camera = Camera(h, w, position=(-0.57651054, 2.99040512, -0.03924271), target=(-0.0, 0.0, 0.0))

    bitmap = gau_to_bitmap(camera, gaussian_objects)

    if show_img:
        plt.figure(figsize=(12, 12))
        plt.imshow(bitmap, vmin=0, vmax=1.0)
        plt.show()
    if store_img:
        bitmap_normalized = (bitmap * 255 / np.max(bitmap)).astype(np.uint8)
        img = Image.fromarray(bitmap_normalized)
        output_filename = datetime.now().strftime('%Y-%m-%d_%H.%M.%S.%f')[:-3]
        img.save(f'./output/{os.name}, {output_filename}.jpg')
        print(f'File saved as {os.name}, {output_filename}.jpg')