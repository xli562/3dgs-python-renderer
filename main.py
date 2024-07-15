from utils.gaussian_utils import *
from utils.plotting_utils import *
from utils.graphics_utils import *
from utils.camera_utils import *

import os
from tqdm import tqdm
import matplotlib.pyplot as plt

ON_LINUX = os.name == 'posix'

print(ON_LINUX)
assert 0

print('Loading gaussians ...')
model:GaussianData = load_gau_from_ply(r'D:\MyCodes\MyPythonCodes\3dgs\3dgs-acceleration\output\chair\point_cloud\iteration_7000\point_cloud.ply', 
                          -1)
gaussian_objects = []
for (pos, scale, rot, opacity, sh) in tqdm(zip(model.xyz, model.scale, model.rot, model.opacity, model.sh), smoothing=0.6):
    gaussian_objects.append(Gaussian(pos, scale, rot, opacity, sh))

h, w = 720, 1080
camera = Camera(h, w, position=(0.94, 1.29, -2.65), target=(0.0, 0.66, 0.07))
# camera = Camera(h, w, position=(-0.57651054, 2.99040512, -0.03924271), target=(-0.0, 0.0, 0.0))

bitmap = gau_to_bitmap(camera, gaussian_objects)

plt.figure(figsize=(12, 12))
plt.imshow(bitmap, vmin=0, vmax=1.0)
plt.show()