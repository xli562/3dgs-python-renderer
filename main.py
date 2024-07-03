from utils.gaussian_utils import *
from utils.file_utils import *
from utils.plotting_utils import *
from utils.camera_utils import *

from tqdm import tqdm
import matplotlib.pyplot as plt



model = load_gau_from_ply(r'D:\MyCodes\MyPythonCodes\3dgs\3dgs-acceleration\output\chair\point_cloud\iteration_7000\point_cloud.ply', 
                          10000)

print('Loading gaussians ...')
gaussian_objects = []
for (pos, scale, rot, opacity, sh) in tqdm(zip(model.xyz, model.scale, model.rot, model.opacity, model.sh)):
    gaussian_objects.append(Gaussian(pos, scale, rot, opacity, sh))

h = 720
w = 1080
# camera = Camera(h, w, position=(0.94, 1.29, -2.65), target=(0.0, 0.66, 0.07))
camera = Camera(h, w, position=(-0.57651054, 2.99040512, -0.03924271), target=(-0.0, 0.0, 0.0))

bitmap = plot_model(camera, gaussian_objects)

plt.figure(figsize=(12, 12))
plt.imshow(bitmap, vmin=0, vmax=1.0)
plt.show()