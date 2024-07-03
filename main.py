from utils.gaussian_utils import *
from utils.file_utils import *
from utils.plotting_utils import *
from utils.camera_utils import *
import matplotlib.pyplot as plt

gaussians = naive_gaussians()
g_width, g_height = 1280, 720
scale_modifier = 1.0

# Iterate over the gaussians and create Gaussian objects
gaussian_objects = []
for (pos, scale, rot, opacity, sh) in zip(gaussians.xyz, gaussians.scale, gaussians.rot, gaussians.opacity, gaussians.sh):
    gau = Gaussian(pos, scale, rot, opacity, sh)
    gaussian_objects.append(gau)

print(gaussian_objects)

camera = Camera(1920, 1080)
fig = plt.figure()
ax = plt.gca()
plot_conics_and_bbs(gaussian_objects, camera)
plt.xlim([0, camera.w])
plt.ylim([0, camera.h])
plt.grid(True)
plt.show()