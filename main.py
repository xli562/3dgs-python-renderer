from utils.gaussian_utils import *
from utils.file_utils import *

gaussians = naive_gaussians()
g_width, g_height = 1280, 720
scale_modifier = 1.0

# Iterate over the gaussians and create Gaussian objects
gaussian_objects = []
for (pos, scale, rot, opacity, sh) in zip(gaussians.xyz, gaussians.scale, gaussians.rot, gaussians.opacity, gaussians.sh):
    gau = Gaussian(pos, scale, rot, opacity, sh)
    gaussian_objects.append(gau)

print(gaussian_objects)