import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.gaussian_utils import Gaussian
from utils.camera_utils import Camera

def plot_conics_and_bbs(gaussian_objects, camera):
    """ Plots conics and bounding boxes for Gaussian 
    objects as viewed by a camera.

    Calculatess the transformation from camera coordinates 
    to screen coordinates. Plots conics using equation 
    A*x**2 + B*x*y + C*y**2 = 1, where A, B, C are the 
    conic parameters. Plots the boundary where the 
    equation is satisfied

    Parameters:
        gaussian_objects (list of Gaussian): The list of Gaussian objects to plot.
        camera (Camera): The camera object that views the Gaussians.

    Returns:
        None

    Notes:
        - The function skips any Gaussian object whose conic cannot be computed.
        - It uses a predefined set of colors to distinguish different Gaussians.
    """
    ax = plt.gca()
    colors = ['r', 'g', 'b', 'y', 'm', 'c']

    for g, color in zip(gaussian_objects, colors):
        (conic, bboxsize_cam, bbox_ndc) = g.get_conic_and_bb(camera)
        if conic is None:
            continue

        A, B, C = conic
        # coordxy is the correct scale to be used with gaussian and is already
        # centered on the gaussian
        coordxy = bboxsize_cam
        x_cam = np.linspace(coordxy[0][0], coordxy[1][0], 100)
        y_cam = np.linspace(coordxy[1][1], coordxy[2][1], 100)
        X, Y = np.meshgrid(x_cam, y_cam)
        
        # 1-sigma ellipse
        F = A*X**2 + 2*B*X*Y + C*Y**2 - 3.00

        bbox_screen = camera.ndc_to_pixel(bbox_ndc)

        # Use bbox offset to position of gaussian in screen coords to position the ellipse
        x_px = np.linspace(bbox_screen[0][0], bbox_screen[1][0], 100)
        y_px = np.linspace(bbox_screen[2][1], bbox_screen[1][1], 100)
        X_px, Y_px = np.meshgrid(x_px, y_px)
        F_val = 0.0
        plt.contour(X_px, Y_px, F, [F_val], colors=color)

        # Plot a rectangle around the gaussian position based on bb
        ul = bbox_screen[0,:2]
        ur = bbox_screen[1,:2]
        lr = bbox_screen[2,:2]
        ll = bbox_screen[3,:2]
        ax.add_patch(plt.Rectangle((ul[0], ul[1]), ur[0] - ul[0], lr[1] - ur[1], fill=False, color=color))

    
def plot_opacity(gaussian: Gaussian, camera: Camera, w: int, h: int, bitmap: np.ndarray, alphas: np.ndarray):
    # Compute the opacity of a gaussian given the camera
    conic, bboxsize_cam, bbox_ndc = gaussian.get_conic_and_bb(camera)

    A, B, C = conic

    screen_height, screen_width = bitmap.shape[:2]
    bbox_screen = camera.ndc_to_pixel(bbox_ndc, screen_width, screen_height)
    
    if np.any(np.isnan(bbox_screen)):
        return

    ul = bbox_screen[0,:2]
    ur = bbox_screen[1,:2]
    lr = bbox_screen[2,:2]
    ll = bbox_screen[3,:2]
    
    y1 = int(np.floor(ul[1]))
    y2 = int(np.ceil(ll[1]))
    
    x1 = int(np.floor(ul[0]))
    x2 = int(np.ceil(ur[0]))
    nx = x2 - x1
    ny = y2 - y1

    # Extract out inputs for the gaussian
    coordxy = bboxsize_cam
    x_cam_1 = coordxy[0][0]   # ul
    x_cam_2 = coordxy[1][0]   # ur
    y_cam_1 = coordxy[1][1]   # ur (y)
    y_cam_2 = coordxy[2][1]   # lr

    camera_dir = gaussian.pos - camera.position
    camera_dir = camera_dir / np.linalg.norm(camera_dir)
    color = gaussian.get_color(camera_dir)

    for x, x_cam in zip(range(x1, x2), np.linspace(x_cam_1, x_cam_2, nx)):
        if x < 0 or x >= w:
            continue
        for y, y_cam in zip(range(y1, y2), np.linspace(y_cam_1, y_cam_2, ny)):
            if y < 0 or y >= h:
                continue

            # Gaussian is typically calculated as f(x, y) = A * exp(-(a*x^2 + 2*b*x*y + c*y^2))
            power = -(A*x_cam**2 + C*y_cam**2)/2.0 - B * x_cam * y_cam
            if power > 0.0:
                continue

            alpha = gaussian.opacity * np.exp(power)
            alpha = min(0.99, alpha)
            if gaussian.opacity < 1.0 / 255.0:
                continue

            # Set the pixel color to the given color and opacity
            # Do alpha blending using "over" method
            old_alpha = alphas[y, x]
            new_alpha = alpha + old_alpha * (1.0 - alpha)
            alphas[y, x] = new_alpha
            bitmap[y, x, :] = (color[0:3]) * alpha + bitmap[y, x, :] * (1.0 - alpha)


def plot_model(camera, gaussian_objects):
    print('Sorting the gaussians by depth')
    indices = np.argsort([gau.get_depth(camera) for gau in gaussian_objects])
    
    print('Plotting with', len(gaussian_objects), 'gaussians')
    bitmap = np.zeros((camera.h, camera.w, 3), np.float32)
    alphas = np.zeros((camera.h, camera.w), np.float32)
    
    for idx in tqdm(indices):
        plot_opacity(gaussian_objects[idx], camera, camera.w, camera.h, bitmap, alphas)
    
    return bitmap
