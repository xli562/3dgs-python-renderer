import numpy as np
from tqdm import tqdm
from random import random
import matplotlib.pyplot as plt
from utils.gaussian_utils import Gaussian
from utils.camera_utils import Camera

def plot_opacity(gaussian: Gaussian, camera: Camera, w: int, h: int, 
                 bitmap: np.ndarray, alphas: np.ndarray):
    """ Computes and applies the opacity of a Gaussian object on 
    a bitmap image given a camera's view.

    Args:
        gaussian (Gaussian): The Gaussian object to plot.
        camera (Camera): The camera viewing the Gaussian.
        w (int): The width of the bitmap image.
        h (int): The height of the bitmap image.
        bitmap (np.ndarray): The bitmap image on which to draw. 
                        Shape (w, h, 3). The 3 is rgb colour.
        alphas (np.ndarray): 2D array containing alpha values for blending.

    Modifies:
        bitmap, alphas
    """
    conic, bboxsize_cam, bbox_ndc = gaussian.get_conic_and_bb(camera)

    A, B, C = conic

    screen_height, screen_width = bitmap.shape[:2]
    bbox_screen = camera.ndc_to_pixel(bbox_ndc, screen_width, screen_height)
    
    if np.any(np.isnan(bbox_screen)):   # Early exit if bitmap is corrupted
        return

    # 4 corners of the bounding box of 1 gaussian in screen coords
    ul = bbox_screen[0,:2]
    ur = bbox_screen[1,:2]
    lr = bbox_screen[2,:2]
    ll = bbox_screen[3,:2]
    
    x1 = int(np.floor(ul[0]))
    x2 = int(np.ceil(ur[0]))
    y1 = int(np.floor(ul[1]))
    y2 = int(np.ceil(ll[1]))
    nx = x2 - x1    # Number of pixls in the x direction
    ny = y2 - y1    # Number of pixls in the y direction

    # Extract out inputs for the gaussian in camera coords
    coordxy = bboxsize_cam  # This line seems to be redundant. Replacing coordxy with bboxsize_cam makes no visible difference to a 1000 Gaussian rendering result.
    x_cam_1 = coordxy[0][0]   # ul
    x_cam_2 = coordxy[1][0]   # ur
    y_cam_1 = coordxy[1][1]   # ur (y)
    y_cam_2 = coordxy[2][1]   # lr

    camera_dir = gaussian.pos - camera.position     # vector pointing from cam. to gau.
    camera_dir = camera_dir / np.linalg.norm(camera_dir)
    color = gaussian.get_color(camera_dir)

    for x, x_cam in zip(range(x1, x2), np.linspace(x_cam_1, x_cam_2, nx)):
        if x < 0 or x >= w:
            continue
        for y, y_cam in zip(range(y1, y2), np.linspace(y_cam_1, y_cam_2, ny)):
            if y < 0 or y >= h:
                continue

            # 2D Gaussian is typically calculated as 
            # f(x, y) = A * exp(-(a*x^2 + 2*b*x*y + c*y^2))
            power = -(A*x_cam**2 + C*y_cam**2)/2.0 - B * x_cam * y_cam
            if power > 0.0:     # Not a Gaussian if power > 0.0
                continue

            if gaussian.opacity < 1.0 / 255.0:
                continue
            alpha = gaussian.opacity * np.exp(power)
            alpha = min(0.99, alpha)

            # Set the pixel color to the given color and opacity
            # Do alpha blending using "over" method
            old_alpha = alphas[y, x]
            new_alpha = alpha + old_alpha * (1.0 - alpha)
            alphas[y, x] = new_alpha
            bitmap[y, x, :] = (color[0:3]) * alpha + bitmap[y, x, :] * (1.0 - alpha)


def fast_plot_opacity(gaussian:Gaussian, camera:Camera, w:int, h:int, 
                 bitmap:np.ndarray, alphas:np.ndarray,
                 tile_size:int=16):
    """ Tries to implement the original paper's rendering algorithm

    Args:
        gaussian (Gaussian): The Gaussian object to plot.
        camera (Camera): The camera viewing the Gaussian.
        w (int): The width of the bitmap image.
        h (int): The height of the bitmap image.
        bitmap (np.ndarray): The bitmap image on which to draw. 
                        Shape (w, h, 3). The 3 is rgb colour.
        alphas (np.ndarray): 2D array containing alpha values for blending.
        tile_size (int) (optional): size of square tiles

    Modifies:
        bitmap
    """
    # Currently only supports unclipped tiles, 
    # therefore w and h need to be multiples of tile_size
    if 


    conic, bboxsize_cam, bbox_ndc = gaussian.get_conic_and_bb(camera)

    A, B, C = conic

    screen_height, screen_width = bitmap.shape[:2]
    bbox_screen = camera.ndc_to_pixel(bbox_ndc, screen_width, screen_height)
    
    if np.any(np.isnan(bbox_screen)):   # Early exit if bitmap is corrupted
        return

    # 4 corners of the bounding box of 1 gaussian in screen coords
    ul = bbox_screen[0,:2]
    ur = bbox_screen[1,:2]
    lr = bbox_screen[2,:2]
    ll = bbox_screen[3,:2]
    
    x1 = int(np.floor(ul[0]))
    x2 = int(np.ceil(ur[0]))
    y1 = int(np.floor(ul[1]))
    y2 = int(np.ceil(ll[1]))
    nx = x2 - x1    # Number of pixls in the x direction
    ny = y2 - y1    # Number of pixls in the y direction

    # Extract out inputs for the gaussian in camera coords
    coordxy = bboxsize_cam  # This line seems to be redundant. Replacing coordxy with bboxsize_cam makes no visible difference to a 1000 Gaussian rendering result.
    x_cam_1 = coordxy[0][0]   # ul
    x_cam_2 = coordxy[1][0]   # ur
    y_cam_1 = coordxy[1][1]   # ur (y)
    y_cam_2 = coordxy[2][1]   # lr

    camera_dir = gaussian.pos - camera.position     # vector pointing from cam. to gau.
    camera_dir = camera_dir / np.linalg.norm(camera_dir)
    color = gaussian.get_color(camera_dir)

    for x, x_cam in zip(range(x1, x2), np.linspace(x_cam_1, x_cam_2, nx)):
        if x < 0 or x >= w:
            continue
        for y, y_cam in zip(range(y1, y2), np.linspace(y_cam_1, y_cam_2, ny)):
            if y < 0 or y >= h:
                continue

            # 2D Gaussian is typically calculated as 
            # f(x, y) = A * exp(-(a*x^2 + 2*b*x*y + c*y^2))
            power = -(A*x_cam**2 + C*y_cam**2)/2.0 - B * x_cam * y_cam
            if power > 0.0:     # Not a Gaussian if power > 0.0
                continue

            if gaussian.opacity < 1.0 / 255.0:
                continue
            alpha = gaussian.opacity * np.exp(power)
            alpha = min(0.99, alpha)

            # Set the pixel color to the given color and opacity
            # Do alpha blending using "over" method
            old_alpha = alphas[y, x]
            new_alpha = alpha + old_alpha * (1.0 - alpha)
            alphas[y, x] = new_alpha
            bitmap[y, x, :] = (color[0:3]) * alpha + bitmap[y, x, :] * (1.0 - alpha)


def gau_to_bitmap(camera, gaussian_objects:list):
    """ Sorts the Gaussian objects by depth from the perspective of the camera, 
    then plots each using plot_opacity() onto a bitmap. This function is
    optimized to handle depth sorting and alpha blending.

    Args:
        camera (Camera): The camera through which the scene is viewed.
        gaussian_objects (list of Gaussian): List of Gaussian objects to render.

    Returns:
        np.ndarray: The rendered bitmap image with the Gaussian objects.
    """
    print('Sorting the gaussians by depth')
    indices = np.argsort([gau.get_depth(camera) for gau in gaussian_objects])
    
    print('Plotting with', len(gaussian_objects), 'gaussians')
    bitmap = np.ones((camera.h, camera.w, 3), np.float32)
    alphas = np.zeros((camera.h, camera.w), np.float32)
    
    for idx in tqdm(indices):
        plot_opacity(gaussian_objects[idx], camera, camera.w, camera.h, bitmap, alphas)
    
    return bitmap

