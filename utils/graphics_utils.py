import numpy as np
from tqdm import tqdm
from utils.gaussian_utils import Gaussian
from utils.camera_utils import Camera


def plot_opacity(gaussian: Gaussian, camera: Camera, w: int, h: int, bitmap: np.ndarray, alphas: np.ndarray):
    """ Computes and applies the opacity of a Gaussian object on a bitmap image given a camera's view.

    Args:
        gaussian (Gaussian): The Gaussian object to plot.
        camera (Camera): The camera viewing the Gaussian.
        w (int): The width of the bitmap image.
        h (int): The height of the bitmap image.
        bitmap (np.ndarray): The bitmap image on which to draw.
        alphas (np.ndarray): Array containing alpha values for blending.

    Modifies the bitmap image to include the rendered Gaussian based on its opacity and camera's view. Handles
    blending of the Gaussian with the existing image content.
    """
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
    bitmap = np.zeros((camera.h, camera.w, 3), np.float32)
    alphas = np.zeros((camera.h, camera.w), np.float32)
    
    for idx in tqdm(indices):
        plot_opacity(gaussian_objects[idx], camera, camera.w, camera.h, bitmap, alphas)
    
    return bitmap
