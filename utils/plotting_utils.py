import numpy as np
import matplotlib.pyplot as plt

def plot_conics_and_bbs(gaussian_objects, camera):
    """ Plots conics and bounding boxes for Gaussian 
    objects as viewed by a camera.

    Calculatess the transformation from camera coordinates 
    to screen coordinates. Draws each Gaussian's bounding box 
    and 1-sigma ellipse based on the conic section.

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