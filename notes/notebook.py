import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import util
from util import Camera
from util_gau import load_ply, naive_gaussian, GaussianData

gaussians = naive_gaussian()
g_width, g_height = 1280, 720
scale_modifier = 1.0
gaussians = naive_gaussian()
g_width, g_height = 1280, 720
scale_modifier = 1.0

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2_0 = 1.0925484305920792
SH_C2_1 = -1.0925484305920792
SH_C2_2 = 0.31539156525252005
SH_C2_3 = -1.0925484305920792
SH_C2_4 = 0.5462742152960396
SH_C3_0 = -0.5900435899266435
SH_C3_1 = 2.890611442640554
SH_C3_2 = -0.4570457994644658
SH_C3_3 = 0.3731763325901154
SH_C3_4 = -0.4570457994644658
SH_C3_5 = 1.445305721320277
SH_C3_6 = -0.5900435899266435

class Gaussian:
    """
    Represents a Gaussian object with properties in 3D space including position, scale, rotation, opacity, and spherical harmonics.

    Attributes:
        pos (np.ndarray): The position vector of the Gaussian in 3D space.
        scale (np.ndarray): Scale factors applied along the Gaussian's axes.
        rot (Rotation): A scipy.spatial.transform.Rotation object representing the rotation.
        opacity (float): The opacity level of the Gaussian.
        sh (np.ndarray): Spherical harmonics coefficients.
        cov3D (np.ndarray): The covariance matrix in 3D.
    """

    def __init__(self, pos, scale, rot, opacity, sh):
        """
        Initializes the Gaussian object with provided position, scale, rotation, opacity, and spherical harmonics coefficients.

        Parameters:
            pos (np.ndarray): 3D position vector.
            scale (np.ndarray): Scaling factors along each axis.
            rot (np.ndarray): Quaternion rotation coefficients (s, x, y, z).
            opacity (np.ndarray): Opacity value.
            sh (np.ndarray): Spherical harmonics coefficients.
        """
        
        self.pos = np.array(pos)
        self.scale = np.array(scale_modifier * scale)
        # Initialize scipy Quaternion from rot (s, x, y, z)
        self.rot = Rotation.from_quat([rot[1], rot[2], rot[3], rot[0]])
        self.opacity = opacity[0]
        self.sh = np.array(sh)
        self.cov3D = self.compute_cov3d()

    def compute_cov3d(self):
        """
        Computes the covariance matrix in 3D based on the Gaussian's scale and rotation.

        Returns:
            np.ndarray: The computed covariance matrix.
        """
        cov3D = np.diag(self.scale**2)
        cov3D = self.rot.as_matrix().T @ cov3D @ self.rot.as_matrix()
        return cov3D

    def get_cov2d(self, camera):
        """
        Projects the 3D covariance matrix to a 2D covariance matrix using the camera's view matrix.

        Parameters:
            camera (Camera): The camera object to provide view matrix and other camera-specific parameters.

        Returns:
            np.ndarray: The 2D covariance matrix after projection.
        """
        view_mat = camera.get_view_matrix()
        g_pos_w = np.append(self.pos, 1.0)
        # g_pos_cam = camera.world_to_cam(self.pos)
        g_pos_cam = view_mat @ g_pos_w
        view_matrix = camera.get_view_matrix()
        [htan_fovx, htan_fovy, focal] = camera.get_htanfovxy_focal()
        focal_x = focal_y = focal

        t = np.copy(g_pos_cam)

        limx = 1.3 * htan_fovx
        limy = 1.3 * htan_fovy
        txtz = t[0]/t[2]
        tytz = t[1]/t[2]

        tx = min(limx, max(-limx, txtz)) * t[2]
        ty = min(limy, max(-limy, tytz)) * t[2]
        tz = t[2]

        J = np.array([
            [focal_x/tz, 0.0, -(focal_x * tx)/(tz * tz)],
            [0.0, focal_y/tz, -(focal_y * ty)/(tz * tz)],
            [0.0, 0.0, 0.0]
        ])
        W = view_matrix[:3, :3].T
        T = W @ J
        cov = T.T @ self.cov3D.T @ T

        cov[0,0] += 0.3
        cov[1,1] += 0.3
        return cov[:2, :2]

    def get_depth(self, camera):
        """
        Calculates the depth of the Gaussian's position relative to the camera.

        Parameters:
            camera (Camera): The camera object to provide the view matrix.

        Returns:
            float: Depth of the Gaussian in camera view space.
        """
        view_matrix = camera.get_view_matrix()
        
        position4 = np.append(self.pos, 1.0)
        g_pos_view = view_matrix @ position4
        depth = g_pos_view[2]
        return depth

    def get_conic_and_bb(self, camera):
        """
        Computes the conic representation and bounding box of the Gaussian in normalized device coordinates.

        Parameters:
            camera (Camera): The camera object to provide camera parameters and matrices.

        Returns:
            tuple: Conic representation, bounding box in camera space, and bounding box in normalized device coordinates.
        """
        cov2d = self.get_cov2d(camera)

        det = np.linalg.det(cov2d)
        if det == 0.0:
            return None
        
        det_inv = 1.0 / det
        cov = [cov2d[0,0], cov2d[0,1], cov2d[1,1]]
        conic = np.array([cov[2] * det_inv, -cov[1] * det_inv, cov[0] * det_inv])
        # cov_inv= np.linalg.inv(cov2d)
        # conic = np.array([cov_inv[0,0], cov_inv[0,1], cov_inv[1,1]])
        # compute 3-sigma bounding box size
        bboxsize_cam = np.array([3.0 * np.sqrt(cov2d[0,0]), 3.0 * np.sqrt(cov2d[1,1])])
        # bboxsize_cam = np.array([3.0 * np.sqrt(cov[0]), 3.0 * np.sqrt(cov[2])])        
        # Divide out camera plane size to get bounding box size in NDC
        wh = np.array([camera.w, camera.h])
        bboxsize_ndc = np.divide(bboxsize_cam, wh) * 2

        vertices = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])
        # Four coordxy values (used to evaluate gaussian, also in camera space coordinates)
        bboxsize_cam = np.multiply(vertices, bboxsize_cam)

        # compute g_pos_screen and gl_position
        view_matrix = camera.get_view_matrix()
        projection_matrix = camera.get_projection_matrix()

        position4 = np.append(self.pos, 1.0)
        g_pos_view = view_matrix @ position4
        g_pos_screen = projection_matrix @ g_pos_view
        g_pos_screen = g_pos_screen / g_pos_screen[3]
        
        bbox_ndc = np.multiply(vertices, bboxsize_ndc) + g_pos_screen[:2]
        bbox_ndc = np.hstack((bbox_ndc, np.zeros((vertices.shape[0],2))))
        bbox_ndc[:,2:4] = g_pos_screen[2:4]

        return conic, bboxsize_cam, bbox_ndc

    def get_color(self, dir) -> np.ndarray:
        """ Samples spherical harmonics to get color for given view direction """
        c0 = self.sh[0:3]   # f_dc_* from the ply file)
        color = SH_C0 * c0

        shdim = len(self.sh)

        if shdim > 3:
            # Add the first order spherical harmonics
            c1 = self.sh[3:6]
            c2 = self.sh[6:9]
            c3 = self.sh[9:12]
    
            x = dir[0]
            y = dir[1]
            z = dir[2]
            color = color - SH_C1 * y * c1 + SH_C1 * z * c2 - SH_C1 * x * c3
            
        if shdim > 12:
            c4 = self.sh[12:15]
            c5 = self.sh[15:18]
            c6 = self.sh[18:21]
            c7 = self.sh[21:24]
            c8 = self.sh[24:27]
    
            (xx, yy, zz) = (x * x, y * y, z * z)
            (xy, yz, xz) = (x * y, y * z, x * z)
            
            color = color +	SH_C2_0 * xy * c4 + \
                SH_C2_1 * yz * c5 + \
                SH_C2_2 * (2.0 * zz - xx - yy) * c6 + \
                SH_C2_3 * xz * c7 + \
                SH_C2_4 * (xx - yy) * c8

        if shdim > 27:
            c9 = self.sh[27:30]
            c10 = self.sh[30:33]
            c11 = self.sh[33:36]
            c12 = self.sh[36:39]
            c13 = self.sh[39:42]
            c14 = self.sh[42:45]
            c15 = self.sh[45:48]
    
            color = color + \
                SH_C3_0 * y * (3.0 * xx - yy) * c9 + \
                SH_C3_1 * xy * z * c10 + \
                SH_C3_2 * y * (4.0 * zz - xx - yy) * c11 + \
                SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * c12 + \
                SH_C3_4 * x * (4.0 * zz - xx - yy) * c13 + \
                SH_C3_5 * z * (xx - yy) * c14 + \
                SH_C3_6 * x * (xx - 3.0 * yy) * c15
        
        color += 0.5
        return np.clip(color, 0.0, 1.0)


# Iterate over the gaussians and create Gaussian objects
gaussian_objects = []
for (pos, scale, rot, opacity, sh) in zip(gaussians.xyz, gaussians.scale, gaussians.rot, gaussians.opacity, gaussians.sh):
    gau = Gaussian(pos, scale, rot, opacity, sh)
    gaussian_objects.append(gau)