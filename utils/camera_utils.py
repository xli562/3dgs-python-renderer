import numpy as np
import glm

class Camera:
    """ Represents a camera in a 3D environment, which can 
    compute transformations from world coords
    to camera coords / projection screens.

    Attributes:
        h (int): Height of the camera's viewport.
        w (int): Width of the camera's viewport.
        position (np.ndarray): Cartesian coords of the camera.
        target (np.ndarray): The target point the camera is looking at.
        znear (float): Distance to the near clipping plane.
        zfar (float): Distance to the far clipping plane.
        fovy (float): Vertical field of view in radians.
        up (np.ndarray): The up direction vector for the camera.
        yaw (float): Yaw angle of camera.
        pitch (float): Pitch angle of camera.
    """

    def __init__(self, h, w, position=(0.0, 0.0, 3.0), target=(0.0, 0.0, 0.0)):
        """ Initialise the camera with specific 
        viewport dimensions and position attributes.
        """
        self.znear = 0.01
        self.zfar = 100
        self.h = h
        self.w = w
        self.fovy = np.pi / 2.0
        self.position = np.array(position)
        self.target = np.array(target)
        self.up = np.array([0.0, -1.0, 0.0])
        self.yaw = -np.pi / 2
        self.pitch = 0

        self.is_pose_dirty = True
        self.is_intrin_dirty = True

        self.last_x = 640
        self.last_y = 360
        self.first_mouse = True

        self.is_leftmouse_pressed = False
        self.is_rightmouse_pressed = False

        self.rot_sensitivity = 0.02
        self.trans_sensitivity = 0.01
        self.zoom_sensitivity = 0.08
        self.roll_sensitivity = 0.03

    def _global_rot_mat(self):
        """ Calculate the global rotation matrix of the camera
        based on the its up-direction vector.

        Returns:
            np.ndarray: The rotation matrix.
        """
        x = np.array([1, 0, 0])
        z = np.cross(x, self.up)
        z = z / np.linalg.norm(z)
        x = np.cross(self.up, z)
        return np.stack([x, self.up, z], axis=-1)

    def get_view_matrix(self):
        """ Compute the view matrix using the camera's position, 
        target, and up vector.

        Returns:
            np.ndarray: The view matrix.
        """
        return np.array(glm.lookAt(self.position, self.target, self.up))

    def get_projection_matrix(self):
        """ Compute the projection matrix based on the camera's 
        field of view and viewport dimensions.

        Returns:
            np.ndarray: The projection matrix.
        """
        project_mat = glm.perspective(
            self.fovy,
            self.w / self.h,
            self.znear,
            self.zfar
        )
        return np.array(project_mat).astype(np.float32)

    def get_htanfovxy_focal(self):
        """ Calculate the horizontal and vertical tangents of 
        the field of view and the focal length.

        Returns:
            list: [horizontal tangent, vertical tangent, focal length]
        """
        htany = np.tan(self.fovy / 2)
        htanx = htany / self.h * self.w
        focal = self.h / (2 * htany)
        return [htanx, htany, focal]

    def get_focal(self):
        """ Calculate the focal length based on the 
        camera's field of view and viewport height.

        Returns:
            float: The focal length.
        """

    def get_htanfovxy(self):
        """ Calculate the horizontal and vertical 
        tangents of the field of view.

        Returns:
            list: [horizontal tangent, vertical tangent]
        """
        htany = np.tan(self.fovy / 2)
        htanx = htany / self.h * self.w
        return [htanx, htany]

    def world_to_cam(self, points):
        """ Transform points from world coordinates to camera coordinates.

        Parameters:
            points (np.ndarray): Points in world coordinates.

        Returns:
            np.ndarray: Points in camera coordinates.
        """
        view_mat = self.get_view_matrix()

        # if points is 3xN, add a fourth row of ones
        if points.shape[0] == 3:
            # If there is only one point, just add a fourth vallue of 1
            if len(points.shape) == 1:
                points = np.append(points, 1.0)
            else:
                points = np.concatenate([points, np.ones((1, points.shape[1]))], axis=0)

        return np.matmul(view_mat, points)

    def cam_to_world(self, points):
        """ Transform points from camera coordinates to world coordinates.

        Parameters:
            points (np.ndarray): Points in camera coordinates.

        Returns:
            np.ndarray: Points in world coordinates.
        """
        view_mat = self.get_view_matrix()
        return np.matmul(view_mat.T, points)

    def cam_to_ndc(self, points):
        """ Transform points from camera coordinates to
        normalized device coordinates.

        Parameters:
            points (np.ndarray): Points in camera coordinates.

        Returns:
            np.ndarray: Points in normalized device coordinates.
        """
        proj_mat = self.get_projection_matrix()
        points_ndc = proj_mat @ points
        if len(points_ndc.shape) == 1:
            points_ndc = points_ndc / points_ndc[3]
        else:
            points_ndc = points_ndc / points_ndc[3, :]
        return points_ndc

    def ndc_to_cam(self, points):
        """ Transform points from normalized 
        device coordinates to camera coordinates.

        Parameters:
            points (np.ndarray): Points in normalized device coordinates.

        Returns:
            np.ndarray: Points in camera coordinates.
        """
        proj_mat = self.get_projection_matrix()
        return np.linalg.inv(proj_mat) @ points

    def ndc_to_pixel(self, points_ndc, screen_width=None, screen_height=None):
        """ Convert points from normalized device coordinates
        to pixel coordinates.

        Parameters:
            points_ndc (np.ndarray): Points in normalized device coordinates.
            screen_width (int, optional): The width of the screen. 
                                        Defaults to camera width.
            screen_height (int, optional): The height of the screen. 
                                        Defaults to camera height.

        Returns:
            np.ndarray: Points in pixel coordinates.
        """
        # Use camera plane size if screen size not specified
        if screen_width is None:
            screen_width = self.w
        if screen_height is None:
            screen_height = self.h

        width_half = screen_width / 2
        height_half = screen_height / 2

        if len(points_ndc.shape) == 1:
            # It is a single point, so just return the pixel coordinates
            return np.array([(points_ndc[0] + 1) * width_half, (1.0 - points_ndc[1]) * height_half])
        else:
            return np.array([(point[0] * width_half + width_half, - point[1] * height_half + height_half)
                for point in points_ndc])

    def update_resolution(self, height, width):
        """ Update the resolution of the camera's viewport.

        Parameters:
            height (int): New height of the viewport.
            width (int): New width of the viewport.
        """
        self.h = height
        self.w = width
        self.is_intrin_dirty = True
