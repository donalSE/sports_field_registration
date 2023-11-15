from data_management.homo_utils import get_dst_points
import cv2
import numpy as np


class HomographyTransformer:
    def __init__(self, pixel_points, real_world_points):
        """
        Initializes the HomographyTransformer with corresponding points.

        :param pixel_points: List of points in pixel coordinates.
        :param real_world_points: List of points in real-world coordinates.
        """
        self.pixel_points = np.array(pixel_points, dtype='float32')
        self.real_world_points = np.array(real_world_points, dtype='float32')
        self.homography_matrix = None
        self.calculate_homography_matrix()

    def calculate_homography_matrix(self):
        """
        Calculates the homography matrix using the provided points.
        """
        if len(self.pixel_points) != len(self.real_world_points):
            raise ValueError("The number of pixel points must match the number of real-world points.")

        if len(self.pixel_points) < 4:
            raise ValueError("At least four points are required to compute the homography matrix.")

        # Compute the homography matrix
        self.homography_matrix, _ = cv2.findHomography(self.pixel_points, self.real_world_points)

    def transform_point(self, pixel_point):
        """
        Transforms a point from pixel coordinates to real-world coordinates using the homography matrix.

        :param pixel_point: The pixel point to transform.
        :return: Transformed point in real-world coordinates.
        """
        if self.homography_matrix is None:
            raise ValueError("Homography matrix has not been computed.")

        pixel_point_homogeneous = np.append(np.array(pixel_point, dtype='float32'), 1)
        real_world_point = self.homography_matrix @ pixel_point_homogeneous
        real_world_point /= real_world_point[-1]  # Convert to non-homogeneous coordinates
        return real_world_point[:-1]

    def get_homography_matrix(self):
        """
        Returns the computed homography matrix.
        """
        return self.homography_matrix
