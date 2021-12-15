import numpy as np
import enum
import cv2

class PixelType(enum.Enum):
    NA = [0., 0., 0.]
    Empty = [0.2, 0.2, 0.2]
    Wall = [0., 1., 0.]
    Unkown_obstacle = [1., 1., 1.]
    Wounded = [1., 0., 0.]
    Exploration_limit = [0, 0., 1.]

class Map:

    def __init__(self, dimension_x, dimension_y,  **kwargs):
        self._map = np.full((dimension_x, dimension_y, 3), PixelType.NA.value)
        self._dimension = self._map.shape
        self.exploration_limits = np.zeros((0,2))

    def _point_in_shape(self, point):
        """
        Check that a point respect the limits of the map
        """

        return 0 <= point[0] < self._dimension[0] and 0 <= point[1] < self._dimension[1]

    def draw_radial(self, center_point, lenght, angle, pixel_type):
        """
        Draw a radial starting from center_point with a certain lenght and angle with the color of
        pixel_type
        """

        if self._point_in_shape(center_point):

            for r in range(int(np.round(lenght))):

                delta_x = r * np.cos(angle)
                delta_y = r * np.sin(angle)

                point = [int(np.round(center_point[0] + delta_x)), int(np.round(center_point[1] + delta_y))]

                self.draw_point(point, pixel_type)

    def draw_radial_with_tip(self, center_point, lenght, angle, pixel_type_radial, pixel_type_tip):
        """
        Draw a radial starting from center_point with a certain length and angle with the color of
        pixel_type_radial and a different color for the tip of radial in an other color pixel_type_tip
        """
        if self._point_in_shape(center_point):

            #drawing the radial
            for r in range(int(np.round(lenght))-1):

                delta_x = r * np.cos(angle)
                delta_y = r * np.sin(angle)

                point = [int(np.round(center_point[0] + delta_x)), int(np.round(center_point[1] + delta_y))]

                self.draw_point(point, pixel_type_radial)

            #drawing the tip
            delta_x = lenght * np.cos(angle)
            delta_y = lenght * np.sin(angle)

            point = [int(np.round(center_point[0] + delta_x)), int(np.round(center_point[1] + delta_y))]

            self.draw_point(point, pixel_type_tip)


    def draw_point(self, point, pixel_type):

        if self._point_in_shape(point):

            if pixel_type == PixelType.Exploration_limit:

                if np.array_equal(self._map[point[0], point[1]], PixelType.NA.value):
                    self._map[point[0], point[1]] = pixel_type.value
                    self.exploration_limits = np.vstack((self.exploration_limits, point))

            else:
                if np.array_equal(self._map[point[0], point[1]], PixelType.Exploration_limit.value):
                    self.exploration_limits = np.delete(self.exploration_limits,
                                                        np.where(np.all(self.exploration_limits==point, axis=1)),
                                                        axis=0)


                self._map[point[0], point[1]] = pixel_type.value


    def display(self):

        cv2.imshow("map", self._map.transpose(1, 0, 2))

