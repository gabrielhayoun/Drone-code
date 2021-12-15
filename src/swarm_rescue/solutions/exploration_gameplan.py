
import numpy as np

from solutions.k_means import KMeans
from solutions.map import Map, PixelType
from spg_overlay.wounded_person import WoundedPerson
from simple_playgrounds.element.elements.basic import Wall

K_MEANS_SAMPLE_SIZE = 100
K_MAX = 5 #k_max pour l'algo kmeans, temporaire
EXPLO_GOALS_UPDATE = 10 #temporaire

class ExplorationGameplan:
    def __init__(self, **kwargs):
        self._explored_map = Map(1500, 1000)  # ATTENTION limite haute
        self._exploration_goals = np.zeros((0,2))
        self._exploration_goals_update_count = 0

    def update_map(self, drone_position, drone_angle, lidar_values, semantic_detections):
        
        LIDAR_SAMPLING = 1
        # Lidar scan entre -90° et 90° avec une résolution de 1/3 des points traités
        lidar_angles = np.linspace(-np.pi / 2, np.pi / 2, len(lidar_values)//LIDAR_SAMPLING)

        # ATTENTION portée lidar à 300px mais pas d'attribut public possédant ce param
        MAX_RANGE_LIDAR = 300
        MAX_RANGE_SEMANTIC = 200
        RESOLUTION_SEMANTIC = 2*np.pi/36/4

        for k in range(len(lidar_values)//LIDAR_SAMPLING):

            lidar_orientation = lidar_angles[k]

            if lidar_values[k] < MAX_RANGE_LIDAR:  # there is an obstacle

                range_obstacle = lidar_values[k]

                if range_obstacle < MAX_RANGE_SEMANTIC:

                    for detection in semantic_detections:

                        if lidar_orientation - RESOLUTION_SEMANTIC/2 < detection.angle < lidar_orientation + RESOLUTION_SEMANTIC/2:

                            if isinstance(detection.entity_type, Wall):
                                self._explored_map.draw_radial_with_tip(drone_position,
                                                                        range_obstacle,
                                                                        drone_angle + lidar_orientation,
                                                                        PixelType.Empty,
                                                                        PixelType.Wall)

                            if isinstance(detection.entity_type, WoundedPerson):
                                self._explored_map.draw_radial_with_tip(drone_position,
                                                                        range_obstacle,
                                                                        drone_angle + lidar_orientation,
                                                                        PixelType.Empty,
                                                                        PixelType.Wounded)


                else:
                    self._explored_map.draw_radial_with_tip(drone_position,
                                                            range_obstacle,
                                                            drone_angle + lidar_orientation,
                                                            PixelType.Empty,
                                                            PixelType.Unkown_obstacle)

            else:

                self._explored_map.draw_radial_with_tip(drone_position,
                                                        MAX_RANGE_LIDAR,
                                                        drone_angle + lidar_orientation,
                                                        PixelType.Empty,
                                                        PixelType.Exploration_limit)

    def get_exploration_goal(self, drone_position):

        if(self._exploration_goals_update_count >= EXPLO_GOALS_UPDATE):
            self.update_exploration_goals()
            self._exploration_goals_update_count = 0

        self._exploration_goals_update_count +=1

        if len(self._exploration_goals) > 0:
            goals_distances = np.linalg.norm(self._exploration_goals-drone_position, axis=-1)
            return self._exploration_goals[np.argmin(goals_distances)]

        else:
            return None

    def update_exploration_goals(self):

        if (len(self._explored_map.exploration_limits) != 0):
            k_means = KMeans(self._explored_map.exploration_limits, K_MEANS_SAMPLE_SIZE)
            self._exploration_goals = k_means.process(K_MAX)

        else:
            self._exploration_goals = np.array([])

    def display_map(self):

        self._explored_map.display()
