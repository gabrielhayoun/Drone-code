import math
import numpy as np

from spg_overlay.wounded_person import WoundedPerson
from simple_playgrounds.element.elements.basic import Wall

K_MEANS_SAMPLE_SIZE = 100
K_MAX = 5  # k_max pour l'algo kmeans, temporaire
EXPLO_GOALS_UPDATE = 10  # temporaire


class ExplorationGameplanNP:
    def __init__(self, **kwargs):
        self.map_explored = np.zeros((1500, 1500))  # ATTENTION limite haute
        self._exploration_goals = np.zeros((0, 2))
        self._exploration_goals_update_count = 0

    def update_map(self, drone_position_measured, drone_angle_measured, lidar_values, semantic_detections):

        LIDAR_SAMPLING = 1
        lidar_angles = np.linspace(-np.pi / 2, np.pi / 2,
                                   len(lidar_values)//LIDAR_SAMPLING)

        # ATTENTION portée lidar à 300px mais pas d'attribut public possédant ce param
        MAX_RANGE_LIDAR = 300
        MAX_RANGE_SEMANTIC = 200
        RESOLUTION_SEMANTIC = 2*np.pi/36/4 # from drone_sensor.py 4 rays in 36 cones
        
        
        #FILTRAGE des données
         #TODO: lissage du LIDAR, filtrage données de position et d'odométrie => ou bien filtrage du LIDAR et direct des positions d'objets ?
        #en dessous supposées filtrées termes à remplacer quand le filtre sera mis en place
        
        
        x_pos = drone_position_measured[0]
        y_pos = drone_position_measured[1]
        
        # TODO: la map est tournée de +90° pour une raison inconnue (pb d'axes ?)
        for i in range(len(lidar_values)):
            angle_lidar = lidar_angles[i]
            angle_pt = drone_angle_measured + angle_lidar 
            distance_pt = lidar_values[i]
            detection = semantic_detections
            x_pt = round(x_pos + distance_pt * math.cos(angle_pt))
            y_pt = round(y_pos + distance_pt * math.sin(angle_pt))
            if distance_pt < MAX_RANGE_LIDAR-10: # -10 pour éviter le bruit du lidar, à réduire quand on aura le filtre
                # mise a jour des points intermédiaires sur le rayon
                for i in range (round(distance_pt)-10):
                    x_maj = round(x_pos + i * math.cos(angle_pt))
                    y_maj = round(y_pos + i * math.sin(angle_pt))
                    self.map_explored[x_maj, y_maj] = 4 # Void
                    
                if distance_pt < MAX_RANGE_SEMANTIC:
                    for detection in semantic_detections:
                        if angle_lidar - RESOLUTION_SEMANTIC/2 < detection.angle <= angle_lidar + RESOLUTION_SEMANTIC/2:
                            if isinstance(detection.entity_type, Wall):
                                self.map_explored[x_pt, y_pt] = 1
                            elif isinstance(detection.entity_type, WoundedPerson):
                                self.map_explored[x_pt, y_pt] = 3
                else : 
                    self.map_explored[x_pt,y_pt] = 2 # Unkown obstacle
        # a 90 degree rotation is needed for "correct" display np.rot90(self.map_explored, k=1)
        return self.map_explored
