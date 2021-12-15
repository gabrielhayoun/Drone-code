import random
import math
from copy import deepcopy
from typing import Optional

import numpy as np

from spg_overlay.drone_abstract import DroneAbstract
from spg_overlay.utils import normalize_angle, sign

from solutions.exploration_gameplan import ExplorationGameplan

#ROTATION VELOCITY = 1.0 => ANGLE STEP = 0.2862392814389153

class MyDroneExplore(DroneAbstract):
    def __init__(self, identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier,
                         should_display_lidar=False,
                         **kwargs)

        self._exploration_gameplan = ExplorationGameplan()

    def display_drone_map(self):
        self._exploration_gameplan.display_map()

    def control(self):
        command = {self.longitudinal_force: 0.0,
                   self.lateral_force: 0.0,
                   self.rotation_velocity: 0.0}

        exploration_goal = self.process_gameplan()

        return command

    def process_gameplan(self):

        self._exploration_gameplan.update_map(self.true_position(),
                                       self.true_angle(),
                                       self.lidar().get_sensor_values(),
                                       self.semantic_cones().sensor_values)

        return self._exploration_gameplan.get_exploration_goal(self.true_position())

    def _compute_direct_trajectory(self, goal_position):

        vector_to_goal = goal_position - self.true_position()

        