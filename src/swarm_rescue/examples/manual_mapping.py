"""
This program can be launched directly.
To move the drone, you have to click on the map, then use the arrows on the keyboard
"""

import math
import os
import random
import sys
import time

# This line add, to sys.path, the path to parent path of this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spg_overlay.misc_data import MiscData
from spg_overlay.drone_abstract import DroneAbstract
from solutions.my_drone_explore_sauv import MyDroneExploreSauv
from spg_overlay.rescue_center import RescueCenter
from maps.walls_01 import add_walls, add_boxes
from maps.map_compet_01 import MyMapCompet01
from spg_overlay import fps_display

from simple_playgrounds.agent.controllers import Keyboard
from simple_playgrounds.playground import SingleRoom
from simple_playgrounds.engine import Engine

# to display the sensors
import cv2


class MyMap(MyMapCompet01):
    pass

class MyDrone(MyDroneExploreSauv):
    def __init__(self, controller=Keyboard(), **kwargs):
        super().__init__(controller=controller, **kwargs)

    def define_message(self):
        """
        Here, we don't need communication...
        """
        pass

    def control(self):
        """
        We only send a command to do nothing
        """
        command = {self.longitudinal_force: 0.0,
                   self.lateral_force: 0.0,
                   self.rotation_velocity: 0.0,
                   self.grasp: 0,
                   self.activate: 0}
        return command


my_playground = MyMap().playground

engine = Engine(playground=my_playground,
                time_limit=MyMap().time_step_limit, screen=True)

# RESCUE CENTER
#rescue_center = RescueCenter(size=[90, 170])
#my_playground.add_element(rescue_center, ((50, 660), 0))

#add_walls(my_playground)
#add_boxes(my_playground)

misc_data = MiscData(size_area=my_playground.size, number_drones=1)
my_drone = MyDrone(misc_data=misc_data, should_display_lidar=False)

start_area = (300, 660)
my_playground.add_agent(my_drone, (
        (random.uniform(start_area[0] - 200, start_area[0] + 200),
         random.uniform(start_area[1] - 70, start_area[1] + 70)),
        random.uniform(-math.pi, math.pi)))

print(my_drone.base_platform.position, my_drone.base_platform.angle)
start_real_time = time.time()
while engine.game_on:

    engine.update_screen()

    actions = {my_drone: my_drone.controller.generate_actions()}

    terminate = engine.step(actions)
    engine.update_observations(grasped_invisible=True)
    #MyMap().explored_map.update({my_drone})

    # my_drone.display() for the lidar only
    #BEGIN WIP
    my_drone.display_drone_map()
    my_drone.process_gameplan()
	#END WIP
 
engine.terminate()
cv2.destroyAllWindows()
