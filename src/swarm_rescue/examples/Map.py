
import os
import sys

# This line add, to sys.path, the path to parent path of this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from simple_playgrounds.common.position_utils import CoordinateSampler
from simple_playgrounds.playground import SingleRoom
from spg_overlay.rescue_center import RescueCenter
from spg_overlay.wounded_person import WoundedPerson
from MyDrone import MyDrone
from maps.walls_01 import add_walls


class MyMap:

    def __init__(self):

        # BUILD MAP
        self.size_area = (1112, 750)
        self.playground = SingleRoom(size=self.size_area, wall_type='light')
        add_walls(self.playground)

        # RESCUE CENTER
        rescue_center = RescueCenter(size=[90, 170])
        self.playground.add_element(rescue_center, ((50, 660), 0))

        # WOUNDED PERSONS
        self.number_wounded_persons = 1
        center_area = (self.size_area[0] * 3 / 4, self.size_area[1] * 3 / 4)
        area_all = CoordinateSampler(center=center_area, area_shape='rectangle',
                                     size=(self.size_area[0] / 2, self.size_area[1] / 2))
        for i in range(self.number_wounded_persons):
            wounded_person = WoundedPerson(graspable=True, rescue_center=rescue_center)
            try:
                self.playground.add_element(wounded_person,((460, 530),0), allow_overlapping=False)
                self.wounded_persons.append(wounded_person)
            except:
                print("Failed to place object 'wounded_person'")

        # DRONE
        self.my_drone = MyDrone()
        self.playground.add_agent(self.my_drone, ((300, 600), 0))