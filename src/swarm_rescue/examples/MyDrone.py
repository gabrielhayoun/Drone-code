# This line add, to sys.path, the path to parent path of this file
import os
import random
import numpy as np
import math
from typing import Optional
from enum import Enum
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spg_overlay.drone_abstract import DroneAbstract
from spg_overlay.utils import sign, normalize_angle
from spg_overlay.rescue_center import RescueCenter
from spg_overlay.wounded_person import WoundedPerson

MAX_VEL_REEL = 0.286239


class MyDrone(DroneAbstract):
    class Activity(Enum):
        """
        All the states of the drone as a state machine
        """
        SEARCHING_WOUNDED = 1
        GRASPING_WOUNDED = 2
        SEARCHING_RESCUE_CENTER = 3
        DROPPING_AT_RESCUE_CENTER = 4

    def __init__(self, wounded_persons_pos,
                 identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier,
                         should_display_lidar=False,
                         **kwargs)
        # The state is initialize to searching wounded person
        self.state = self.Activity.SEARCHING_WOUNDED

        # Those values are used by the random control function
        self.counterStraight = 0
        self.angleStopTurning = 0
        self.isTurning = False
        self.path_followed = None
        self.path = None
        self.wounded_pos = wounded_persons_pos
        self.lock = False

    def define_message(self):
        """
        Here, we don't need communication...
        """
        pass

    def control(self):

        command = {self.longitudinal_force: 0.0,
                   self.lateral_force: 0.0,
                   self.rotation_velocity: 0.0,
                   self.grasp: 0,
                   self.activate: 0}

        found_wounded, found_rescue_center, command_semantic = self.process_semantic_sensor(self.semantic_cones())

        #############
        # TRANSITIONS OF THE STATE MACHINE
        #############

        if self.state is self.Activity.SEARCHING_WOUNDED and found_wounded:
            self.state = self.Activity.GRASPING_WOUNDED

        elif self.state is self.Activity.GRASPING_WOUNDED and self.grasp.grasped_element:
            self.state = self.state.SEARCHING_RESCUE_CENTER

        elif self.state is self.Activity.GRASPING_WOUNDED and not found_wounded:
            self.state = self.state.SEARCHING_WOUNDED

        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER and found_rescue_center:
            self.state = self.Activity.DROPPING_AT_RESCUE_CENTER

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and not self.grasp.grasped_element:
            self.state = self.Activity.SEARCHING_WOUNDED

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and not found_rescue_center:
            self.state = self.Activity.SEARCHING_RESCUE_CENTER

        ##########
        # COMMANDS FOR EACH STATE
        # Searching randomly, but when a rescue center or wounded person is detected, we use a special command
        ##########
        if self.state is self.Activity.SEARCHING_WOUNDED:
            command = self.control_deter()
            command[self.grasp] = 0

        elif self.state is self.Activity.GRASPING_WOUNDED:
            command = command_semantic
            command[self.grasp] = 1

        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER:
            command = self.control_random()
            command[self.grasp] = 1

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER:
            command = command_semantic
            command[self.grasp] = 1

        return command

    def process_lidar_sensor(self):

        "Return True is a wall is in front of the Drone"

        lidar = self.lidar().sensor_values
        close_wall = [False, False, False]  # Gauche, Face, Droite

        if lidar[0] < 80:
            close_wall[0] = True

        if lidar[90] < 80:
            close_wall[1] = True

        if lidar[179] < 80:
            close_wall[2] = True

        return close_wall

    def process_touch_sensor(self):
        """
        Returns True if the drone hits an obstacle
        """
        touched = False
        detection = max(self.touch().sensor_values)

        if detection > 0.5:
            touched = True

        return touched

    def discret_path(self, path):

        new_path = []

        for i in range(2, len(path)):

            if path[i][0] != path[i - 1][0] and path[i - 1][1] != path[i - 2][1]:

                new_path.append(path[i - 1])

            elif path[i][1] != path[i - 1][1] and path[i - 1][0] != path[i - 2][0]:

                new_path.append(path[i - 1])

        return new_path

    def discret_path_n(self, path,n):

        new_path = []

        for i in range (0,len(path),n) :

            new_path.append(path[i])

        return new_path

    def control_deter(self):

        # path = Astar(self.map)
        epsilon = 0.1

        command = {self.longitudinal_force: 0.0,
                   self.rotation_velocity: 0.0,
                   }
        command_null = {self.longitudinal_force: 0.0,
                   self.rotation_velocity: 0.0}

        rotation_velocity_max = 0.1

        wounded_pos = self.wounded_pos

        self.counterStraight += 1

        position = self.measured_position()
        path = self.path

        ind_obj = np.argmax(self.path_followed)
        obj = path[ind_obj]


        position = self.measured_position()
        theta = self.measured_angle()

        P = 20

        arg1 = position[1] - obj[1]
        arg2 = position[0] - obj[0]


        alpha = math.atan2(arg1, arg2) + math.pi
        if alpha == 2*math.pi : alpha = 0

        ### On change le système de coordonnée de la position angulaire pour aller de [0,2pi] vers [0,pi] , [-pi,0]

        #if theta >= 0 and theta < math.pi:

            #theta = theta

        #else:

            #theta = theta - 2 * math.pi

        print('############### AVANT MANOEUVRE ###############')
        print('x_0 = ' + str(position[0]) + ', y_0 = ' + str(position[1]))
        print('x_1 = ' + str(obj[0]) + ', y_1 =' + str(obj[1]))
        print('theta= ' + str(theta))
        print('alpha = ' + str(alpha))
        print(self.path_followed)
        print('###############################################')


        if  theta - alpha < math.pi  and abs(theta - alpha) > 0.3:
            command[self.rotation_velocity] =     - min(abs(theta-alpha), MAX_VEL_REEL)


        elif theta - alpha > math.pi  and abs(theta - alpha) > 0.3:

            command[self.rotation_velocity] =   min(abs(theta-alpha), MAX_VEL_REEL)


        else :

            command[self.longitudinal_force] = max(0.1,min((((position[0] - obj[0]) ** 2 + (position[1] - obj[1]) ** 2) ** 0.5)/100,0.4))
            #command[self.longitudinal_force] = 0.4


            if ((position[0] - obj[0]) ** 2 + (position[1] - obj[1]) ** 2) ** 0.5 < P:
                self.path_followed[ind_obj] = 0
                self.path_followed[ind_obj - 1] = - 1

        return command

        '''

        if arg1/arg2 < 0 :

            alpha = alpha + math.pi

        if alpha < 0 :

            alpha = alpha +2*math.pi

        if abs(attitude - alpha) > epsilon:
            command[self.rotation_velocity] = rotation_velocity_max

        else:
            command[self.longitudinal_force] = 1

        position = self.measured_position()

        if ((position[0] - obj[0]) ** 2 + (position[1] - obj[1]) ** 2) ** 0.5 < P:
            self.path_followed[ind_obj] = 0
            self.path_followed[ind_obj - 1] = -1
            
            '''


    def get_map(self, map):

        self.map = map

        position = [self.measured_position(self)]
        attitude = self.measured_angle(self)
        wounded_to_find = wounded_pos[0]
        distance_x = wounded_to_find[0] - position[0]
        distance_y = wounded_to_find[1] - position[1]

        "Vertical optimization"
        if distance_y<0:
            while attitude != 3*math.PI/4:
                command_turn

    def control_random(self):
        """
        The Drone will move forward and turn for a random angle when an obstacle is hit
        """

        position = self.measured_position()
        attitude = self.measured_angle()

        command_straight = {self.longitudinal_force: 1.0,
                            self.rotation_velocity: 0.0}

        command_turn = {self.longitudinal_force: 0.0,
                        self.rotation_velocity: 1.0}

        touched = self.process_touch_sensor()

        self.counterStraight += 1

        if touched and not self.isTurning and self.counterStraight > 10:
            self.isTurning = True
            self.angleStopTurning = random.uniform(-math.pi, math.pi)

        diff_angle = normalize_angle(self.angleStopTurning - self.measured_angle())
        if self.isTurning and abs(diff_angle) < 0.2:
            self.isTurning = False
            self.counterStraight = 0

        if self.isTurning:
            return command_turn
        else:
            return command_straight

    def process_semantic_sensor(self, the_semantic_sensor):
        """
        According his state in the state machine, the Drone will move towards a wound person or the rescue center
        """
        command = {self.longitudinal_force: 1.0,
                   self.lateral_force: 0.0,
                   self.rotation_velocity: 0.0}
        rotation_velocity_max = 0.6

        detection_semantic = the_semantic_sensor.sensor_values
        # print(detection_semantic[0].distance)
        best_angle = 1000

        found_wounded = False

        if (self.state is self.Activity.SEARCHING_WOUNDED
            or self.state is self.Activity.GRASPING_WOUNDED) \
                and detection_semantic:
            scores = []
            for detection in detection_semantic:

                # If the wounded person detected is held by nobody
                if detection.entity \
                        and isinstance(detection.entity, WoundedPerson) \
                        and len(detection.entity.held_by) == 0:
                    found_wounded = True
                    v = (detection.angle * detection.angle) + (detection.distance * detection.distance / 10 ** 5)
                    scores.append((v, detection.angle, detection.distance))

            # Select the best one among wounded persons detected
            best_score = 10000
            for score in scores:
                if score[0] < best_score:
                    best_score = score[0]
                    best_angle = score[1]

        found_rescue_center = False
        if (self.state is self.Activity.SEARCHING_RESCUE_CENTER
            or self.state is self.Activity.DROPPING_AT_RESCUE_CENTER) \
                and detection_semantic:
            for detection in detection_semantic:
                if isinstance(detection.entity, RescueCenter):
                    found_rescue_center = True
                    best_angle = detection.angle

        if found_rescue_center or found_wounded:
            a = sign(best_angle)
            # The robot will turn until best_angle is 0
            command[self.rotation_velocity] = a * rotation_velocity_max

        return found_wounded, found_rescue_center, command