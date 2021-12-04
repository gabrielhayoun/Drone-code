# This line add, to sys.path, the path to parent path of this file
import os
import random
import math
import keras
import numpy as np
from collections import deque
from typing import Optional
from enum import Enum
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spg_overlay.drone_abstract import DroneAbstract
from spg_overlay.utils import sign, normalize_angle
from spg_overlay.rescue_center import RescueCenter
from spg_overlay.wounded_person import WoundedPerson
from GameAI import GameAI


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class MyDroneAI(DroneAbstract):
    class Activity(Enum):
        """
        All the states of the drone as a state machine
        """
        SEARCHING_WOUNDED = 1
        GRASPING_WOUNDED = 2
        SEARCHING_RESCUE_CENTER = 3
        DROPPING_AT_RESCUE_CENTER = 4

    def __init__(self,
                 identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier,
                         should_display_lidar=False,
                         **kwargs)
        # The state is initialize to searching wounded person
        self.state = self.Activity.SEARCHING_WOUNDED

        # Those values are used by the random control function
        self.counterStraight = 0
        self.angleStopTurning = 0
        self.nb_games = 0
        self.epsilon = 0 # Randomness
        self.gamma = 0 # Discount rate
        self.memory = deque(maxlen = MAX_MEMORY) #popleft()
        self.isTurning = False
        self.model = None
        self.trainer = None

    def get_state(self):

        # TODO

        pass

    def remember(self, state, action, reward, next_state,done):
        self.memory.append((state, action, reward, next_state,done)) #popleft() is max memory is reached
        pass

    def train_long_memory(self):

        if len(self.memory) > BATCH_SIZE :
            mini_sample = random.sample(self.memory, BATCH_SIZE) # List of tuples

        else :
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

        pass

    def train_short_memory(self, state, action, reward, next_state, done):

        self.trainer.train_step(state, action, reward, next_state, done)
        pass

    def get_action(self, state):
        self.epsilon = 80-self.nb_games

        ## TODO 28' of the video 3
        # final_move =
        pass

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
            command = self.control_random()
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

    def process_touch_sensor(self):
        """
        Returns True if the drone hits an obstacle
        """
        touched = False
        detection = max(self.touch().sensor_values)

        if detection > 0.5:
            touched = True

        return touched

    def control_random(self):
        """
        The Drone will move forward and turn for a random angle when an obstacle is hit
        """
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
        print(detection_semantic[0].distance)
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
