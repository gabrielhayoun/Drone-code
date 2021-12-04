"""
This program can be launched directly.
To move the drone, you have to click on the map, then use the arrows on the keyboard
"""
import os
import time
import sys
from Map import MyMap

from simple_playgrounds.engine import Engine

# This line add, to sys.path, the path to parent path of this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class Game:

    def __init__(self):
        self.my_map = MyMap()
        self.engine = Engine(playground=self.my_map.playground, screen=True)
        self.reset()

    def reset(self):
        self.my_map = MyMap()
        self.engine = Engine(playground=self.my_map.playground, screen=True)
        self.iteration = 0;


if __name__ == '__main__':

    game = Game()

    while game.engine.game_on:

        game.engine.update_screen()
        game.engine.update_observations()

        actions = {game.my_map.my_drone: game.my_map.my_drone.control()}

        terminate = game.engine.step(actions)

        time.sleep(0.002)

        if terminate:
            game.engine.terminate()

    game.engine.terminate()
