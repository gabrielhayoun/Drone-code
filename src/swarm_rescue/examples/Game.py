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


if __name__ == '__main__':

    game = MyMap()
    game.build_map()

    engine = Engine(playground=game.playground, screen=True)


    while engine.game_on:

        engine.update_screen()
        engine.update_observations()

        actions = {game.my_drone: game.my_drone.control()}

        terminate = engine.step(actions)

        time.sleep(0.5)

        if terminate:
            engine.terminate()

    engine.terminate()
