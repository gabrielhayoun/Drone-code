"""
This program can be launched directly.
To move the drone, you have to click on the map, then use the arrows on the keyboard
"""
import os
import time
import cv2
import sys
from Map import MyMap
from simple_playgrounds.engine import Engine
from Astar import Astar

# This line add, to sys.path, the path to parent path of this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == '__main__':

    game = MyMap()
    game.build_map()

    # Position du drone et du blessé
    goal = (40,40)
    start = (300, 660)

    engine = Engine(playground=game.playground, screen=True)

    ## Acquisition de la map en np.array
    map = game.explored_map._map_playground

    ## Run de l'algo A*
    a = Astar(map,goal,start)
    path = a.aStar()

    ## On trace le chemin donné par l'algo
    for cell in path :

        y,x = path[cell]
        cell = x,y
        map[cell] = 120

    ## Acquisition des positions des blessés
    wounded_pos = game.my_drone.wounded_pos

    ## On trace la position des blessés
    for x in wounded_pos :
        map[x[1],x[0]] = 200

    ## On plot la map
    cv2.imshow("Map", map)

    game.my_drone.get_map(map)


    while engine.game_on:

        engine.update_screen()
        engine.update_observations()

        actions = {game.my_drone: game.my_drone.control()}

        terminate = engine.step(actions)

        time.sleep(0.05)

        if terminate:
            engine.terminate()

    engine.terminate()
