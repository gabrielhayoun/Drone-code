"""
This program can be launched directly.
To move the drone, you have to click on the map, then use the arrows on the keyboard
"""
import os
import pickle
import numpy as np
import time
import cv2
import sys
from Map import MyMap
from simple_playgrounds.engine import Engine
from Astar import Astar
from MyDrone import MyDrone

# This line add, to sys.path, the path to parent path of this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == '__main__':

    ## Construction de la Map
    game = MyMap()
    game.build_map()

    engine = Engine(playground=game.playground, screen=True)

    # BUILD DRONES
    drones = [MyDrone((40,40))
              for i in
              range(9)]

    game.set_drones(drones)

    ## Acquisition de la map en np.array
    map = game.explored_map._map_playground

    ## Run de l'algo A*
    #a = Astar(map)
    #grid = a.grid

    #path = a.aStar(start,goal,grid)

    path_list = []

    ## Acquisition des positions des blessés
    #wounded_pos = game.my_drone.wounded_pos

    with open("grid.txt", "rb") as fp:  # Unpickling

        grid = pickle.load(fp)

    wounded_persons_pos = [(40, 40), (90, 40), (330, 40),
                           (35, 300), (495, 50), (245, 275),
                           (385, 520), (460, 530), (1080, 50)]

    a = Astar(map,grid)
    #path = a.aStar((300,660),(40,40),grid)

    for i in range (len(wounded_persons_pos)):

        path = a.aStar((300,660),wounded_persons_pos[i],grid)
        l = []
        for cell in path:

            y, x = path[cell]
            cell = x, y
            l.append((y, x))
        l = l[::-1]
        path_list.append(l)


    ## On plot la map
    #cv2.imshow("Map", map)

    #game.my_drone.get_map(map)



    for i in range (9) :

        path = game.drones[i].discret_path(path_list[i])
        game.drones[i].path = path

        l = np.ones(len(path))
        game.drones[i].path_followed = l


    while engine.game_on:

        engine.update_screen()
        engine.update_observations()


        #actions = {game.my_drone: game.my_drone.control()}

        # COMPUTE ACTIONS
        actions = {}
        for i in range(0, 9):
            actions[game.drones[i]] = game.drones[i].control()

        terminate = engine.step(actions)

        time.sleep(0.00001)

        if terminate:
            engine.terminate()

    engine.terminate()
