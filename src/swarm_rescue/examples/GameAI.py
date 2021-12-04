"""
This program can be launched directly.
To move the drone, you have to click on the map, then use the arrows on the keyboard
"""
import os
import time
import sys
from Map import MyMap
from MyDroneAI import MyDroneAI

from simple_playgrounds.engine import Engine

# This line add, to sys.path, the path to parent path of this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class GameAI:

    def __init__(self):
        self.my_map = MyMap()
        self.engine = Engine(playground=self.my_map.playground, screen=True)
        self.reset()

    def reset(self):
        self.my_map = MyMap()
        self.engine = Engine(playground=self.my_map.playground, screen=True)
        self.iteration = 0;


def train():

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    drone = MyDroneAI()
    game = GameAI()

    while game.engine.game_on:

        game.engine.update_screen()
        game.engine.update_observations()

        # Get old state
        state_old = drone.get_state(game)

        # Get move
        final_move = drone.get_action(state_old)

        # Perform the move and get new state
        # Implémenter reward, done, score
        reward, done, score = 0,0,0
        actions = {game.my_map.my_drone: game.my_map.my_drone.control()}
        state_new = drone.get_state(game)

        # Train short memory
        drone.train_short_memory(state_old,final_move,reward,state_new,done)

        #Remember
        drone.remember(state_old,final_move,reward,state_new,done)

        if done :

            # Train long memory, plot the result
            game.reset()
            drone.nb_games += 1
            drone.train_long_memory()

            if score > record:
                record = total_score

        print('Game', drone.nb_games, 'Score', score, 'Record : ', record)

if __name__ == '__main__':

    game = GameAI()

    while game.engine.game_on:

        game.engine.update_screen()
        game.engine.update_observations()

        actions = {game.my_map.my_drone: game.my_map.my_drone.control()}

        terminate = game.engine.step(actions)

        time.sleep(0.002)

        if terminate:
            game.engine.terminate()

    game.engine.terminate()
