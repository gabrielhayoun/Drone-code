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
from MyDrone import MyDrone
from RL import *


# This line add, to sys.path, the path to parent path of this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == '__main__':



    ## Construction de la Map
    game = MyMap()
    game.build_map()

    engine = Engine(playground=game.playground, screen=True)

    ## Acquisition de la map en np.array
    map = game.explored_map._map_playground


    ## Acquisition des positions des blessés

    wounded_persons_pos = [(40, 40), (90, 40), (330, 40),
                           (35, 300), (495, 50), (245, 275),
                           (385, 520), (460, 530), (1080, 50)]


    ## On plot la map
    #cv2.imshow("Map", map)











    def qtrain(model, map, **opt):

        global epsilon
        n_epoch = opt.get('n_epoch', 15000)
        max_memory = opt.get('max_memory', 1000)
        data_size = opt.get('data_size', 50)
        weights_file = opt.get('weights_file', "")
        name = opt.get('name', 'model')
        start_time = datetime.datetime.now()

        # If you want to continue training from a previous model,
        # just supply the h5 file name to weights_file option
        if weights_file:
            print("loading weights from file: %s" % (weights_file,))
            model.load_weights(weights_file)

        # Construct environment/game from numpy array: map (see above)

        qmap = Qmap(map)

        # Initialize experience replay object
        experience = Experience(model, max_memory=max_memory)

        win_history = []  # history of win/lose game
        n_free_cells = len(qmap.free_cells)
        hsize = qmap.map.size // 2  # history window size
        win_rate = 0.0
        imctr = 1

        while engine.game_on:
            for epoch in range(n_epoch):
                loss = 0.0
                drone_cell = random.choice(qmap.free_cells)
                game.set_drone(drone_cell)
                qmap.reset(drone_cell)
                game_over = False

                # get initial envstate (1d flattened canvas)
                envstate = qmap.observe()

                n_episodes = 0

                while not game_over:
                    engine.update_screen()
                    engine.update_observations()
                    prev_envstate = envstate
                    # Get next action
                    if np.random.rand() < epsilon:
                        action = random.choice(actions)
                    else:
                        action = np.argmax(experience.predict(prev_envstate))

                    # Apply action, get reward and new envstate
                    envstate, reward, game_status = qmap.act(game.mydrone, action)
                    action_to_do = {game.mydrone: game.mydrone.get_action(action)}

                    # COMPUTE ACTIONS
                    terminate = engine.step(action_to_do)

                    if game_status == 'win':
                        win_history.append(1)
                        game_over = True
                    elif game_status == 'lose':
                        win_history.append(0)
                        game_over = True
                    else:
                        game_over = False

                    time.sleep(0.00001)

                    # Store episode (experience)
                    episode = [prev_envstate, action, reward, envstate, game_over]
                    experience.remember(episode)
                    n_episodes += 1

                    # Train neural network model
                    inputs, targets = experience.get_data(data_size=data_size)
                    h = model.fit(
                        inputs,
                        targets,
                        epochs=8,
                        batch_size=16,
                        verbose=0,
                    )
                    loss = model.evaluate(inputs, targets, verbose=0)

                if len(win_history) > hsize:
                    win_rate = sum(win_history[-hsize:]) / hsize

                dt = datetime.datetime.now() - start_time
                t = format_time(dt.total_seconds())
                template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
                print(template.format(epoch, n_epoch - 1, loss, n_episodes, sum(win_history), win_rate, t))
                # we simply check if training has exhausted all free cells and if in all
                # cases the agent won
                if win_rate > 0.9: epsilon = 0.05
                if sum(win_history[-hsize:]) == hsize:
                    print("Reached 100%% win rate at epoch: %d" % (epoch,))
                    break


                game.playground.remove_agent(game.mydrone)


            engine.terminate()



        # Save trained model weights and architecture, this will be used by the visualization code
        h5file = name + ".h5"
        json_file = name + ".json"
        model.save_weights(h5file, overwrite=True)
        with open(json_file, "w") as outfile:
            json.dump(model.to_json(), outfile)
        end_time = datetime.datetime.now()
        dt = datetime.datetime.now() - start_time
        seconds = dt.total_seconds()
        t = format_time(seconds)
        print('files: %s, %s' % (h5file, json_file))
        print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, data_size, t))
        return seconds