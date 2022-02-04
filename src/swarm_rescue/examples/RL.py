from __future__ import print_function
import os, sys, time, datetime, json, random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
import matplotlib.pyplot as plt
%matplotlib inline


visited_mark = 0.8 * 255   # Cells visited by the rat will be painted by gray 0.8
pos_mark = 0.5 * 255      # The current drone cell will be painteg by gray 0.5
LEFT_05 = 0
UP_05 = 1
RIGHT_05 = 2
DOWN_05 = 3
LEFT = 4
UP = 5
RIGHT = 6
DOWN = 7
ROTATE_D = 8
ROTATE_I = 9
STAND = 10


# Actions dictionary
actions_dict = {
    LEFT_05: 'half left',
    UP_05: 'half up',
    RIGHT_05: 'half right',
    DOWN_05: 'half down',
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
    ROTATE_D: 'rotate direct',
    ROTATE_I: 'rotate indirect',
    STAND: 'stand'
}

actions = [0,1,2,3,4,5,6,7,8,9,10]

num_actions = len(actions_dict)

# Exploration factor
epsilon = 0.1


# map is a 2d Numpy array of int between 0 to 255
# 255 corresponds to a wall, and 0 a free cell
# pos = (row, col) initial drone position (defaults to (300,660))

class Qmap(object):
    def __init__(self, map, pos=(300, 660), wounded_pos=(40,40)):
        self._map = np.array(map)
        nrows, ncols = self._map.shape
        self.target = wounded_pos # target cell
        self.free_cells = [(r, c) for r in range(nrows) for c in range(ncols) if self._map[r, c] == 0]
        self.free_cells.remove(self.target)
        if self._map[self.target] == 255:
            raise Exception("Invalid map: target cell cannot be blocked!")
        if not pos in self.free_cells:
            raise Exception("Invalid Drone Location: must sit on a free cell")
        self.reset(pos)

    def reset(self, pos):
        self.pos = pos
        self.map = np.copy(self._map)
        nrows, ncols = self.map.shape
        row, col = pos
        self.map[row, col] = pos_mark
        self.state = (row, col, 'start')
        self.min_reward = -0.5 * self.map.size
        self.total_reward = 0
        self.visited = set()

    def update_state(self, action, drone):
        nrows, ncols = self.map.shape
        nrow, ncol, nmode = pos_row, pos_col, mode = self.state

        if self.map[pos_row, pos_col] > 0.0:
            self.visited.add((pos_row, pos_col))  # mark visited cell

        if drone.process_touch_sensor:
            nmode = 'blocked'
        else:
            nmode = 'valid'
            if action == LEFT:
                pos = drone.controle_RL('HALF LEFT')
            elif action == UP:
                pos = drone.controle_RL('HALF UP')
            if action == RIGHT:
                pos = drone.controle_RL('HALF RIGHT')
            elif action == DOWN:
                pos = drone.controle_RL('HALF DOWN')
            if action == LEFT:
                pos = drone.controle_RL('LEFT')
            elif action == UP:
                pos = drone.controle_RL('UP')
            if action == RIGHT:
                pos = drone.controle_RL('RIGHT')
            elif action == DOWN:
                pos = drone.controle_RL('DOWN')
            if action == ROTATE_D:
                pos = drone.controle_RL('ROTATE_D')
            elif action == ROTATE_I:
                pos = drone.controle_RL('ROTATE_I')
            if action == STAND:
                pos = drone.controle_RL('STAND')
        # new state
        self.state = (pos[0], pos[1], nmode)

    def get_reward(self):
        pos_row, pos_col, mode = self.state
        nrows, ncols = self.map.shape
        if pos_row == self.target[0] and pos_col == self.target[1]:
            return 1.0
        if mode == 'blocked':
            return -0.75
        if (pos_row, pos_col) in self.visited:
            return -0.25
        if mode == 'valid':
            return -0.04

    def act(self, drone , action):
        self.update_state(action, drone)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        return envstate, reward, status

    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    def draw_env(self):
        canvas = np.copy(self.map)
        nrows, ncols = self.map.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r, c] > 255:
                    canvas[r, c] = 0
        # draw the drone
        row, col, valid = self.state
        canvas[row, col] = pos_mark
        return canvas

    def game_status(self, drone):
        if self.total_reward < self.min_reward:
            return 'lose'
        pos_row, pos_col, mode = self.state
        nrows, ncols = self.map.shape
        if self.visible(self.target, drone):
            return 'win'
        return 'not_over'

    def visible (self, drone):
        return drone.is_visible(self.target)





def show(qmap):
    plt.grid('on')
    nrows, ncols = qmap.map.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmap.map)
    for row,col in qmap.visited:
        canvas[row,col] = 0.6
    pos_row, pos_col, _ = qmap.state
    canvas[pos_row, pos_col] = 0.3   # rat cell
    canvas[40, 40] = 0.9 # cheese cell
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    return img


map = [
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.]
]

qmap = Qmap(map)
canvas, reward, game_over = qmap.act(DOWN)
print("reward=", reward)
show(qmap)


qmap.act(DOWN)  # move down
qmap.act(RIGHT)  # move right
qmap.act(RIGHT)  # move right
qmap.act(RIGHT)  # move right
qmap.act(UP)  # move up
show(qmap)


def play_game(model, qmap, pos):
    qmap.reset(pos)
    envstate = qmap.observe()
    while True:
        prev_envstate = envstate
        # get next action
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])

        # apply action, get rewards and new state
        envstate, reward, game_status = qmap.act(action)
        if game_status == 'win':
            return True
        elif game_status == 'lose':
            return False



class Experience(object):
    def __init__(self, model, max_memory=100, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]

    def remember(self, episode):
        # episode = [envstate, action, reward, envstate_next, game_over]
        # memory[i] = episode
        # envstate == flattened 1d map cells info, including rat cell (see method: observe)
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, envstate):
        return self.model.predict(envstate)[0]

    def get_data(self, data_size=10):
        env_size = self.memory[0][0].shape[1]   # envstate 1d size (1st element of episode)
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            inputs[i] = envstate
            # There should be no target values for actions not taken.
            targets[i] = self.predict(envstate)
            # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
            Q_sa = np.max(self.predict(envstate_next))
            if game_over:
                targets[i, action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets


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

    for epoch in range(n_epoch):
        loss = 0.0
        rat_cell = random.choice(qmap.free_cells)
        qmap.reset(rat_cell)
        game_over = False

        # get initial envstate (1d flattened canvas)
        envstate = qmap.observe()

        n_episodes = 0
        while not game_over:
            valid_actions = qmap.valid_actions()
            if not valid_actions: break
            prev_envstate = envstate
            # Get next action
            if np.random.rand() < epsilon:
                action = random.choice(valid_actions)
            else:
                action = np.argmax(experience.predict(prev_envstate))

            # Apply action, get reward and new envstate
            envstate, reward, game_status = qmap.act(action)
            if game_status == 'win':
                win_history.append(1)
                game_over = True
            elif game_status == 'lose':
                win_history.append(0)
                game_over = True
            else:
                game_over = False

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


# This is a small utility for printing readable time strings:
def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)

def build_model(map, lr=0.001):
    model = Sequential()
    model.add(Dense(map.size, input_shape=(map.size,)))
    model.add(PReLU())
    model.add(Dense(map.size))
    model.add(PReLU())
    model.add(Dense(num_actions))
    model.compile(optimizer='adam', loss='mse')
    return model

map =  np.array([
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  0.,  1.,  0.],
    [ 0.,  0.,  0.,  1.,  1.,  1.,  0.],
    [ 1.,  1.,  1.,  1.,  0.,  0.,  1.],
    [ 1.,  0.,  0.,  0.,  1.,  1.,  1.],
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.]
])

qmap = Qmap(map)
show(qmap)

model = build_model(map)
qtrain(model, map, epochs=1000, max_memory=8*map.size, data_size=32)

# Fancy Notebook CSS Style
# from nbstyle import *
# HTML('<style>%s</style>' % (fancy(),))