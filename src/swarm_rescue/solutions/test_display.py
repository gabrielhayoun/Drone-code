import numpy as np
import matplotlib.pyplot as plt


map_explored = np.random.randint(5, size=(1500, 1000))
map_rgb = np.zeros((1500, 1000, 3))

for i in range(map_explored.shape[0]):
    for j in range(map_explored.shape[1]):
        if map_explored[i, j] == 0:  # Unexplored
            map_rgb[i, j] = [0, 0, 0]
        elif map_explored[i, j] == 2:  # Unknown obstacle
            map_rgb[i, j] = [180, 100, 220]
        elif map_explored[i, j] == 1:  # Wall
            map_rgb[i, j] = [255, 255, 255]
        elif map_explored[i, j] == 4:  # Void
            map_rgb[i, j] = [40, 40, 40]
        elif map_explored[i, j] == 3:  # Wounded
            map_rgb[i, j] = [200, 60, 60]
                
plt.imshow(map_rgb)
plt.show()
