import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# load maze image
img = Image.open("MAZE_0.png").convert("L")
maze = np.array(img)

print("Maze size:", maze.shape)

plt.imshow(maze, cmap="gray")
plt.title("Maze")
plt.show()