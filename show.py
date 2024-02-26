import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.colors as mcolors
from matplotlib.scale import FuncScale

# Define the custom color function
def custom_color(value):
    hue = (value ** .25) * 0.4
    rgb = mcolors.hsv_to_rgb([hue, 1.0, 0.8])
    hex_color = mcolors.to_hex(rgb)
    return hex_color

# Define the exponential scaling function
def exp_scale():
    def forward(x):
        return x ** .25  # Adjust the exponent as needed
    def inverse(x):
        return x ** 4  # Inverse function of the forward function
    return forward, inverse

# Define the size of the grid
grid_size = 5

# Create an array of values ranging from 0 to 1
values = np.linspace(0, 1, grid_size * grid_size)

# Reshape the array into a grid
values_grid = values.reshape(grid_size, grid_size)

fig = plt.figure()
ax = fig.add_subplot(111)
rectangles = []
for i in range(10000):
    rectangles.append(matplotlib.patches.Rectangle((i/10000,0), .1, .1, color=custom_color(i/10000)))

for rectangle in rectangles:
    ax.add_patch(rectangle)

# Set the aspect ratio to equal to make squares square
ax.set_aspect('equal')

ax.set_xscale('function', functions=exp_scale())

# Show the plot
plt.show()
