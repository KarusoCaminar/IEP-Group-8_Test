#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 14:42:13 2020
@author: Ion Gabriel Ion, Dimitrios Loukrezis

This script solves Task 7.1:
1. Loads and visualizes a finite-difference grid from a file.
2. Plots a function defined on that grid.
"""

import numpy as np
import matplotlib.pyplot as plt
from help_functions import load_from_file, get_triangulation

# --- Configuration ---
# Define the grid file to be loaded. Other options include:
# 'Lshape_2.txt', 'Cshape_1.txt', or 'Cshape_2.txt'.
grid_filename = 'Lshape_1.txt'


# --- Task 7.1.1: Load and Display Grid Points ---
print(f"Loading grid from file: {grid_filename}")

# Load grid data: positions, connectivity, and boundary indices.
pos, conn, bd = load_from_file(grid_filename)

# Create a figure for the grid visualization.
plt.figure(figsize=(10, 8))

# Identify inner points by taking the set difference of all points and boundary points.
all_indices = np.arange(pos.shape[0])
inner_points_indices = np.setdiff1d(all_indices, bd)

# Plot inner points and boundary points with different colors for clarity.
plt.scatter(pos[inner_points_indices, 0], pos[inner_points_indices, 1], c='blue', label='Inner Points', s=15)
plt.scatter(pos[bd, 0], pos[bd, 1], c='red', label='Boundary Points', s=15)

# Add plot details for better presentation.
plt.title(f'Grid Visualization for "{grid_filename}"')
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.legend()
plt.grid(True)
plt.axis('equal')  # Ensure correct aspect ratio.
plt.show()


# --- Task 7.1.2: Represent a Function on the Grid ---

# Define the function f(x,y) = sin(2*pi*x) * sin(2*pi*y).
# This is applied vectorized to all coordinates for efficiency.
x_coords = pos[:, 0]
y_coords = pos[:, 1]
f_values = np.sin(2 * np.pi * x_coords) * np.sin(2 * np.pi * y_coords)

# A triangulation is required to plot a surface on the unstructured grid.
triangulation = get_triangulation(pos, conn, bd)

# Create a new figure for the function plot.
plt.figure(figsize=(10, 8))

# Use tripcolor to plot the function values on the triangulated grid.
# The color of each triangle corresponds to the function's value.
plt.tripcolor(pos[:, 0], pos[:, 1], triangles=triangulation, facecolors=f_values, cmap='viridis', shading='flat')

# Add a color bar to serve as a legend for the function values.
plt.colorbar(label='Function values f(x,y)')

# Add plot details.
plt.title('Plot of f(x,y) = sin(2πx)sin(2πy) on the Grid')
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.grid(True)
plt.axis('equal')
plt.show()
