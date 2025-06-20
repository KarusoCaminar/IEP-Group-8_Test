#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is dedicated to creating high-quality visualizations for the
FDTD wave propagation simulation. It loads pre-computed simulation data
and generates an animation, which can be saved as a GIF or MP4 file.

It assumes that 'script3.py' has already been run to generate:
1. The grid file (e.g., 'Lshape_1.txt')
2. The solution data ('solution_data.npy')
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from help_functions import load_from_file, get_triangulation

# --- 1. Configuration and Data Loading ---
GRID_FILENAME = 'Lshape_1.txt'
SOLUTION_FILENAME = 'solution_data.npy'
ANIMATION_FILENAME = 'wave_propagation.gif' # Output filename

print("--- Visualization Script ---")

# Load the grid geometry
try:
    print(f"Loading grid from {GRID_FILENAME}...")
    pos, conn, bd = load_from_file(GRID_FILENAME)
except FileNotFoundError:
    print(f"ERROR: Grid file '{GRID_FILENAME}' not found. Please run the main scripts first.")
    exit()

# Load the pre-computed time-domain solution
try:
    print(f"Loading solution data from {SOLUTION_FILENAME}...")
    solution = np.load(SOLUTION_FILENAME)
except FileNotFoundError:
    print(f"ERROR: Solution file '{SOLUTION_FILENAME}' not found. Please run script3.py and save the solution first.")
    exit()

# --- 2. Animation Setup ---
# Setup the figure and axis for the animation
fig, ax = plt.subplots(figsize=(8, 7))
triangulation = get_triangulation(pos, conn, bd)

# Determine a consistent color scale for the entire animation
# We scale the max value slightly for better visual contrast
vmax = np.max(np.abs(solution)) * 0.7
vmin = -vmax

# Create the initial plot object (the first frame)
# 'shading="gouraud"' creates a smoother look than 'flat'
tripcolor = ax.tripcolor(
    pos[:, 0], pos[:, 1], triangles=triangulation,
    facecolors=solution[:, 0],
    cmap='seismic', vmin=vmin, vmax=vmax, shading='gouraud'
)

# Add a colorbar and set plot details
cbar = fig.colorbar(tripcolor, label='Wave Amplitude u(x,y,t)')
ax.set_xlabel('x-coordinate')
ax.set_ylabel('y-coordinate')
ax.set_aspect('equal', 'box')

# Dynamic title object that will be updated in the animation
time_text = ax.set_title('', fontsize=12)

# --- 3. Animation Logic ---
def update(frame):
    """
    This function is called for each frame of the animation.
    It updates the plot data with the solution at the new time step.
    """
    # Calculate current simulation time
    # This assumes Gt was created with T=20.0 and Nt=solution.shape[1] in the simulation script
    current_time = (frame / (solution.shape[1] - 1)) * 20.0
    
    # Update the plot data and title
    tripcolor.set_array(solution[:, frame])
    time_text.set_text(f'Wave Propagation in L-Shape Domain\nTime = {current_time:.2f} s')
    
    # Return the updated plot elements
    return tripcolor, time_text

# --- 4. Create and Save the Animation ---
# We use every 5th frame to keep the animation smooth but the file size reasonable
animation_frames = range(0, solution.shape[1], 5)

print("\nCreating animation (this may take a moment)...")
ani = animation.FuncAnimation(
    fig, update, frames=animation_frames, blit=True, interval=30
)

# Save the animation to a file (e.g., GIF)
# Requires 'pillow' library: pip install pillow
try:
    ani.save(ANIMATION_FILENAME, writer='pillow', fps=25)
    print(f"Animation successfully saved to '{ANIMATION_FILENAME}'")
except Exception as e:
    print(f"Error saving animation: {e}")
    print("Please ensure you have an appropriate writer installed (e.g., 'pip install pillow' for GIFs).")

# Display the animation
plt.show()
