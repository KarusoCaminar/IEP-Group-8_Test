
==================================
README: Python Wave Simulation
==================================

HOW TO RUN
----------

1. Install required libraries:
   pip install numpy scipy matplotlib pillow

2. Run scripts in the following order:

   1. python script1.py
      - Purpose: Visualizes the grid and a sample function (Task 7.1).

   2. python script2.py
      - Purpose: Computes and plots the eigenmodes and eigenfrequencies (Task 7.2).

   3. python script3.py
      - Purpose: Runs the wave simulation, creates the final comparison plot, and saves the simulation data (Task 7.3).
      - IMPORTANT: This creates the 'solution_data.npy' file, which is required for Step 4.

   4. python create_animation.py
      - Purpose: (Optional) Creates an animation (.gif) of the wave propagation from the saved data.
      - Prerequisite: The 'solution_data.npy' file must exist.


FILE DESCRIPTIONS
-----------------

help_functions.py   - Module with all core functions (matrix creation, solvers, etc.).

script1.py          - Solution for Task 7.1 (Grid Visualization).

script2.py          - Solution for Task 7.2 (Helmholtz Analysis).

script3.py          - Solution for Task 7.3 (FDTD Analysis & Result Saving).

create_animation.py - Creates an animation (.gif) from the simulation results of script3.py.

*.txt               - Data files containing the grid geometries (L-shape, C-shape).

```
