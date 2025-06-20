#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 18:07:38 2020
@author: Ion Gabriel Ion, Dimitrios Loukrezis

This script solves Task 7.3:
1. Loads an L-shaped grid.
2. Determines a stable time step using the CFL condition.
3. Sets up initial conditions for a wave simulation.
4. Solves the 2D wave equation using the FDTD method.
5. Extracts a time signal from a probe point and computes its frequency spectrum.
6. Compares the FDTD spectrum with the eigenfrequencies from the Helmholtz solution.
"""

import numpy as np
import matplotlib.pyplot as plt
from help_functions import *

# --- 1. Setup: Load Grid and Define Parameters (Task 7.3.1) ---
FILENAME = 'Lshape_1.txt'
C = 4.0   # Wave propagation speed
T = 20.0  # Total simulation time

print("--- Task 7.3: Solving the Wave Equation in Time Domain ---")
print(f"Loading grid from {FILENAME}...")
pos, conn, bd = load_from_file(FILENAME)
n_points = pos.shape[0]

# --- 2. Time Step Calculation (CFL Condition) (Task 7.3.2) ---
# Determine grid spacing h (assuming hx = hy for simplicity)
inner_indices = np.setdiff1d(np.arange(n_points), bd)
p_idx = inner_indices[0]
h = pos[conn[p_idx, 1], 0] - pos[p_idx, 0]

# The Courant-Friedrichs-Lewy (CFL) condition ensures numerical stability.
dt_max = h / (C * np.sqrt(2))
# Select a time step slightly smaller than the maximum for guaranteed stability.
dt = 0.95 * dt_max
n_steps = int(np.ceil(T / dt))
Gt = np.linspace(0, T, n_steps)

print("\n--- Stability Analysis ---")
print(f"Grid spacing h = {h:.4f}")
print(f"Max stable time step (CFL) dt_max = {dt_max:.6f} s")
print(f"Chosen time step dt = {Gt[1]-Gt[0]:.6f} s")
print(f"Number of time steps for T={T}s: {n_steps}")

# --- 3. Construct Laplace Matrix (Task 7.3.3) ---
print("\nConstructing Laplace matrix L...")
L = construct_matrix(pos, conn, bd)

# --- 4. Define Initial Conditions (Task 7.3.4) ---
print("Defining initial conditions...")
# Initial displacement is a Gaussian pulse.
p0 = np.exp(-((pos[:, 0] - 0.375)**2 / 0.01 + (pos[:, 1] - 0.75)**2 / 0.01))
# Initial velocity is zero.
v0 = np.zeros(n_points)
# Enforce the Dirichlet boundary condition (u=0) on the initial state.
p0[bd] = 0.0

# --- 5. & 6. Solve the Wave Equation (Task 7.3.5 & 7.3.6) ---
print("\nStarting FDTD simulation...")
solution_over_time = solve_timedomain(L, C, Gt, p0, v0)
print("Simulation finished.")

# --- 7. Spectral Analysis and Comparison (Task 7.3.7) ---
# Define a probe point to record the time signal.
probe_point_coords = np.array([1.75, 0.25])
# Find the index of the grid point closest to the probe coordinates.
probe_idx = np.argmin(np.linalg.norm(pos - probe_point_coords, axis=1))
time_signal = solution_over_time[probe_idx, :]
print(f"\nAnalyzing signal at point closest to {probe_point_coords}")

# Compute the frequency spectrum of the recorded signal.
freqs, amps = spectrum_signal(Gt, time_signal)

# --- The Crucial Comparison ---
# For comparison, load the eigenfrequencies calculated for the L-Shape in Task 7.2.
# These values should ideally be recalculated by running script2.py.
eigenfrequencies_from_helmholtz = np.array([
    4.0084, 4.7070, 5.8361, 6.3688, 6.5492, 7.0315, 7.4268, 7.9734
])

# Plot the FDTD spectrum and overlay the Helmholtz eigenfrequencies.
plt.figure(figsize=(14, 7))
plt.plot(freqs, amps, label='Spectrum from Time Signal (FDTD)', color='blue', zorder=2)

# Plot vertical lines at the locations of the calculated eigenfrequencies.
for i, f_eig in enumerate(eigenfrequencies_from_helmholtz):
    if f_eig <= 8.0:
        label = 'Eigenfrequencies (from Helmholtz)' if i == 0 else ""
        plt.axvline(x=f_eig, color='red', linestyle='--', linewidth=1.5, label=label, zorder=1)

# Finalize the plot.
plt.title('Comparison: FDTD Spectrum vs. Helmholtz Eigenfrequencies')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, 8)
plt.grid(True, linestyle=':', which='both')
plt.legend()
plt.show()
