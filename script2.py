#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 17:56:49 2020
@author: Ion Gabriel Ion, Dimitrios Loukrezis

This script solves Task 7.2:
1. Computes eigenfrequencies and eigenmodes of a 2D Helmholtz equation on a box domain.
2. Compares the numerical results with an analytical solution.
3. Performs a convergence study to validate the FDM implementation.
4. Repeats the analysis for a more complex L-shaped domain.
"""
import numpy as np
import scipy.sparse 
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from help_functions import *

# Global parameter
C = 4.0  # Propagation speed

# --- Part 1: Box Domain Analysis ---
print("\n--- Eigenfrequencies and Eigenmodes of a Box Domain ---")

# --- Domain and Discretization Setup ---
lx, ly = 2.0, 1.0
Nx, Ny = 32, 16 
pos_box, conn_box, bd_box = tensor_product_grid(0, lx, 0, ly, Nx, Ny)
L_box = construct_matrix(pos_box, conn_box, bd_box)

# --- Task 7.2.2: Compute Eigenvalues and Eigenvectors ---
num_eigenvalues_to_compute = 12 
# Use scipy.sparse.linalg.eigs for sparse matrices.
# 'SR' finds eigenvalues with the smallest real part, corresponding to the lowest frequencies.
eigenvalues_L, eigenvectors = scipy.sparse.linalg.eigs(L_box, k=num_eigenvalues_to_compute, which='SR')

# Convert eigenvalues of L to physical eigenvalues of Helmholtz equation (v = -lambda_L)
eigenvalues_v = -eigenvalues_L
# Filter out non-physical (zero or negative) eigenvalues resulting from numerical inaccuracies.
valid_mask = np.real(eigenvalues_v) > 1e-9
eigenvalues_v = np.real(eigenvalues_v[valid_mask])
eigenvectors = np.real(eigenvectors[:, valid_mask])

if eigenvalues_v.size == 0:
    raise RuntimeError("No valid eigenvalues found for the box domain. Check matrix L.")

# Sort eigenvalues and corresponding eigenvectors in ascending order.
sort_indices = np.argsort(eigenvalues_v)
eigenvalues_v = eigenvalues_v[sort_indices]
eigenvectors = eigenvectors[:, sort_indices]

# Convert eigenvalues to physical eigenfrequencies in Hz.
eigenfrequencies_fd = (C / (2 * np.pi)) * np.sqrt(eigenvalues_v)
print(f"First 8 Computed Eigenfrequencies (FD): {np.round(eigenfrequencies_fd[:8], 4)}")

# --- Task 7.2.3: Plot the first 6 Eigenmodes ---
tri_box = get_triangulation(pos_box, conn_box, bd_box)
for i in range(6):
    plt.figure(figsize=(10, 6))
    plt.tripcolor(pos_box[:, 0], pos_box[:, 1], eigenvectors[:, i], triangles=tri_box, cmap='viridis', shading='flat')
    plt.colorbar(label='Amplitude')
    plt.title(f'Eigenmode {i+1} of Box, f = {eigenfrequencies_fd[i]:.4f} Hz')
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.axis('equal')
    plt.show()

# --- Task 7.2.5: Compare with Analytical Solution ---
num_analytical_freqs = 8
eigenfrequencies_analytical = analytical_eigenfrequencies_2d(num_analytical_freqs, C, lx, ly)
print(f"\nAnalytical Eigenfrequencies:          {np.round(eigenfrequencies_analytical, 4)}")
abs_errors = np.abs(eigenfrequencies_analytical - eigenfrequencies_fd[:num_analytical_freqs])
print(f"Absolute Errors:                      {np.round(abs_errors, 4)}")

# --- Part 2: Convergence Study (Task 7.2.6) ---
print("\n--- Convergence Study for the First Eigenfrequency ---")
discretization_levels = [(16, 8), (32, 16), (64, 32), (80, 40)]
errors_conv = []
num_points_conv = []
f_analytical_first = analytical_eigenfrequencies_2d(1, C, lx, ly)[0]

for (Nx_conv, Ny_conv) in discretization_levels:
    print(f"Calculating for grid size: ({Nx_conv}, {Ny_conv})")
    pos_conv, conn_conv, bd_conv = tensor_product_grid(0, lx, 0, ly, Nx_conv, Ny_conv)
    L_conv = construct_matrix(pos_conv, conn_conv, bd_conv)
    
    vals_conv, _ = scipy.sparse.linalg.eigs(L_conv, k=5, which='SR')
    
    # Calculate the first eigenfrequency from the smallest valid eigenvalue
    v_conv = -np.real(vals_conv[vals_conv < -1e-9])
    f_fd_first = (C / (2 * np.pi)) * np.sqrt(np.min(v_conv))
    
    errors_conv.append(np.abs(f_fd_first - f_analytical_first))
    num_points_conv.append(pos_conv.shape[0])

# Plotting the convergence results on a log-log scale
plt.figure(figsize=(8, 6))
plt.loglog(num_points_conv, errors_conv, 'bo-', label='Absolute Error of First Eigenfrequency')
plt.title('Convergence Study for Box Domain')
plt.xlabel('Number of Grid Points (Np)')
plt.ylabel('Absolute Error')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

# --- Part 3: L-Shaped Domain Analysis (Task 7.2.7) ---
print("\n--- Eigenfrequencies and Eigenmodes of an L-Shaped Domain ---")
pos_l, conn_l, bd_l = load_from_file('Lshape_1.txt')
L_lshape = construct_matrix(pos_l, conn_l, bd_l)

# The process is identical to the box domain, showcasing the code's generality.
eigenvalues_L_l, eigenvectors_l = scipy.sparse.linalg.eigs(L_lshape, k=num_eigenvalues_lshape, which='SR')
eigenvalues_v_l = -np.real(eigenvalues_L_l[eigenvalues_L_l < -1e-9])
eigenvectors_l = np.real(eigenvectors_l[:, eigenvalues_L_l < -1e-9])

if eigenvalues_v_l.size == 0:
    raise RuntimeError("No valid eigenvalues found for the L-Shape. Check matrix L.")

sort_indices_l = np.argsort(eigenvalues_v_l)
eigenvalues_v_l = eigenvalues_v_l[sort_indices_l]
eigenvectors_l = eigenvectors_l[:, sort_indices_l]
eigenfrequencies_l = (C / (2 * np.pi)) * np.sqrt(eigenvalues_v_l)
print(f"First 8 Computed Eigenfrequencies (L-Shape): {np.round(eigenfrequencies_l[:8], 4)}")

# Plot the first 8 eigenmodes for the L-shaped domain
tri_l = get_triangulation(pos_l, conn_l, bd_l)
for i in range(8):
    plt.figure(figsize=(8, 6))
    plt.tripcolor(pos_l[:, 0], pos_l[:, 1], eigenvectors_l[:, i], triangles=tri_l, cmap='viridis', shading='flat')
    plt.colorbar(label='Amplitude')
    plt.title(f'Eigenmode {i+1} of L-Shape, f = {eigenfrequencies_l[i]:.4f} Hz')
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.axis('equal')
    plt.show()
