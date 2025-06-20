#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 17:56:49 2020

@author: Ion Gabriel Ion, Dimitrios Loukrezis
"""
import numpy as np
import scipy.sparse 
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import json 
import pickle

from help_functions import *

# propagation velocity
c = 4

#%% Box domain
print()
print('Eigenfrequencies and eigenmodes of a box domain')
print()

# Domain and discretization parameters
lx, ly = 2.0, 1.0
Nx, Ny = 32, 16 
pos_box, conn_box, bd_box = tensor_product_grid(0, lx, 0, ly, Nx, Ny)

# Construct the Laplace operator matrix
L_box = construct_matrix(pos_box, conn_box, bd_box)

##### Task 7.2 2): Eigenwerte und Eigenvektoren berechnen
num_eigenvalues_to_compute = 12 

eigenvalues_L, eigenvectors = scipy.sparse.linalg.eigs(L_box, k=num_eigenvalues_to_compute, which='SR')

eigenvalues_v = -eigenvalues_L
valid_mask = np.real(eigenvalues_v) > 1e-9
eigenvalues_v = np.real(eigenvalues_v[valid_mask])
eigenvectors = np.real(eigenvectors[:, valid_mask])

if eigenvalues_v.size == 0:
    raise RuntimeError("Es wurden keine gültigen Eigenwerte für die Box-Domain gefunden. Überprüfen Sie die Matrix L.")

sort_indices = np.argsort(eigenvalues_v)
eigenvalues_v = eigenvalues_v[sort_indices]
eigenvectors = eigenvectors[:, sort_indices]

eigenfrequencies_fd = (c / (2 * np.pi)) * np.sqrt(eigenvalues_v)

print(f'Computed Eigenfrequencies (FD): {np.round(eigenfrequencies_fd[:8], 4)}')


##### Task 7.2 3): Plot der ersten 6 Eigenmoden
tri_box = get_triangulation(pos_box, conn_box, bd_box)

for i in range(6):
    plt.figure(figsize=(10, 6))
    # KORREKTUR: Der Eigenvektor wird als 3. positional argument (C) übergeben,
    # nicht als 'facecolors'-keyword.
    plt.tripcolor(pos_box[:, 0], pos_box[:, 1], eigenvectors[:, i], triangles=tri_box, cmap='viridis', shading='flat')
    plt.colorbar(label='Amplitude')
    plt.title(f'Eigenmode {i+1} der Box, f = {eigenfrequencies_fd[i]:.4f} Hz')
    plt.xlabel('x-Koordinate')
    plt.ylabel('y-Koordinate')
    plt.axis('equal')
    plt.show()

##### Task 7.2 5): Vergleich mit analytischer Lösung
num_analytical_freqs = 8
eigenfrequencies_analytical = analytical_eigenfrequencies_2d(num_analytical_freqs, c, lx, ly)

print(f'Analytical Eigenfrequencies:    {np.round(eigenfrequencies_analytical, 4)}')

abs_errors = np.abs(eigenfrequencies_analytical - eigenfrequencies_fd[:num_analytical_freqs])
print(f'Absolute Errors:                {np.round(abs_errors, 4)}')


#%% Convergence study
print()
print('Convergence study of eigenfrequencies of a box domain')
print()

##### Task 7.2 6): Konvergenzstudie
discretization_levels = [(16, 8), (32, 16), (64, 32), (80, 40)]
errors_conv = []
num_points_conv = []

f_analytical_first = analytical_eigenfrequencies_2d(1, c, lx, ly)[0]

for (Nx_conv, Ny_conv) in discretization_levels:
    print(f"Calculating for grid size: ({Nx_conv}, {Ny_conv})")
    pos_conv, conn_conv, bd_conv = tensor_product_grid(0, lx, 0, ly, Nx_conv, Ny_conv)
    L_conv = construct_matrix(pos_conv, conn_conv, bd_conv)
    
    vals_conv, _ = scipy.sparse.linalg.eigs(L_conv, k=5, which='SR')
    
    v_conv = -np.real(vals_conv[vals_conv < -1e-9])
    f_fd_first = (c / (2 * np.pi)) * np.sqrt(np.min(v_conv))
    
    errors_conv.append(np.abs(f_fd_first - f_analytical_first))
    num_points_conv.append(pos_conv.shape[0])

plt.figure(figsize=(8, 6))
plt.loglog(num_points_conv, errors_conv, 'bo-', label='Absoluter Fehler der ersten Eigenfrequenz')
plt.title('Konvergenzstudie für Box-Domain')
plt.xlabel('Anzahl der Gitterpunkte (Np)')
plt.ylabel('Absoluter Fehler')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()


#%% Other shapes
print()
print('Other shapes')

##### Task 7.2 7): L-Form Geometrie
pos_l, conn_l, bd_l = load_from_file('Lshape_1.txt')

L_lshape = construct_matrix(pos_l, conn_l, bd_l)

num_eigenvalues_lshape = 12
eigenvalues_L_l, eigenvectors_l = scipy.sparse.linalg.eigs(L_lshape, k=num_eigenvalues_lshape, which='SR')

eigenvalues_v_l = -np.real(eigenvalues_L_l[eigenvalues_L_l < -1e-9])
eigenvectors_l = np.real(eigenvectors_l[:, eigenvalues_L_l < -1e-9])

if eigenvalues_v_l.size == 0:
    raise RuntimeError("Es wurden keine gültigen Eigenwerte für die L-Form gefunden. Überprüfen Sie die Matrix L.")

sort_indices_l = np.argsort(eigenvalues_v_l)
eigenvalues_v_l = eigenvalues_v_l[sort_indices_l]
eigenvectors_l = eigenvectors_l[:, sort_indices_l]

eigenfrequencies_l = (c / (2 * np.pi)) * np.sqrt(eigenvalues_v_l)

print(f'Computed Eigenfrequencies (L-Shape): {np.round(eigenfrequencies_l[:8], 4)}')

tri_l = get_triangulation(pos_l, conn_l, bd_l)
for i in range(8):
    plt.figure(figsize=(8, 6))
    # KORREKTUR: Auch hier den Eigenvektor als 3. positional argument übergeben.
    plt.tripcolor(pos_l[:, 0], pos_l[:, 1], eigenvectors_l[:, i], triangles=tri_l, cmap='viridis', shading='flat')
    plt.colorbar(label='Amplitude')
    plt.title(f'Eigenmode {i+1} der L-Form, f = {eigenfrequencies_l[i]:.4f} Hz')
    plt.xlabel('x-Koordinate')
    plt.ylabel('y-Koordinate')
    plt.axis('equal')
    plt.show()