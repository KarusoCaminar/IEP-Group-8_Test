#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 18:07:38 2020

@author: Ion Gabriel Ion, Dimitrios Loukrezis
"""

import numpy as np
import scipy.sparse 
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import json 
import pickle

from help_functions import *

# Propagation velocity
c = 4

#%% Time domain simulation
print()
print('Time domain simulation of the wave equation on an L-shaped domain')
print()

##### Task 7.3 1): Gitter laden
grid_filename = 'Lshape_2.txt'
pos_l, conn_l, bd_l = load_from_file(grid_filename)

##### Task 7.3 2): Zeit-Parameter und CFL-Bedingung
Tmax = 20.0

inner_point_idx = np.setdiff1d(np.arange(pos_l.shape[0]), bd_l)[0]
hx = pos_l[conn_l[inner_point_idx, 1], 0] - pos_l[inner_point_idx, 0]
hy = pos_l[conn_l[inner_point_idx, 3], 1] - pos_l[inner_point_idx, 1]

tau_max = np.sqrt(hx**2 + hy**2) / (2 * c)
tau = 0.99 * tau_max

Nt = int(np.ceil(Tmax / tau))
Gt = np.linspace(0, Tmax, Nt)

print(f"Maximaler Zeitschritt (tau_max): {tau_max:.6f} s")
print(f"Gewählter Zeitschritt (tau):     {tau:.6f} s")
print(f"Anzahl der Zeitschritte (Nt):    {Nt}")


##### Task 7.3 3): Laplace-Operator erstellen
L_lshape = construct_matrix(pos_l, conn_l, bd_l)


##### Task 7.3 4): Anfangsbedingungen implementieren
p0 = np.exp(-((pos_l[:, 0] - 0.375)**2 / 0.01) - ((pos_l[:, 1] - 0.75)**2 / 0.01))
p0[bd_l] = 0
v0 = np.zeros_like(p0)


##### Task 7.3 6): Wellengleichung lösen
solution_over_time = solve_timedomain(L_lshape, c, Gt, p0, v0)

    
##### Task 7.3 7): Spektralanalyse des Zeitsignals
probe_point_coords = np.array([1.75, 0.25])
distances = np.linalg.norm(pos_l - probe_point_coords, axis=1)
probe_idx = np.argmin(distances)

time_signal = solution_over_time[probe_idx, :]

freqs, spectrum = spectrum_signal(Gt, time_signal)

# Berechne die Eigenfrequenzen aus der Helmholtz-Gleichung zum Vergleich
num_eigenvalues_lshape = 12
eigenvalues_L_l, _ = scipy.sparse.linalg.eigs(L_lshape, k=num_eigenvalues_lshape, which='SR')
eigenvalues_v_l = -np.real(eigenvalues_L_l[eigenvalues_L_l < -1e-9])
eigenfrequencies_l = (c / (2 * np.pi)) * np.sqrt(np.sort(eigenvalues_v_l))


# Plot des Frequenzspektrums
plt.figure(figsize=(12, 6))
plt.plot(freqs, spectrum, label='Spektrum des Zeitsignals am Punkt (1.75, 0.25)')

# KORREKTUR: Robuste Legenden-Logik und angepasster Frequenzbereich
legend_added = False
for f in eigenfrequencies_l:
    # Zeige alle berechneten Eigenfrequenzen im neuen Plotbereich an
    if f < 200: # Angepasster Frequenzbereich
        label_text = ""
        # Füge nur einmal ein Label für die Legende hinzu, um sie nicht zu überladen
        if not legend_added:
            label_text = 'Berechnete Eigenfrequenzen (Helmholtz)'
            legend_added = True
        plt.axvline(x=f, color='r', linestyle='--', label=label_text)

plt.title('Frequenzspektrum vs. berechnete Eigenfrequenzen')
plt.xlabel('Frequenz [Hz]')
plt.ylabel('Amplitude')
# KORREKTUR: Frequenzbereich an die erwarteten Eigenfrequenzen angepasst
plt.xlim(0, 200) 
plt.grid(True)
plt.legend()
plt.show()