#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 14:42:13 2020

@author: Ion Gabriel Ion, Dimitrios Loukrezis
"""

import numpy as np
# Unbenutzte Module wurden für sauberen Code entfernt.
# import scipy.sparse 
# import scipy.sparse.linalg
import matplotlib.pyplot as plt

# Import der notwendigen Hilfsfunktionen.
# Die Datei help_functions.py muss sich im selben Verzeichnis befinden.
from help_functions import load_from_file, get_triangulation

# --- Anfang der Lösung ---

# Dateiname für das zu ladende Gitter. 
# Sie können hier auch 'Lshape_2.txt', 'Cshape_1.txt' oder 'Cshape_2.txt' verwenden.
grid_filename = 'Lshape_1.txt'


##### Task 7.1 1): Gitter laden und als Scatter-Plot darstellen
print(f"Lade Gitter aus der Datei: {grid_filename}")

# Lade die Gitterdaten (Positionen, Konnektivität, Randpunkt-Indizes) aus der Datei.
pos, conn, bd = load_from_file(grid_filename) # 

# Erstelle eine neue Figur für die Visualisierung.
plt.figure(figsize=(10, 8))

# Identifiziere die Indizes der inneren Punkte.
# Dies ist die Menge aller Punkte (von 0 bis Np-1) abzüglich der Randpunkte.
all_indices = np.arange(pos.shape[0])
inner_points_indices = np.setdiff1d(all_indices, bd)

# Plotte die inneren Punkte in einer Farbe (z.B. Blau).
plt.scatter(pos[inner_points_indices, 0], pos[inner_points_indices, 1], c='blue', label='Innere Punkte', s=15) # 
# Plotte die Randpunkte in einer anderen Farbe (z.B. Rot).
plt.scatter(pos[bd, 0], pos[bd, 1], c='red', label='Randpunkte', s=15) # 

# Füge dem Plot Titel und Achsenbeschriftungen hinzu, um die Übersichtlichkeit zu gewährleisten.
plt.title(f'Gitter-Visualisierung von "{grid_filename}"')
plt.xlabel('x-Koordinate')
plt.ylabel('y-Koordinate')
plt.legend()
plt.grid(True)
plt.axis('equal') # Stellt sicher, dass die Skalierung auf x- und y-Achse gleich ist.
plt.show()


##### Task 7.1 2): Funktion f(x,y) auf dem Gitter darstellen

# Definiere die Funktion f(x,y) = sin(2*pi*x) * sin(2*pi*y).
# Wir wenden sie direkt auf die gesamten x- und y-Koordinaten-Arrays an (vektorisierte Berechnung).
x_coords = pos[:, 0]
y_coords = pos[:, 1]
f_values = np.sin(2 * np.pi * x_coords) * np.sin(2 * np.pi * y_coords) # 

# Berechne die Triangulierung, die für den tripcolor-Plot benötigt wird.
# Diese zerlegt die Fläche in kleine Dreiecke.
triangulation = get_triangulation(pos, conn, bd) # 

# Erstelle eine neue Figur für den tripcolor-Plot.
plt.figure(figsize=(10, 8))

# Plotte die Funktionswerte auf dem triangulierten Gitter.
# Die Farbe jedes Dreiecks repräsentiert den Funktionswert.
plt.tripcolor(pos[:, 0], pos[:, 1], triangles=triangulation, facecolors=f_values, cmap='viridis', shading='flat') # 

# Füge eine Farbleiste hinzu, um die Funktionswerte zu interpretieren.
plt.colorbar(label='Funktionswerte f(x,y)') # 

# Füge Titel und Beschriftungen hinzu.
plt.title('Darstellung von f(x,y) = sin(2πx)sin(2πy) auf dem Gitter')
plt.xlabel('x-Koordinate')
plt.ylabel('y-Koordinate')
plt.grid(True)
plt.axis('equal')
plt.show()


# --- Ende der Lösung ---