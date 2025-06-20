#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 08:57:33 2020

@author: Ion Gabriel Ion, Dimitrios Loukrezis
"""

import numpy as np
import scipy.sparse 
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import json 
import pickle

def load_from_file(fname):
    """
    Loads the grid from the given file.
    """
    with open(fname, 'rb') as file:
        data = pickle.load(file)
    pos = data["pos"]
    conn = data["conn"]
    bd = data["bd"]
    return pos,conn,bd


def save_to_file(fname,pos,conn,bd):
    """
    Save the geometry description to a file
    """
    to_write = {'pos':pos, 'conn':conn, 'bd': bd}
    with open(fname, 'wb') as file:
        pickle.dump(to_write,file)

    
def analytical_eigenfrequencies_2d(Ne, c, lx, ly):
    """
    Computes the first Ne eigenvalues using the analytical solution for a box domain.
    """
    limit = int(np.sqrt(Ne)) + 5 
    m_modes = np.arange(1, limit + 1)
    n_modes = np.arange(1, limit + 1)
    
    eigenfrequencies = []
    
    for m in m_modes:
        for n in n_modes:
            term1 = (m * np.pi / lx)**2
            term2 = (n * np.pi / ly)**2
            freq = (c / (2 * np.pi)) * np.sqrt(term1 + term2)
            eigenfrequencies.append(freq)
            
    unique_freqs = sorted(list(set(eigenfrequencies)))
    
    return np.array(unique_freqs[:Ne])


def tensor_product_grid(ax,bx,ay,by,Nx,Ny):
    """
    constructs the grid corresponding to the box domain [ax,bx] x [ay,by].
    """
    xs = np.linspace(ax,bx,Nx)
    ys = np.linspace(ay,by,Ny)

    X,Y = np.meshgrid(xs,ys)
    IDX = np.arange(xs.size*ys.size).reshape(X.shape)
    Xm = np.hstack( (-np.ones((X.shape[0],1),dtype=np.int32),IDX[:,:-1]) )
    Xp = np.hstack( (IDX[:,1:],-np.ones((X.shape[0],1),dtype=np.int32)) )
    Ym = np.vstack( (-np.ones((1,X.shape[1]),dtype=np.int32),IDX[:-1,:]))
    Yp = np.vstack( (IDX[1:,:],-np.ones((1,X.shape[1]),dtype=np.int32)))
    
    pos = np.hstack((X.reshape([-1,1]),Y.reshape([-1,1])))
    conn = np.hstack((Xm.reshape([-1,1]),Xp.reshape([-1,1]),Ym.reshape([-1,1]),Yp.reshape([-1,1])))
    
    bd_indices = list(set(IDX[:,0].tolist() + IDX[:,-1].tolist() + IDX[0,:].tolist() + IDX[-1,:].tolist() ))
    
    return pos,conn,bd_indices
    
def construct_matrix(positions,connectivity,boundary):
    """
    Constructs the discrete Lapalce operator as explained in the script.
    """
    Np = positions.shape[0] 
    inner_point_idx = np.setdiff1d(np.arange(Np), boundary)[0]
    
    hx = positions[connectivity[inner_point_idx, 1], 0] - positions[inner_point_idx, 0]
    hy = positions[connectivity[inner_point_idx, 3], 1] - positions[inner_point_idx, 1]
    
    data, row, col = [], [], []

    for i in range(Np):
        if i in boundary:
            row.append(i)
            col.append(i)
            data.append(1.0)
        else:
            row.append(i)
            col.append(connectivity[i, 0])
            data.append(1 / hx**2)
            
            row.append(i)
            col.append(connectivity[i, 1])
            data.append(1 / hx**2)
            
            row.append(i)
            col.append(connectivity[i, 2])
            data.append(1 / hy**2)

            row.append(i)
            col.append(connectivity[i, 3])
            data.append(1 / hy**2)

            row.append(i)
            col.append(i)
            data.append(-2 / hx**2 - 2 / hy**2)
            
    L = scipy.sparse.coo_matrix((data, (row, col)), shape=(Np, Np))
   
    return L
    
def get_triangulation(positions,connectivity,boundary):
    """
    Compute the traingulation given the grid description.
    """
    tri = []
    Np = positions.shape[0]
    for i in range(Np):
        if not i in boundary:
            tri.append([connectivity[i,2],i,connectivity[i,0]])
            tri.append([connectivity[i,1],i,connectivity[i,3]])
        else:
            if  connectivity[i,0]!=-1 and connectivity[i,2]!=-1 and (connectivity[i,1]==-1 or connectivity[i,3]==-1 or connectivity[i,1] in boundary or connectivity[i,3] in boundary):
                tri.append([connectivity[i,2],i,connectivity[i,0]])
            if  connectivity[i,1]!=-1 and connectivity[i,3]!=-1 and (connectivity[i,0]==-1 or connectivity[i,2]==-1 or connectivity[i,0] in boundary or connectivity[i,2] in boundary):
                tri.append([connectivity[i,1],i,connectivity[i,3]])
            
    tri = np.array(tri)
    return tri
    
def spectrum_signal(Gt, signal):
    """
    Perform the discrete Fourier transform of a given time signal
    """
    # only signals with an even size
    Gt = Gt[:Gt.size//2*2]
    signal = signal[:signal.size//2*2]
    
    dt    = Gt[1] - Gt[0]
    freqs = np.linspace(0.0, 1.0/(2*dt), Gt.size//2)
    
    Cs = np.fft.fft(signal)
    Cs = np.abs(Cs[:Cs.size//2])
    Cs /= Cs.size  
   
    return freqs, Cs

def solve_timedomain(L, c, Gt, p0, v0):
    """
    Solves the wave equation on the given time grid with the initial 
    conditions u(x,y,t=0) = p0(x,y) and u'(x,y,t=0) = v0(x,y).
    """
    
    ##### Task 7.3 5)
    
    Nt = Gt.size
    Np = L.shape[0]
    tau = Gt[1] - Gt[0]
    
    # Initialisiere die Lösungsmatrix, um die Zustände u für alle Zeitpunkte zu speichern.
    solution = np.zeros((Np, Nt))
    
    # Setze die Anfangsbedingung für die Auslenkung (t=0)
    solution[:, 0] = p0
    
    # --- Spezielle Behandlung für den ersten Zeitschritt (k=0) ---
    # Wir benötigen u bei t=-tau (u^{-1}), um die Schleife zu starten.
    # Dies wird aus der Anfangsgeschwindigkeit v0 approximiert: u^{-1} = u^0 - tau*v0
    u_prev = p0 - tau * v0
    
    # Berechne den Zustand zum Zeitpunkt t=tau (u^1) mit der Update-Formel.
    # Hier verwenden wir u^0 (also p0) und das fiktive u^{-1} (also u_prev).
    # u_curr entspricht u^1.
    u_curr = tau**2 * c**2 * (L @ p0) + 2 * p0 - u_prev
    solution[:, 1] = u_curr
    
    # --- Hauptschleife für alle weiteren Zeitschritte ---
    # Die Schleife läuft von k=1 bis Nt-2, um u^2 bis u^{Nt-1} zu berechnen.
    for k in range(1, Nt - 1):
        if k % 100 == 0:
            print(f'time step: {k} / {Nt-1}')
        
        # u^{k+1} = tau^2*c^2*L*u^k + 2*u^k - u^{k-1}
        # In unseren Variablen:
        # u_next = tau^2*c^2*L*u_curr + 2*u_curr - u_prev
        u_next = tau**2 * c**2 * (L @ solution[:, k]) + 2 * solution[:, k] - solution[:, k-1]
        
        # Speichere das Ergebnis für den nächsten Zeitschritt
        solution[:, k+1] = u_next
        
    return solution