#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 08:57:33 2020
@author: Ion Gabriel Ion, Dimitrios Loukrezis

This module provides helper functions for the Scientific Computing with Python exercise.
It includes utilities for loading and saving grid data, constructing the discrete
Laplace operator, and solving the time-domain wave equation.
"""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import pickle

def load_from_file(fname):
    """
    Loads grid data from a specified pickle file.

    Args:
        fname (str): The name of the file to load.

    Returns:
        tuple: A tuple containing:
            - pos (np.ndarray): Np x 2 array of point coordinates.
            - conn (np.ndarray): Np x 4 array of neighbor indices.
            - bd (list): List of indices corresponding to boundary points.
    """
    with open(fname, 'rb') as file:
        data = pickle.load(file)
    pos = data["pos"]
    conn = data["conn"]
    bd = data["bd"]
    return pos, conn, bd


def save_to_file(fname, pos, conn, bd):
    """
    Saves a grid geometry description to a pickle file.
    """
    to_write = {'pos': pos, 'conn': conn, 'bd': bd}
    with open(fname, 'wb') as file:
        pickle.dump(to_write, file)


def analytical_eigenfrequencies_2d(Ne, c, lx, ly):
    """
    Computes the first Ne analytical eigenfrequencies for a rectangular domain.

    Args:
        Ne (int): The number of eigenfrequencies to compute.
        c (float): The wave propagation speed.
        lx (float): The length of the domain in the x-direction.
        ly (float): The length of the domain in the y-direction.

    Returns:
        np.ndarray: An array of the first Ne unique, sorted eigenfrequencies.
    """
    # Generate a sufficient number of (m, n) modes to find Ne unique frequencies
    limit = int(np.sqrt(Ne)) + 5
    m_modes = np.arange(1, limit + 1)
    n_modes = np.arange(1, limit + 1)
    
    eigenfrequencies = []
    
    # Calculate frequencies for all (m, n) combinations
    for m in m_modes:
        for n in n_modes:
            term1 = (m * np.pi / lx)**2
            term2 = (n * np.pi / ly)**2
            freq = (c / (2 * np.pi)) * np.sqrt(term1 + term2)
            eigenfrequencies.append(freq)
    
    # Return the first Ne unique, sorted frequencies
    unique_freqs = sorted(list(set(eigenfrequencies)))
    return np.array(unique_freqs[:Ne])


def tensor_product_grid(ax, bx, ay, by, Nx, Ny):
    """
    Constructs a grid for a rectangular domain [ax, bx] x [ay, by].
    """
    xs = np.linspace(ax, bx, Nx)
    ys = np.linspace(ay, by, Ny)

    X, Y = np.meshgrid(xs, ys)
    IDX = np.arange(xs.size * ys.size).reshape(X.shape)
    
    # Build connectivity matrix for a structured grid
    Xm = np.hstack((-np.ones((X.shape[0], 1), dtype=np.int32), IDX[:, :-1]))
    Xp = np.hstack((IDX[:, 1:], -np.ones((X.shape[0], 1), dtype=np.int32)))
    Ym = np.vstack((-np.ones((1, X.shape[1]), dtype=np.int32), IDX[:-1, :]))
    Yp = np.vstack((IDX[1:, :], -np.ones((1, X.shape[1]), dtype=np.int32)))
    
    pos = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
    conn = np.hstack((Xm.reshape(-1, 1), Xp.reshape(-1, 1), Ym.reshape(-1, 1), Yp.reshape(-1, 1)))
    
    # Identify boundary indices
    bd_indices = list(set(IDX[:, 0].tolist() + IDX[:, -1].tolist() + IDX[0, :].tolist() + IDX[-1, :].tolist()))
    
    return pos, conn, bd_indices


def construct_matrix(positions, connectivity, boundary):
    """
    Constructs the sparse discrete Laplace operator using a 5-point stencil.
    
    This function implements the Finite Difference Method to approximate the
    Laplacian operator. For boundary points, it sets the diagonal element to 1
    to later enforce Dirichlet boundary conditions.
    """
    Np = positions.shape[0]
    
    # Determine grid spacings hx and hy from a sample inner point
    inner_point_idx = np.setdiff1d(np.arange(Np), boundary)[0]
    hx = positions[connectivity[inner_point_idx, 1], 0] - positions[inner_point_idx, 0]
    hy = positions[connectivity[inner_point_idx, 3], 1] - positions[inner_point_idx, 1]
    
    data, row, col = [], [], []

    # Iterate through all grid points to build the matrix entries
    for i in range(Np):
        if i in boundary:
            # For Dirichlet boundary points, set L_ii = 1 and other entries in row i to 0
            row.append(i)
            col.append(i)
            data.append(1.0)
        else:
            # For inner points, apply the 5-point stencil for the Laplacian
            # Off-diagonal elements
            row.extend([i, i, i, i])
            col.extend([connectivity[i, 0], connectivity[i, 1], connectivity[i, 2], connectivity[i, 3]])
            data.extend([1 / hx**2, 1 / hx**2, 1 / hy**2, 1 / hy**2])
            
            # Diagonal element
            row.append(i)
            col.append(i)
            data.append(-2 / hx**2 - 2 / hy**2)
            
    # Construct the sparse matrix in COO format for efficiency
    L = scipy.sparse.coo_matrix((data, (row, col)), shape=(Np, Np))
   
    # Convert to CSR format for fast matrix-vector products
    return L.tocsr()


def get_triangulation(positions, connectivity, boundary):
    """
    Computes a triangulation of the domain for plotting purposes.
    """
    tri = []
    Np = positions.shape[0]
    
    # Iterate through all points to form triangles with their neighbors
    for i in range(Np):
        if i not in boundary:
            # Inner points can form triangles in four directions
            tri.append([connectivity[i, 2], i, connectivity[i, 0]])
            tri.append([connectivity[i, 1], i, connectivity[i, 3]])
        else:
            # Boundary points form triangles only if neighbors are valid
            if connectivity[i, 0] != -1 and connectivity[i, 2] != -1:
                tri.append([connectivity[i, 2], i, connectivity[i, 0]])
            if connectivity[i, 1] != -1 and connectivity[i, 3] != -1:
                tri.append([connectivity[i, 1], i, connectivity[i, 3]])
            
    return np.array(tri)

    
def spectrum_signal(Gt, signal):
    """
    Performs a discrete Fourier transform (FFT) on a given time signal.
    """
    # Ensure signal has an even number of samples for simplicity
    n_samples = Gt.size // 2 * 2
    Gt = Gt[:n_samples]
    signal = signal[:n_samples]
    
    dt = Gt[1] - Gt[0]
    # Use rfftfreq for real-valued signals to get positive frequencies
    freqs = np.fft.rfftfreq(n_samples, d=dt)
    
    # Compute FFT and normalize the amplitude
    Cs = np.fft.rfft(signal)
    Cs = np.abs(Cs) / n_samples
   
    return freqs, Cs


def solve_timedomain(L, c, Gt, p0, v0):
    """
    Solves the 2D wave equation using an explicit FDTD leapfrog scheme.

    Args:
        L (scipy.sparse.csr_matrix): The discrete Laplace operator.
        c (float): Wave propagation speed.
        Gt (np.ndarray): The time grid.
        p0 (np.ndarray): Initial displacement at t=0.
        v0 (np.ndarray): Initial velocity at t=0.

    Returns:
        np.ndarray: A Np x Nt matrix containing the solution over time.
    """
    Nt = Gt.size
    Np = L.shape[0]
    tau = Gt[1] - Gt[0]
    
    solution = np.zeros((Np, Nt))
    solution[:, 0] = p0
    
    # First step requires special handling using the initial velocity
    # Approximate u^{-1} using a backward difference for the velocity
    u_prev = p0 - tau * v0
    
    # Compute u^1 using the main update rule with k=0
    solution[:, 1] = tau**2 * c**2 * (L @ p0) + 2 * p0 - u_prev
    
    # Main time-stepping loop (leapfrog method)
    for k in range(1, Nt - 1):
        if k % 100 == 0:
            print(f'Time step: {k} / {Nt-1}')
        
        # u^{k+1} = tau^2*c^2*L*u^k + 2*u^k - u^{k-1}
        u_next = tau**2 * c**2 * (L @ solution[:, k]) + 2 * solution[:, k] - solution[:, k-1]
        
        solution[:, k+1] = u_next
        
    return solution
