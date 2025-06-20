#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full script to solve the 2D wave equation on a given domain, based on
Exercise 7.3.

This script is adapted to be fully compatible with the specific
`help_functions.py` provided for the course. It has been combined into
a single file for ease of use.
"""

import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.animation as animation
import pickle

# =============================================================================
# HELPER FUNCTIONS (as provided in the user's file)
# The solve_timedomain function has been completed as per Task 7.3.5.
# =============================================================================

def load_from_file(fname):
    """
    Loads the grid from the given file using pickle.
    """
    with open(fname, 'rb') as file:
        data = pickle.load(file)
    pos = data["pos"]
    conn = data["conn"]
    bd = data["bd"]
    return pos, conn, bd


def save_to_file(fname, pos, conn, bd):
    """
    Save the geometry description to a file.
    """
    to_write = {'pos': pos, 'conn': conn, 'bd': bd}
    with open(fname, 'wb') as file:
        pickle.dump(to_write, file)


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


def tensor_product_grid(ax, bx, ay, by, Nx, Ny):
    """
    Constructs the grid corresponding to the box domain [ax,bx] x [ay,by].
    """
    xs = np.linspace(ax, bx, Nx)
    ys = np.linspace(ay, by, Ny)

    X, Y = np.meshgrid(xs, ys)
    IDX = np.arange(xs.size * ys.size).reshape(X.shape)
    Xm = np.hstack((-np.ones((X.shape[0], 1), dtype=np.int32), IDX[:, :-1]))
    Xp = np.hstack((IDX[:, 1:], -np.ones((X.shape[0], 1), dtype=np.int32)))
    Ym = np.vstack((-np.ones((1, X.shape[1]), dtype=np.int32), IDX[:-1, :]))
    Yp = np.vstack((IDX[1:, :], -np.ones((1, X.shape[1]), dtype=np.int32)))
    
    pos = np.hstack((X.reshape([-1, 1]), Y.reshape([-1, 1])))
    conn = np.hstack((Xm.reshape([-1, 1]), Xp.reshape([-1, 1]), Ym.reshape([-1, 1]), Yp.reshape([-1, 1])))
    
    bd_indices = list(set(IDX[:, 0].tolist() + IDX[:, -1].tolist() + IDX[0, :].tolist() + IDX[-1, :].tolist()))
    
    return pos, conn, bd_indices


def construct_matrix(positions, connectivity, boundary):
    """
    Constructs the discrete Laplace operator as explained in the script.
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
   
    return L.tocsr() # Convert to CSR for efficient matrix-vector products


def get_triangulation(positions, connectivity, boundary):
    """
    Compute the triangulation given the grid description.
    """
    tri = []
    Np = positions.shape[0]
    all_points = set(range(Np))
    inner_points = all_points - set(boundary)

    for i in inner_points:
        # For inner points, we can form triangles with neighbors
        neighbors = connectivity[i]
        if neighbors[0] != -1 and neighbors[2] != -1:
            tri.append([i, neighbors[0], neighbors[2]])
        if neighbors[1] != -1 and neighbors[2] != -1:
            tri.append([i, neighbors[2], neighbors[1]])
        if neighbors[1] != -1 and neighbors[3] != -1:
            tri.append([i, neighbors[1], neighbors[3]])
        if neighbors[0] != -1 and neighbors[3] != -1:
            tri.append([i, neighbors[3], neighbors[0]])
            
    tri = np.array(tri)
    # Remove duplicate triangles if any
    unique_tri = np.unique(np.sort(tri, axis=1), axis=0)
    return mtri.Triangulation(positions[:, 0], positions[:, 1], triangles=unique_tri)


def spectrum_signal(Gt, signal):
    """
    Perform the discrete Fourier transform of a given time signal.
    """
    # Only signals with an even size
    n_samples = len(Gt) // 2 * 2
    Gt = Gt[:n_samples]
    signal = signal[:n_samples]
    
    dt = Gt[1] - Gt[0]
    freqs = np.fft.rfftfreq(n_samples, d=dt)
    
    Cs = np.fft.rfft(signal)
    Cs = np.abs(Cs) / n_samples
   
    return freqs, Cs


def solve_timedomain(L, c, Gt, p0, v0):
    """
    Solves the wave equation on the given time grid with the initial 
    conditions u(x,y,t=0) = p0(x,y) and u'(x,y,t=0) = v0(x,y).
    This function implements the core of Task 7.3.5.
    """
    Nt = Gt.size
    Np = L.shape[0]
    tau = Gt[1] - Gt[0]
    
    # Initialize the solution matrix to store the state u at all time steps
    solution = np.zeros((Np, Nt))
    
    # Set the initial condition for displacement (t=0)
    solution[:, 0] = p0
    
    # --- Special handling for the first time step (k=0) ---
    # We need u at t=-tau (u^{-1}) to start the loop.
    # This is approximated from the initial velocity v0: u^{-1} = u^0 - tau*v0
    u_prev = p0 - tau * v0
    
    # Now use the main update formula for k=0 to find u at t=tau (u^1)
    # u_curr corresponds to u^1
    u_curr = tau**2 * c**2 * (L @ p0) + 2 * p0 - u_prev
    solution[:, 1] = u_curr
    
    # --- Main loop for all subsequent time steps (Leapfrog method) ---
    # The loop runs from k=1 to Nt-2, to compute u^2 up to u^{Nt-1}.
    for k in range(1, Nt - 1):
        if k % 100 == 0:
            print(f'Time step: {k} / {Nt-1}')
        
        # u^{k+1} = tau^2*c^2*L*u^k + 2*u^k - u^{k-1}
        # In our variables:
        # u_next is the solution at k+1
        # solution[:, k] is u^k
        # solution[:, k-1] is u^{k-1}
        u_next = tau**2 * c**2 * (L @ solution[:, k]) + 2 * solution[:, k] - solution[:, k-1]
        
        # Store the result for the next time step
        solution[:, k+1] = u_next
        
    return solution

# =============================================================================
# MAIN SCRIPT
# =============================================================================

if __name__ == "__main__":
    
    # --- 1. Parameters and Grid Loading (Task 7.3.1) ---
    FILENAME = 'Lshape_1.txt' # This must be a .pkl file for load_from_file
    C = 4.0   # Wave propagation speed
    T = 20.0  # Total simulation time

    print(f"Loading grid from {FILENAME}...")
    try:
        # Use the provided loading function
        pos, conn, bd = load_from_file(FILENAME)
        n_points = pos.shape[0]
        print(f"Grid with {n_points} points loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: The file '{FILENAME}' was not found.")
        print("Please ensure it's a pickle file and is in the same directory.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        exit()

    # --- 2. Calculate Stability Condition (CFL) (Task 7.3.2) ---
    # Find grid spacing h (assuming hx=hy)
    all_indices = np.arange(n_points)
    inner_indices = np.setdiff1d(all_indices, bd)
    
    p_idx = inner_indices[0]
    neighbor_idx = conn[p_idx, 1] # Index of the right neighbor
    h = pos[neighbor_idx, 0] - pos[p_idx, 0]

    # CFL condition for 2D: dt <= h / (c * sqrt(2))
    dt_max = h / (C * np.sqrt(2))
    # Choose dt slightly smaller than the maximum for guaranteed stability
    dt = 0.9 * dt_max 
    n_steps = int(np.ceil(T / dt))
    # Create the time grid, which is required by the provided functions
    Gt = np.linspace(0, T, n_steps)

    print("\n--- Stability Analysis (CFL) ---")
    print(f"Grid spacing h = {h:.4f}")
    print(f"Maximum stable time step dt_max = {dt_max:.6f} s")
    print(f"Chosen time step dt = {Gt[1]-Gt[0]:.6f} s")
    print(f"Number of time steps for T={T}s: {n_steps}")

    # --- 3. Construct Laplace Matrix L (Task 7.3.3) ---
    print("\nConstructing Laplace matrix L...")
    L = construct_matrix(pos, conn, bd)

    # --- 4. Define Initial Conditions (Task 7.3.4) ---
    print("Defining initial conditions...")
    x = pos[:, 0]
    y = pos[:, 1]

    # Initial displacement: A Gaussian pulse
    p0 = np.exp(-((x - 0.375)**2 / 0.01 + (y - 0.75)**2 / 0.01))
    # Initial velocity is zero
    v0 = np.zeros(n_points)

    # Important: Apply boundary conditions to initial values
    p0[bd] = 0.0
    v0[bd] = 0.0

    # --- 5. & 6. Solve the Wave Equation (Task 7.3.5 & 7.3.6) ---
    print("\nStarting FDTD simulation (this may take a moment)...")
    solution = solve_timedomain(L, C, Gt, p0, v0)
    print("Simulation finished.")

    # --- 7. Extract and Analyze Time Signal (Task 7.3.7) ---
    # Find the index of the point closest to (1.75, 0.25)
    target_point = np.array([1.75, 0.25])
    distances = np.linalg.norm(pos - target_point, axis=1)
    signal_idx = np.argmin(distances)

    print(f"\nExtracting time signal at point {pos[signal_idx]} (Index {signal_idx})")
    time_signal = solution[signal_idx, :]

    # Calculate frequency spectrum
    freqs, amps = spectrum_signal(Gt, time_signal)

    # Plot the frequency spectrum
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, amps, 'b-')
    plt.title(f'Frequency Spectrum of the Signal at Point ({pos[signal_idx][0]}, {pos[signal_idx][1]})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim(0, 8) # As required by the exercise
    plt.grid(True)
    plt.show()

    # --- Optional: Visualize the wave propagation as an animation ---
    print("\nCreating animation of the wave propagation (close the plot window to exit)...")
    fig, ax = plt.subplots(figsize=(8, 6))
    # Use the provided get_triangulation function
    triang = get_triangulation(pos, conn, bd) 
    
    # Find min/max values for a consistent color scale
    vmax = np.max(np.abs(solution)) * 0.5 # Scale down for better visualization
    vmin = -vmax

    # Initial plot
    tripcolor_plot = ax.tripcolor(triang.x, triang.y, triang.triangles, solution[:, 0], cmap='seismic', vmin=vmin, vmax=vmax, shading='gouraud')
    cbar = fig.colorbar(tripcolor_plot, label='Displacement u')
    ax.set_title(f'Wave Propagation, t = {Gt[0]:.2f} s')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', 'box')

    def update(frame):
        ax.set_title(f'Wave Propagation, t = {Gt[frame]:.2f} s')
        tripcolor_plot.set_array(solution[:, frame])
        return tripcolor_plot,

    # Create the animation. We only use every 10th frame to make it faster.
    animation_frames = range(0, len(Gt), 10)
    ani = animation.FuncAnimation(fig, update, frames=animation_frames, blit=True, interval=20)
    plt.show()
