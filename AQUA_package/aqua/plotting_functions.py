 # Some functions to make it easier to plot the 
# membrane variables agains each other

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from scipy.signal import find_peaks, peak_widths


def plot_ISI_dist(spikes, bins = 50, range = (0, 300), fig = None, ax = None):
    """
    Plots the ISI distribution from the spike time data.

    params:
        spikes:     (N_spikes, ) array of spike times

    """

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (8, 8))

    ISI = np.ediff1d(spikes)

    ax.hist(ISI, bins, range)

    ax.set_xlabel("Interspike Interval (ISI) [ms]")
    ax.set_ylabel("Counts")

    return fig, ax


def plot_membrane_variables(X, T, split = []):
    # X has shape: 3 x N_iter
    # T has shape: N_iter
    # split: describes the sub_range of values to zoom into.
    if len(split) == 0:
        split = range(len(T))
    fig, ax = plt.subplots(3, 1, figsize = (15, 5), sharex = 'all')
    colors = ['r', 'g', 'b']
    labels = ['v', 'u', 'w']
    for i in range(3):
        ax[i].plot(T[split], X[i, split], c = colors[i])
        #ax[i].title.set_text(labels[i])
        #ax[i].set_xlabel("Time [ms]")
        ax[i].set_ylabel(labels[i])
    ax[-1].set_xlabel("Time [ms]")
    return fig, ax


def plot_potential_versus_injected(X, T, I_inj, split = [], fig = None, ax = None, label = None):
    """
    Draws the membrane potential and the injected current against time
    params:
        X:          (3, N_iter) array from simulation
        T:          (N_iter, ) time array
        I_inj:      (N_iter, ) injected current
        split:      continuous array of indices to plot
        fig, ax:    pyplot fig and ax objects
        label:      line label
    """

    if len(split) == 0:
        split = range(len(T))

    if fig is None or ax is None:
        fig, ax = plt.subplots(2, 1, figsize = (8, 8))
    #fig.tight_layout()

    ax[0].plot(T[split], I_inj[split], c = 'r', label = label)
    #ax[0].title.set_text('injected current')
    ax[0].set_ylabel('Injected current [pA]')

    ax[1].plot(T[split], X[0, split], label = label)
    #ax[1].title.set_text('membrane response')
    ax[1].set_ylabel('membrane \n potential [mV]')
    ax[1].set_xlabel('Time [ms]')

    return fig, ax


def time_dependent_frequency(spikes):
    #spike times are in ms
    freq = 1000/np.ediff1d(spikes)
    fig, ax = plt.subplots(1, 1, figsize = (10, 10))

    ax.plot(spikes[:-1], freq)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_ylim(0, 1.1*np.max(freq))

    return fig, ax


def compare_trains(X1, X2, T, I, indices = []):
    # X1 and X2 are membrane variable values
    # T is just the time
    # I is the injected current
    # range = [low, high] is the range for plotting to zoom in.
    
    #plot
    if indices == None:
        indices = range(0, len(T))

    fig, ax = plt.subplots(3, 1, figsize = (10, 5))
    fig.tight_layout()

    ax[0].plot(T, I, c = 'r')
    ax[0].title.set_text('injected current')
    ax[0].set_ylabel('injected current [pA]')

    ax[1].plot(T, X1, c = 'orange')
    ax[1].plot(T, X2, c = 'blue')
    ax[1].set_ylabel('v [mV]')

    ax[2].plot(T[indices], X1[indices], c = 'orange', label = 'no autapse')
    ax[2].plot(T[indices], X2[indices], c = 'blue', label = 'autapse')
    ax[2].set_xlabel('Time [s]')
    ax[2].set_ylabel('v [mV]')
    ax[2].legend()

    return fig, ax


def plot_raster(spike_array, total_time, ax=None, **kwargs):
    """
    Plots a raster plot onto a given or new axes object.
    
    Parameters:
    - spike_array: 2D numpy array (neurons x time_steps)
    - total_time: Total duration
    - ax: (Optional) Existing matplotlib axes object
    - **kwargs: Pass-through arguments for line styling (e.g., color, linewidth)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    num_trains = spike_array.shape[0]
    
    # Set default styles but allow override via kwargs
    color = kwargs.get('color', 'black')
    linewidth = kwargs.get('linewidth', 1.5)

    for i in range(num_trains):
        spike_indices = np.where(spike_array[i, :] > 0)[0]
        spike_times = spike_array[i, spike_indices]
        
        # Plotting the spikes
        ax.vlines(spike_times, i, i + 0.8, color=color, linewidth=linewidth)

    ax.set_xlim(0, total_time)
    ax.set_ylim(-0.5, num_trains)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron Index")
    
    return fig, ax



def plot_ISI_w_peaks(spike_times, bins = 50, x_range = (0, 100), fig = None, ax = None):
    # --- 1. Calculate ISIs ---
    # Assuming spike_times is your 1D array of spike timestamps
    isis = np.diff(spike_times)

    # --- 2. Create Histogram ---
    # You can adjust 'bins' to change the resolution of your analysis
    counts, bin_edges = np.histogram(isis, bins = bins, range = x_range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # --- 3. Detect Peaks and Widths ---
    # height: minimum count to be considered a peak
    # distance: minimum number of bins between peaks
    peaks, properties = find_peaks(counts, height=np.max(counts)*0.5, distance=0.1*bins)
    results_half = peak_widths(counts, peaks, rel_height=0.5)

    # Mapping indices back to time units for plotting
    def idx_to_val(idx):
        return np.interp(idx, np.arange(len(bin_centers)), bin_centers)

    peak_times = bin_centers[peaks]
    left_vals = idx_to_val(results_half[2])
    right_vals = idx_to_val(results_half[3])
    width_heights = results_half[1]

    # --- 4. Plotting ---
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (4, 4))
    
    ax.hist(isis, bins=50, range = x_range, color='skyblue', edgecolor='black', alpha=0.7, label='ISI Histogram')

    # Add vertical lines at peaks
    for pt in peak_times:
        ax.axvline(x=pt, color='red', linestyle='--', linewidth=2, label='Peak' if pt == peak_times[0] else "")

    # Add horizontal double-headed arrows for widths
    for i in range(len(peaks)):
        y = width_heights[i]
        ax.annotate('', xy=(left_vals[i], y), xytext=(right_vals[i], y),
                    arrowprops=dict(arrowstyle='<->', color='green', lw=2, shrinkA=0, shrinkB=0))
        ax.text((left_vals[i] + right_vals[i])/2, y, f' {right_vals[i]-left_vals[i]:.3f} ms', 
                ha='center', va='bottom', color='green', fontweight='bold')

    ax.set_xlabel('Interspike Interval (ms)')
    ax.set_ylabel('Frequency')
    ax.set_xlim(x_range)


def plot_first_return_map(spike_times, burn_in=0, ax=None, **kwargs):
    """
    Produces a first return map of Inter-Spike Intervals (ISIs).
    
    Parameters:
    - spike_times: List or array of timestamps.
    - burn_in: Time threshold to ignore initial transients.
    - ax: Optional matplotlib axes object to plot onto.
    - **kwargs: Passed to ax.scatter (e.g., color, s, alpha).
    """
    # 1. Convert to numpy array for efficient filtering
    spikes = np.asarray(spike_times)
    steady_spikes = spikes[spikes >= burn_in]
    
    if len(steady_spikes) < 3:
        print("Insufficient spikes for a return map.")
        return ax

    # 2. Calculate ISIs and Return Pairs
    isis = np.diff(steady_spikes)
    x = isis[:-1]
    y = isis[1:]
    
    # 3. Handle Axes logic
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # 4. Plotting
    # Default styling if not provided in kwargs
    kwargs.setdefault('c', 'teal')
    kwargs.setdefault('alpha', 0.6)
    kwargs.setdefault('s', 20)
    
    ax.scatter(x, y, **kwargs)
    
    # Add identity line
    max_val = max(np.max(x), np.max(y))
    ax.plot([0, max_val], [0, max_val], color='black', 
            linestyle='--', alpha=0.3, label='ISI[n] = ISI[n+1]')
    
    ax.set_xlabel('ISI$_{n}$ (ms)')
    ax.set_ylabel('ISI$_{n+1}$ (ms)')
    ax.set_title("First Return Map")
    ax.grid(True, alpha=0.3)
    
    return ax



def plot_3D(X, split):
    # Plots a 2x2 figure with a 3D phase plot, and 3 2D projections.
    fig = plt.figure(figsize = (10, 10))
    ax1 = fig.add_subplot(221, projection = '3d')
    #ax2 = fig.add_subplot(122, projection = '3d')
    ax1.plot(X[0, split], X[1, split], X[2, split], color = 'b')
    ax1.set_xlabel('v')
    ax1.set_ylabel('u')
    ax1.set_zlabel('w')
    ax1.set_title('3D phase plot')
    #ax2.plot(X_autapse[0], X_autapse[1], X_autapse[2], color = 'b')

    ax2 = fig.add_subplot(222)
    ax2.plot(X[1, split], X[2, split])
    ax2.set_xlabel('u')
    ax2.set_ylabel('w')
    ax2.set_title('U-W projection')

    ax3 = fig.add_subplot(223)
    ax3.plot(X[0, split], X[1, split])
    ax3.set_xlabel('v')
    ax3.set_ylabel('u')
    ax3.set_title('V-U projection')

    ax4 = fig.add_subplot(224)
    ax4.plot(X[0, split], X[2, split])
    ax4.set_xlabel('v')
    ax4.set_ylabel('w')
    ax4.set_title('V-W projection')

    fig.tight_layout(pad = 2.0)
    return fig


def plot_3D_gradient(X, split):
    # Plots a 2x2 figure with a 3D phase plot, and 3 2D projections.
    time = np.linspace(0, len(split), len(split))
    cmap = 'plasma'

    fig = plt.figure(figsize = (8, 8))
    ax1 = fig.add_subplot(221, projection = '3d')
    #ax2 = fig.add_subplot(122, projection = '3d')

    # find max and min of each dimension for plotting
    minV = np.min(X[0, split]) - abs(0.1*np.min(X[0, split]))
    maxV = np.max(X[0, split]) + abs(0.1*np.max(X[0, split]))
    minU = np.min(X[1, split]) - abs(0.1*np.min(X[1, split]))
    maxU = np.max(X[1, split]) + abs(0.1*np.max(X[1, split]))
    minW = np.min(X[2, split]) - abs(0.1*np.min(X[2, split]))
    maxW = np.max(X[2, split]) + abs(0.1*np.max(X[2, split]))


    # For the 3D plot
    pointsUVW = np.array([X[0, split], X[1, split], X[2, split]]).T.reshape(-1, 1, 3)
    segmentsUVW = np.concatenate([pointsUVW[:-1], pointsUVW[1:]], axis = 1)
    lineCollect3D = Line3DCollection(segmentsUVW, array=time, cmap = cmap, linewidths = 2)

    ax1.add_collection(lineCollect3D)
    ax1.set_xlabel('v')
    ax1.set_ylabel('u')
    ax1.set_zlabel('w')
    ax1.set_title('3D phase plot')
    ax1.set_xlim(minV, maxV)
    ax1.set_ylim(minU, maxU)
    ax1.set_zlim(minW, maxW)


    # For the 2D projections - UW
    pointsUW = np.array([X[1, split], X[2, split]]).T.reshape(-1, 1, 2)
    segmentsUW = np.concatenate([pointsUW[:-1], pointsUW[1:]], axis = 1)
    collectUW = LineCollection(segmentsUW, array = time, cmap = cmap, linewidth = 2)

    ax2 = fig.add_subplot(222)
    ax2.add_collection(collectUW)
    ax2.set_xlabel('u')
    ax2.set_ylabel('w')
    ax2.set_xlim(minU, maxU)
    ax2.set_ylim(minW, maxW)
    ax2.set_title('U-W projection')


    #For the 2D projection - UV
    pointsVU = np.array([X[0, split], X[1, split]]).T.reshape(-1, 1, 2)
    segmentsVU = np.concatenate([pointsVU[:-1], pointsVU[1:]], axis = 1)
    collectVU = LineCollection(segmentsVU, array = time, cmap = cmap, linewidth = 2)

    ax3 = fig.add_subplot(223)
    ax3.add_collection(collectVU)
    ax3.set_xlabel('v')
    ax3.set_ylabel('u')
    ax3.set_xlim(minV, maxV)
    ax3.set_ylim(minU, maxU)
    ax3.set_title('V-U projection')


    #For the 2D projection - VW
    pointsVW = np.array([X[0, split], X[2, split]]).T.reshape(-1, 1, 2)
    segmentsVW = np.concatenate([pointsVW[:-1], pointsVW[1:]], axis = 1)
    collectVW = LineCollection(segmentsVW, array = time, cmap = cmap, linewidth = 2)

    ax4 = fig.add_subplot(224)
    ax4.add_collection(collectVW)
    ax4.set_xlabel('v')
    ax4.set_ylabel('w')
    ax4.set_xlim(minV, maxV)
    ax4.set_ylim(minW, maxW)
    ax4.set_title('V-W projection')

    fig.tight_layout(pad = 2.0)
    return fig


def plot_time_gradient(X, T, split):
    # returns a line collection which can be plotted 
    # as a trajectory with time as the color
    time = np.linspace(0, len(split), len(split))
    cmap = 'plasma'

    
    
    points = np.array([X[0, split], X[1, split]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis = 1)
    collect = LineCollection(segments, array = time, cmap = cmap, linewidth = 2)


    return collect


def plot_VUtime(X, T, split, I, neuron, N_dim):
    """
    X is the time series from updating the neuron
    T is the corresponding times
    split represents specific times in the simulation to plot
    I is the height of the input
    
    
    """
    # Plots a 2x2 figure with a 3D phase plot, and 3 2D projections.
    fig = plt.figure(figsize = (10, 10))
    ax1 = fig.add_subplot(111, projection = '3d')

    # min and max values
    minV = np.min(X[0, split])
    maxV = np.max(X[0, split])
    minT = np.min(T[split])
    maxT = np.max(T[split])

    # plot the trajectory
    ax1.plot(X[0, split], T[split], X[1, split], color = 'black') #u on z-axis
    ax1.set_xlabel('v [mV]')
    ax1.set_ylabel('time')
    ax1.set_zlabel('u')
    ax1.set_title('V-U phase plot through time. ')
    #set axis limits
    ax1.set_zlim(-50, 50)

    #nullcline variables
    v = np.linspace(minV, maxV, N_dim)
    t = np.linspace(minT, maxT, N_dim)
    w = np.roll(X[2, split], neuron.tau)[::N_dim]
    V_grid, W_grid = np.meshgrid(v, w) # the w array is time-dependent and will be plotted on the time axis.
    
    # I_grid must resemble the W_grid and vary along the same axis.
    I_inj = I[split][::N_dim]
    I_inj = np.reshape(I_inj, (len(I_inj), 1))
    I_grid = np.tile(I_inj, N_dim)

    # T_grid must have the same shape as W_grid
    T_inj = T[split][::N_dim]
    T_inj = np.reshape(T_inj, (len(T_inj), 1))
    T_grid = np.tile(T_inj, N_dim)
    
    
    # v-nullcline is time-dependent
    v_null = 0.04*V_grid**2 + 5*V_grid + 140 + I_grid + W_grid   # represents the height of the surface
    v_null[np.where(v_null > 50)] = np.nan
    u_null = neuron.b*V_grid

    surf1 = ax1.plot_surface(V_grid, T_grid, v_null, cmap = 'coolwarm', antialiased = False, linewidth = 0, alpha = 0.5)
    surf2 = ax1.plot_surface(V_grid, T_grid, u_null, cmap = 'PuBuGn', antialiased = False, linewidth = 0, alpha = 0.5) 
    return fig


def plot_bifurcation_I(spikes, I_range, steady_state = True, fig = None, ax = None):
    """
    Plots the steady-state or instantaneous ISIs versus injected current. 
    
    INPUT:
        spikes:         ndarray of spike times
                        each row is a different simulation
        I_range:        array
                        injected currents
        steady_state:   boolean
                        True if steady state ISIs, otherwise instantaneous
    
    OUTPUT:
        fig, ax:        matplotlib obj
    
    """

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (10, 10))
    
    for i in range(np.shape(spikes)[0]):
        spike_times = spikes[i, ~np.isnan(spikes[i])] # get row and remove nan values
        if steady_state:
            subSpikes = spike_times[-int(0.5*len(spike_times)):] # last half of spikes
        else: # if instantaneous is desired
            subSpikes = spike_times[:int(0.5*len(spike_times))]  # first half of spikes
        
        ISI = np.diff(subSpikes)
        isi_vals, isi_counts = np.unique(np.round(ISI, decimals = 4), return_counts = True)
        ax.scatter(I_range[i]*np.ones(np.shape(isi_vals)[0]), isi_vals, c = 'black', s = 0.8, marker = "o")

    return fig, ax

"""
def animate_train(X, T, I, out_dir):
    # Create an animation for the injected current and output membrane variables
"""