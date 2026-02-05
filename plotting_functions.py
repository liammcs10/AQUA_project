 # Some functions to make it easier to plot the 
# membrane variables agains each other

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from batchAQUA_general import *



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


def plot_potential_versus_injected(X, T, I_inj, split = []):

    if len(split) == 0:
        split = range(len(T))

    fig, ax = plt.subplots(2, 1, figsize = (8, 8))
    #fig.tight_layout()

    ax[0].plot(T[split], I_inj[split], c = 'r')
    #ax[0].title.set_text('injected current')
    ax[0].set_ylabel('Injected current [pA]')

    ax[1].plot(T[split], X[0, split], c = 'blue')
    #ax[1].title.set_text('membrane response')
    ax[1].set_ylabel('membrane \n potential [mV]')
    ax[1].set_xlabel('Time [ms]')

    return fig, ax

def plot_variables_versus_injected(X, T, I_inj, split = []):

    if len(split) == 0:
        split = range(len(T))
    
    fig, ax = plt.subplots(4, 1, figsize = (15, 10), sharex = 'all')

    # order: I_inj, V, U, W
    colors = ['red', 'black', 'green', 'blue']
    labels = ['I', 'v', 'u', 'w']
    for i in range(4):
        if i == 0:
            ax[i].plot(T[split], I_inj[split], c = colors[i])
        else:
            ax[i].plot(T[split], X[i-1, split], c = colors[i])
        #ax[i].title.set_text(labels[i])
        #ax[i].set_xlabel("Time [ms]")
        ax[i].set_ylabel(labels[i])
    
    ax[-1].set_xlabel("Time [ms]")
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
    # X1 and X2 neuron variables values
    # T is just the time
    # I is the injected current
    # range = np.arange(low, high] is the range for plotting to zoom in.
    
    #plot
    if len(indices) == 0:
        indices = range(0, len(T))

    fig, ax = plt.subplots(3, 1, figsize = (10, 5))
    fig.tight_layout()

    ax[0].plot(T, I, c = 'r')
    ax[0].title.set_text('injected current')
    ax[0].set_ylabel('injected current [pA]')

    ax[1].plot(T, X1[0, :], c = 'orange')
    ax[1].plot(T, X2[0, :], c = 'blue')
    ax[1].set_ylabel('v [mV]')

    ax[2].plot(T[indices], X1[0, indices], c = 'orange', label = 'no autapse')
    ax[2].plot(T[indices], X2[0, indices], c = 'blue', label = 'autapse')
    ax[2].set_xlabel('Time [s]')
    ax[2].set_ylabel('v [mV]')
    ax[2].legend()

    return fig, ax


def plot_bifurcation_I(spikes, I_range, steady_state = True):
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

    ax.set_title("Bifurcation diagram")
    ax.set_ylabel("ISIs [ms]")
    ax.set_xlabel("Driving current [pA]")
    return fig, ax

def plot_bifurcation_f(spikes, f_range, steady_state = True):
    """
    Plots the steady-state or instantaneous ISIs versus injected current. 
    
    INPUT:
        spikes:         ndarray of spike times
                        each row is a different simulation
        f_range:        array
                        frequency of stimulus
        steady_state:   boolean
                        True if steady state ISIs, otherwise instantaneous
    
    OUTPUT:
        fig, ax:        matplotlib obj
    
    """
    fig, ax = plt.subplots(1, 1, figsize = (10, 10))
    for i in range(np.shape(spikes)[0]):
        spike_times = spikes[i, ~np.isnan(spikes[i])] # get row and remove nan values
        if steady_state:
            subSpikes = spike_times[-int(0.5*len(spike_times)):] # last half of spikes
        else: # if instantaneous desired
            subSpikes = spike_times[:int(0.5*len(spike_times))]  # first half of spikes
        
        ISI = np.diff(subSpikes)
        isi_vals, isi_counts = np.unique(np.round(ISI, decimals = 4), return_counts = True)
        ax.scatter(f_range[i]*np.ones(np.shape(isi_vals)[0]), isi_vals, c = 'black', s = 0.8, marker = "o")

    ax.set_title("Bifurcation diagram")
    ax.set_ylabel("ISIs [ms]")
    ax.set_xlabel("Driving frequency [Hz]")
    return fig, ax

def first_return(spikes, fig, ax):
    """
    Returns a figure of the first return map for a spike train

    IN:
        spikes:     time of spikes in ms (1 neuron only)
    
    """

    #fig, ax = plt.subplots(1, 1, figsize = (5, 5))

    ISI = np.diff(spikes)
    ax.scatter(ISI[:-1], ISI[1:], c = 'black', s = 1.5, marker = 'o')

    ax.set_title("First Return Map")
    ax.set_xlabel("ISI_t")
    ax.set_ylabel("ISI_t+1")

    return fig, ax


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
    minU = np.min(X[1, split]) - abs(0.2*np.min(X[1, split]))
    maxU = np.max(X[1, split]) + abs(0.1*np.max(X[1, split]))
    minW = np.min(X[2, split]) - abs(0.2*np.min(X[2, split]))
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
    Plots the trajectory in the VU-plane with nullclines.

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


def plot_phase_portrait_UV(X, T, spikes, I, params, time = None, split = None, quiver = True, fig = None, ax = None):
    """
    Plots the phase portrait of the AQUA neuron in the V-U plane. 
    Shown are the nullclines (which may be dynamic in the autaptic case), labelled trajectories between spikes,
    and labels for the initiation of the autaptic current.
    
    Params:
        X:          ndarray - (3, N_iter)
                    Array of neuron variables through time.
        T:          array - (N_iter)
                    time array
        spikes:     array: (N_spikes)
                    array of spike times
        I:          arr - (N_iter)
                    injected current
        split:      time period to plot
        optional fig and ax
    
    """
    dt = T[1] - T[0]        # simulation timestep


    # check if figure and axis is passed.
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (8, 8))

    # check if split is passed
    if split is None:
        split = np.arange(0, len(T))

    # get min and max values of each variable
    minV = np.min(X[0, split])
    maxV = np.max(X[0, split])
    minU = np.min(X[1, split])
    maxU = np.max(X[1, split])

    # split the trajectory
    split_time = split[-1] * dt                     # end time of split
    spike_times = spikes[spikes < split_time]       # get spike times within split
    
    # section the trajectory...
    time_indices = [int(spike_time/dt) for spike_time in spike_times]
    time_indices.insert(0, 0.)
    time_indices.append(split[-1])  # end of the trial
    print(time_indices)


    v_sections = []
    u_sections = []
    w_sections = []
    prev_idx = 0
    for n, idx in enumerate(time_indices[1:]):
        v_sections.append(list(X[0, prev_idx:idx]))
        u_sections.append(list(X[1, prev_idx:idx]))
        w_sections.append(list(X[2, prev_idx:idx]))
        prev_idx = idx + 1

    """
    v_sections = pad_list(v_sections)
    u_sections = pad_list(u_sections)
    w_sections = pad_list(w_sections)
    """
    N_sections = len(v_sections)

    # plot the u-nullcline, static
    v = np.linspace(minV-10, maxV, 500)    # v-coords for plotting
    u_null = params['b'] * (v - params['v_r'])
    ax.plot(v, u_null, color = 'black', label = 'u-nullcline')

    # autapse delay in timesteps
    aut_timestep = int(params['tau']/dt)

    # plot trajectories and nullclines
    traj_colors = plt.get_cmap("Blues")(np.linspace(0.2, 1, N_sections))
    v_null_colors = plt.get_cmap("Reds")(np.linspace(0.2, 1, N_sections))
    v_null_star_colors = plt.get_cmap("Purples")(np.linspace(0.2, 1, N_sections))
    v_null_starstar_colors = plt.get_cmap("Greens")(np.linspace(0.2, 1, N_sections))

    for n in range(N_sections):
        # plot trajectory for that section
        ax.plot(v_sections[n], u_sections[n], color = traj_colors[n])

        # plot reference v-nullcline
        if n == 0:
            v_null_ref = params['k'] * (v - params['v_r']) * (v - params['v_t']) + I[time] # no autapse
            ax.plot(v, v_null_ref, color = v_null_colors[n], alpha = 0.5, label = f"reference nullcline")

        if n > 0:
            # plot v-nullcline
            v_null = params['k'] * (v - params['v_r']) * (v - params['v_t']) + I[time] + X[2, int(time_indices[n] + time - aut_timestep)]
            ax.plot(v, v_null, color = v_null_colors[n], label = f"v_nullcline time: {(time_indices[n] + time) * dt}")

            # plot circle at time, time
            ax.scatter(v_sections[n][time], u_sections[n][time], marker = 's', color = 'black', label = f"neuron state")

            # write spike number at the reset value
            ax.text(v_sections[n][0], u_sections[n][0], s = f"{n}", color = traj_colors[n])

        # plot the flow field
        if quiver:
            N_arr = 50
            x = np.linspace(minV-50, maxV+20, N_arr)
            y = np.linspace(minU-10, maxU+10, N_arr)
            X_pt, Y_pt = np.meshgrid(x, y)

            N_neurons = N_arr*N_arr
            params_list = []
            for i in range(N_neurons):
                params_list.append(params)
            
            batch = batchAQUA(params_list)
            x_start = np.zeros((N_neurons, 3))
            x_start[:, 0] = X_pt.flatten()
            x_start[:, 1] = Y_pt.flatten()
            x_start[:, 1] = X[2, time - aut_timestep]   # include effect of autapse
            t_start = np.zeros(N_neurons)
            
            batch.Initialise(x_start, t_start)

            grad = batch.neuron_model(batch.x, 0., I[time])
            U = grad[:, 0].reshape((N_arr, N_arr))
            V = grad[:, 1].reshape((N_arr, N_arr))
            ax.quiver(X_pt, Y_pt, U, V)

        """ plotting more than one nullcline...
        # print nullclines based on autapse values.
        if params['f'] != 0. and n != 0:
            autapse_delay = int(params['tau']/dt)
            autapse_decay = int((1./params['e'])/dt)
            # label where autapse activates
            ax.text(v_sections[n][autapse_delay], u_sections[n][autapse_delay], s = f"{n}*", color = traj_colors[n])
            v_null_star = params['k'] * (v - params['v_r']) * (v - params['v_t']) + I[time_indices[n]] + w_sections[n][0]
            ax.plot(v, v_null_star, color = v_null_star_colors[n], label = f"v_nullcline* {n+1}")

            # label autapse after the characteristic decay time
            ax.text(v_sections[n][autapse_delay + autapse_decay], u_sections[n][autapse_delay + autapse_decay], s = f"{n}**", color = traj_colors[n])
            v_null_starstar = params['k'] * (v - params['v_r']) * (v - params['v_t']) + I[time_indices[n]] + w_sections[n][autapse_decay]
            ax.plot(v, v_null_starstar, color = v_null_starstar_colors[n], label = f"v_nullcline** {n+1}")
        """
    ax.set(xlabel = "Membane Potential, v [mV]", ylabel = "Adaptation Current, u [pA]",
        ylim = (minU-50, maxU+20), xlim = (minV-10, maxV+10))
    

    return fig, ax
