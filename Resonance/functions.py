
import sys
sys.path.append("..\\")

import numpy as np
from batchAQUA_general import batchAQUA
from AQUA_general import AQUA
from stimulus import *
import matplotlib.pyplot as plt
from plotting_functions import *




def find_threshold(neuron_params, I_list, T, dt):
    """
    Estimates the threshold current for spiking for a given neuron by first
    finding a rough approximation and then refining it. Requires fewer trials.
    IN:
        neuron:     dict
                    batchAQUA neuron params

        I_list:     array
                    list of step heights to consider
    OUT:
        threshold   float
                    just below threshold current for spiking
    
    """

    N_iter = int(1000*T/dt) # number of iterations
    N_neurons = np.shape(I_list)[0]

    I_inj = np.array([i*np.ones(N_iter) for i in I_list])

    # batch list of params
    params = [neuron_params for n in range(N_neurons)]

    x_start = np.full((N_neurons, 3), fill_value = np.array([neuron_params["c"], 0., 0.]))
    t_start = np.zeros(N_neurons)

    neurons = batchAQUA(params)
    neurons.Initialise(x_start, t_start)

    _, _, spikes = neurons.update_batch(dt, N_iter, I_inj)

    idx_threshold = np.argwhere(np.isnan(spikes[:, 0]).flatten())[-1]

    sub_threshold = I_list[idx_threshold]
    above_threshold = I_list[idx_threshold + 1]

    #redefine I_inj
    I_list_thresh = np.linspace(sub_threshold, above_threshold, N_neurons)
    I_inj_thresh = np.array([i*np.ones(N_iter) for i in I_list_thresh])

    neurons.Initialise(x_start, t_start)
    X, T, spikes = neurons.update_batch(dt, N_iter, I_inj_thresh)

    idx_threshold = np.argwhere(np.isnan(spikes[:, 0]).flatten())[-1]

    threshold = I_list_thresh[idx_threshold][0] # closer estimate of the threshold
    steady_state = X[idx_threshold, :, -1][0]

    return threshold, steady_state

def find_pulse_height(neuron_params, I_list, threshold, x_ini, pulse_duration):
    """
    Finds the minimum pulse height of a given duration to elicit a spike at 
    some baseline current intensity.

    IN:
        neuron_params:      dict
                            AQUA neuron parameters
        I_list:             array
                            pulse height values
        threshold:          float
                            baseline current value
        pulse_durations:    float
                            duration of the pulse in ms
    
    OUT:
        pulse_height:       float
                            height of the pulse in pA
        spike_time:         float
                            time of the spike in ms (relative to end of pulse)
    
    """

    T = 0.5 # s
    dt = 0.01 # ms
    N_iter = int(1000*T/dt) # number of iterations

    N_neurons = np.shape(I_list)[0]

    delay = 100

    I_inj = np.array([spikes_constant(N_iter, dt, threshold, 0.0, 1, i, pulse_duration, delay) for i in I_list])

    # batch list of params
    params = [neuron_params for n in range(N_neurons)]

    x_start = np.full((N_neurons, 3), fill_value = x_ini)
    t_start = np.zeros(N_neurons)

    neurons = batchAQUA(params)
    neurons.Initialise(x_start, t_start)

    _, _, spikes = neurons.update_batch(dt, N_iter, I_inj)
    
    idx_low = np.argwhere(np.isnan(spikes[:, 0]).flatten())[-1] # last pulse to not produce a spike
    idx_high = np.argwhere(~np.isnan(spikes[:, 0].flatten()))[0] # first pulse to produce a spike

    pulse_height_low = I_list[idx_low]          # lower bound estimate
    pulse_height_high = I_list[idx_high]        # upper bound estimate


    # refine the search
    I_list_refine = np.linspace(pulse_height_low, pulse_height_high, N_neurons)
    I_inj_refine = np.array([spikes_constant(N_iter, dt, threshold, 0.0, 1, i, pulse_duration, delay) for i in I_list_refine])

    # re-initialise the neurons
    neurons.Initialise(x_start, t_start)
    _, _, spikes = neurons.update_batch(dt, N_iter, I_inj_refine)

    idx_threshold = np.argwhere(~np.isnan(spikes[:, 0]).flatten())[0]   # first pulse to make spike
    idx_2 = np.argwhere(np.isnan(spikes[:, 0].flatten()))[-1]   # last pulse to not make spike.
    pulse_height = I_list_refine[idx_threshold][0]      # close estimate of the pulse height
    pulse_height2 = I_list_refine[idx_2][0]             # the next lowest pulse which didn't produce a spike.


    pulse_end = delay + pulse_duration - dt
    spike_time = spikes[idx_threshold, 0] - pulse_end
    
    return pulse_height, pulse_height2, spike_time

def find_2nd_pulse_height(neuron_params, I_list, threshold, x_ini, pulse_duration, first_pulse_height, time_to_spike, resonant_ISI):
    """
    Finds the minimum pulse height of the 2nd pulse to induce a spike given it arrives 
    at the neuron's resonance frequency.
    
    IN:
        neuron_params:      dict
                            AQUA neuron parameters

        I_list:             array
                            pulse height values

        threshold:          float
                            baseline current value

        x_ini:              array (3, )
                            initial conditions

        pulse_durations:    float
                            duration of the pulse in ms

        first_pulse_height: float
                            height of pulse to enduce a spike

        time_to_spike:      float
                            time of spike relative to end of the first pulse

        resonant_ISI:       float
                            time of 2nd pulse start relative to the spike to be resonant
    OUT:
        pulse_height:       float
                            height of the pulse in pA

    """
    T = 0.5 # s
    dt = 0.01 # ms
    N_iter = int(1000*T/dt) # number of iterations

    N_neurons = np.shape(I_list)[0]

    delay = 100 # ms

    # I_inj = np.array([spikes_constant(N_iter, dt, threshold, 0.0, 1, i, pulse_duration, delay) for i in I_list])
    I_inj = threshold*np.ones((N_neurons, N_iter))
    pulse1_start = int(delay/dt)
    pulse1_end = int((delay+pulse_duration)/dt)
    I_inj[:, pulse1_start:pulse1_end] += first_pulse_height # create the first pulse

    pulse2_start = pulse1_end + int((time_to_spike + resonant_ISI)/dt)
    pulse2_end = pulse2_start + int(pulse_duration/dt)

    for n, i in enumerate(I_list):      # define the second pulse as 
        I_inj[n, pulse2_start:pulse2_end] += i  


    # batch list of params
    params = [neuron_params for n in range(N_neurons)]

    x_start = np.full((N_neurons, 3), fill_value = x_ini)
    t_start = np.zeros(N_neurons)

    neurons = batchAQUA(params)
    neurons.Initialise(x_start, t_start)

    _, _, spikes = neurons.update_batch(dt, N_iter, I_inj)

    idx_low = np.argwhere(np.isnan(spikes[:, 1]).flatten())[-1]     # last pulse no spike
    idx_high = np.argwhere(~np.isnan(spikes[:, 1].flatten()))[0]    # first pulse to spike
    pulse_height_low = I_list[idx_low]            # first estimate of the pulse height
    pulse_height_high = I_list[idx_high]          # second estimate of the pulse height

    """ FOR more accurate pulse height...
    # refine the search
    I_list_refine = np.linspace(pulse_height_low, pulse_height_high, N_neurons)
    I_inj_refine = threshold*np.ones((N_neurons, N_iter))
    I_inj_refine[:, pulse1_start:pulse1_end] += first_pulse_height
    for n, i in enumerate(I_list_refine):      # define the second pulse as 
        I_inj_refine[n, pulse2_start:pulse2_end] += i
    
    # re-initialise the neurons
    neurons.Initialise(x_start, t_start)
    _, _, spikes = neurons.update_batch(dt, N_iter, I_inj_refine)

    idx_threshold = np.argwhere(~np.isnan(spikes[:, 1]).flatten())[0]   #
    pulse_height = I_list_refine[idx_threshold][0]     # close estimate of the pulse height
    """
    return pulse_height_high


def get_resonance_bands(resonant_f, spike_boolean):
    """
    Returns the resonance bands given the frequencies tested and number of spikes produced
    
    """
    spike_boolean = spike_boolean.astype(int)

    band_limits = np.diff(spike_boolean, prepend = 0, append = 0)

    band_starts = np.where(band_limits == -1)[0]-1
    band_ends = np.where(band_limits == 1)[0]

    frequency_bands = np.zeros((len(band_starts), 2))

    for n in range(len(band_starts)): # band_starts and ends should be the same length
        frequency_bands[n] = np.array([resonant_f[band_starts[n]], resonant_f[band_ends[n]]])
    
    return frequency_bands

#def hazems_pulses(pulse_height, pulse_durations, pulse_ISI, frequency):




"""- - - - PLOTTING FUNCTIONS - - - - """

def plot_resonance_map(frequencies, bands, fig = None, ax = None, title = 'Resonance Map of neurons'):
    """
    Returns a heat map of the resonance frequencies of the neurons

    IN:
        frequencies:    array (N_neurons, N_frequencies)
                        list of frequencies tested

        bands:          ndarray (N_neurons, N_frequencies)
                        resonance bands for each neuron
    
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (8, 8))

    unique_vals = np.unique(bands)
    print(unique_vals)
    num_vals = len(unique_vals)
    base_cmap = plt.cm.get_cmap("Greys", num_vals)
    im = ax.imshow(bands, origin = 'lower', cmap = base_cmap, aspect = 'auto', extent = [frequencies[0], frequencies[-1], 0, np.shape(bands)[0]])
    cbar = fig.colorbar(im, ax = ax, shrink = 0.5)
    cbar.set_ticks(np.linspace(np.min(unique_vals), np.max(unique_vals), num_vals))
    cbar.set_ticklabels(unique_vals)

    ax.set_xlabel('Frequency (Hz)', fontsize = 16)
    ax.set_ylabel('Neuron index', fontsize = 16)

    # define grid
    ax.set_xticks(np.arange(0, np.shape(bands)[1], 5), minor = True)
    ax.set_xticks(np.arange(0, np.shape(bands)[1], 25))
    ax.set_yticks(np.arange(0, np.shape(bands)[0], 1), minor = True)
    ax.set_yticks(np.arange(0, np.shape(bands)[0], 5))
    ax.grid(alpha = 0.4, which = 'minor')
    ax.grid(alpha = 0.8, which = 'major')

    ax.set_title(title, fontsize = 20)

    return fig, ax, im

def plot_resonance_for_autapse_param(param_df, frequencies, bands, autapse_params, fig = None, ax = None, title = 'Resonance Map of neurons ordered by'):
    """
    Plots the resonance properties of neurons for a given autapse parameter.
    Y label only shows the first parameter value sorting the list.

    IN:
        parameters:     DataFrame
                        dataframe of all neuron parameters tested

        frequencies:    array (N_neurons, N_frequencies)
                        list of frequencies tested

        bands:          ndarray (N_neurons, N_frequencies)
                        resonance bands for each neuron

        autapse_param:  list
                        names of the autapse params in order to sort the frequency bands.
    """

    # Gets the neuron params
    param_values = param_df[autapse_params[0]].values
    unique_params = np.unique(param_values)
    
    num_neurons_with_param = np.zeros(len(unique_params))
    for n, val in enumerate(unique_params):
        num_neurons_with_param[n] = np.sum(param_values == val)
    cumsum = np.cumsum(num_neurons_with_param)

    # order the autapse
    ordered_df = param_df.sort_values(by = autapse_params)
    sorted_indices = np.array(ordered_df.index.tolist(), dtype = int)

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (8, 8))

    unique_vals = np.unique(bands)
    num_vals = len(unique_vals)
    base_cmap = plt.cm.get_cmap("Greys", num_vals)
    im = ax.imshow(bands[sorted_indices, :], origin = 'lower', cmap = base_cmap, aspect = 'auto', extent = [frequencies[0], frequencies[-1], 0, len(param_values)])
    cbar = fig.colorbar(im, ax = ax, shrink = 0.5)
    cbar.set_ticks(np.linspace(np.min(unique_vals), np.max(unique_vals), num_vals))
    cbar.set_ticklabels(unique_vals)
    
    ax.set_xlabel('Frequency (Hz)', fontsize = 16)
    ax.set_ylabel(f'{autapse_params[0]} value', fontsize = 16)
    cumsum = np.insert(cumsum, 0, 0.)
    tick_loc = (cumsum[:-1] + cumsum[1:])/2.0
    ax.set_yticks(tick_loc, labels = np.round(unique_params, 2))

    xmin = 0.0
    xmax = np.max(frequencies)
    ax.set_xlim((xmin, xmax))
    ax.hlines(y = np.cumsum(num_neurons_with_param), xmin = xmin, xmax = xmax, colors = "grey", linestyles = "dashed")

    ax.set_title(f'{title} {autapse_params}', fontsize = 20)
    
    # define grid
    ax.set_xticks(np.arange(0, np.shape(bands)[1], 5), minor = True)
    ax.set_xticks(np.arange(0, np.shape(bands)[1], 25))
    ax.set_yticks(np.arange(0, np.shape(bands)[0], 1), minor = True)
    ax.set_yticks(np.arange(0, np.shape(bands)[0], 5))
    ax.grid(alpha = 0.4, which = 'minor')
    ax.grid(alpha = 0.8, which = 'major')

    return fig, ax, im

def plot_resonance_to_pulses(param_df, frequencies1, bands1, bands2, autapse_param = None, title = "Resonance by pulse number"):
    """
    Subtract test 1 resonance from 3 pulses resonance. Determines if pulse 3 caused the 2nd spike.
    1st axis plots where the second pulse produced a spike (basically just test 1)
    2nd axis plots where only the third pulse produced a spike (subtract test1 from 3_pulses)
    """
    
    fig, ax = plt.subplots(2, 1, figsize = (8, 8), constrained_layout = True)

    if autapse_param is None:
        fig, ax[0], _ = plot_resonance_map(frequencies1, bands1, fig, ax[0])
        fig, ax[1], _ = plot_resonance_map(frequencies1, bands2-bands1, fig, ax[1])
    else:
        fig, ax[0], _ = plot_resonance_for_autapse_param(param_df, frequencies1, bands1, autapse_param, fig, ax[0])
        fig, ax[1], _ = plot_resonance_for_autapse_param(param_df, frequencies1, bands2-bands1, autapse_param, fig, ax[1])
    

    fig.suptitle(title, fontsize = 22)
    ax[0].set_title("Resonance to second pulse only", fontsize = 16)
    ax[1].set_title("Resonance to third pulse only", fontsize = 16)

    return fig, ax
    # Now for third pulse



""" SIMULATION 2 PLOTS"""
# Plots of the spike to pulse ratio from simulation 2
def plot_spike_to_pulse_ratio(multipulse_freq, spike_to_pulse_ratio):
    """
    Plots the spike to pulse ratio for each neuron tested

    IN:
        spike_to_pulse_ratio:   array (N_neurons, N_isi)
                                ratio of spikes to pulses for each neuron at each isi
    """

    fig, ax = plt.subplots(1, 1, figsize = (8, 8))

    cax = ax.imshow(spike_to_pulse_ratio[:, :50], origin = 'lower', cmap = 'viridis', aspect = 'auto')
    fig.colorbar(cax, ax = ax, label = 'Spike to Pulse Ratio')

    ax.set_xlabel('Pulse frequency [Hz]', fontsize = 16)
    #ax.set_xticks(np.linspace(0, np.shape(spike_to_pulse_ratio)[1], 5), labels = np.round(multipulse_freq[np.linspace(0, np.shape(spike_to_pulse_ratio)[1]-1, 5, dtype = int)], 2))
    ax.set_ylabel('Neuron index', fontsize = 16)

    ax.set_title('Spike to Pulse Ratio for each neuron', fontsize = 20)

    return fig, ax

def plot_spike_to_pulse_ratio_for_param(parameters, multipulse_freq, spike_to_pulse_ratio, autapse_param):
    """
    Plots the spike to pulse ratio for each neuron tested

    IN:
        spike_to_pulse_ratio:   array (N_neurons, N_isi)
                                ratio of spikes to pulses for each neuron at each isi
    """

    fig, ax = plt.subplots(1, 1, figsize = (8, 8))

    param_values = parameters[autapse_param].values
    unique_params = np.unique(param_values)

    argsort_params = np.argsort(param_values)       # indices that would sort the array

    cax = ax.imshow(spike_to_pulse_ratio[argsort_params], origin = 'lower', cmap = 'viridis', aspect = 'auto', extent = [0, np.shape(spike_to_pulse_ratio)[1], 0, np.shape(spike_to_pulse_ratio)[0]])
    fig.colorbar(cax, ax = ax, label = 'Spike to Pulse Ratio')

    ax.set_xlabel('Pulse frequency [Hz]', fontsize = 16)
    #ax.set_xticks(np.linspace(0, np.shape(spike_to_pulse_ratio)[1], np.shape(multipulse_freq)[0]), labels = np.round(multipulse_freq, 2))
    ax.set_ylabel(f'{autapse_param} value', fontsize = 16)
    ax.set_yticks(np.linspace(np.min(param_values), np.max(param_values), len(unique_params)), labels = np.round(unique_params, 2))

    ax.set_title(f'Spike to Pulse Ratio for each neuron, sorted by the {autapse_param} parameter', fontsize = 20)

    return fig, ax

# Plots for the spike to pulse frequency from simulation 2
def plot_spike_to_pulse_frequency(multipulse_freq, spike_to_pulse_freq_ratio):
    """
    Plots the spike to pulse frequency for each neuron tested

    IN:
        spike_freq:     array (N_neurons, N_isi)
                        frequency of spikes for each neuron at each isi
    """

    fig, ax = plt.subplots(1, 1, figsize = (8, 8))

    cax = ax.imshow(spike_to_pulse_freq_ratio, origin = 'lower', cmap = 'viridis', aspect = 'auto', extent = [0, np.shape(spike_to_pulse_freq_ratio)[1], 0, np.shape(spike_to_pulse_freq_ratio)[0]])
    fig.colorbar(cax, ax = ax, label = 'Spike to Pulse Frequency Ratio')

    ax.set_xlabel('Pulse frequency [Hz]', fontsize = 16)
    #ax.set_xticks(np.linspace(0, np.shape(spike_to_pulse_freq_ratio)[1], np.shape(multipulse_freq)[0]), labels = np.round(multipulse_freq, 2))
    ax.set_ylabel('Neuron index', fontsize = 16)

    ax.set_title('Spike/Pulse Frequency for each neuron', fontsize = 20)

    return fig, ax

def plot_spike_to_pulse_frequency_for_param(parameters, multipulse_freq, spike_to_pulse_freq_ratio, autapse_param):
    """
    Plots the spike to pulse ratio for each neuron tested

    IN:
        spike_to_pulse_ratio:   array (N_neurons, N_isi)
                                ratio of spikes to pulses for each neuron at each isi
    """

    fig, ax = plt.subplots(1, 1, figsize = (8, 8))

    param_values = parameters[autapse_param].values
    unique_params = np.unique(param_values)

    argsort_params = np.argsort(param_values)       # indices that would sort the array

    cax = ax.imshow(spike_to_pulse_freq_ratio[argsort_params], origin = 'lower', cmap = 'viridis', aspect = 'auto', extent = [0, np.shape(spike_to_pulse_freq_ratio)[1], 0, np.shape(spike_to_pulse_freq_ratio)[0]])
    fig.colorbar(cax, ax = ax, label = 'Spike to Pulse Ratio')

    ax.set_xlabel('Pulse frequency [Hz]', fontsize = 16)
    #ax.set_xticks(np.linspace(0, np.shape(spike_to_pulse_freq_ratio)[1], np.shape(multipulse_freq)[0]), labels = np.round(multipulse_freq, 2))
    ax.set_ylabel(f'{autapse_param} value', fontsize = 16)
    ax.set_yticks(np.linspace(np.min(param_values), np.max(param_values), len(unique_params)), labels = np.round(unique_params, 2))

    ax.set_title(f'Spike to Pulse Ratio for each neuron, sorted by the {autapse_param} parameter', fontsize = 20)

    return fig, ax


"""
spike_times and pulse_starts can tell us about which pulses induce a spike.

can get:
    - how many pulses needed to produce the first spike?
    - After a spike, does the next pulse also produce a spike?

"""
def pulses_to_spike(spike_times, pulse_starts, multipulse_freq):
    """
    Plot the number of pulses at a specific frequency required to produce a spike.
    
    Args:
        spike_times:        ndarray (N_sims, N_spikes)
                            times for each spike.
        pulse_starts:       ndarray (N_freq, N_pulses)
                            pulse start times.
        multipulse_freq:    array (N_freq, )
                            corresponding frequency for each row of pulse_starts.
    Returns:
        fig, ax;            matplotlib objects
    """
    N_sims = int(np.shape(spike_times)[0])
    N_freq = int(np.shape(pulse_starts)[0])
    N_neurons = int(N_sims/N_freq)
    N_pulses = int(np.shape(pulse_starts)[1])

    multipulse_ISIs = 1000/multipulse_freq

    first_spike = spike_times[:, 0]
    first_spike = np.nan_to_num(first_spike, nan = np.max(pulse_starts) + 1)
    num_pulses_to_spike = np.zeros((N_neurons, N_freq), dtype = int)

    for n in range(N_neurons):
        comparison_arr = first_spike[n::N_neurons][:, np.newaxis] # sample each frequency from neuron n
        #print(pulse_starts[:, 0])
        #print(comparison_arr[:, 0])
        num_pulses_to_spike[n, :] = np.sum(pulse_starts < comparison_arr, axis = 1)

    #print(np.shape(num_pulses_to_spike))
    #print(num_pulses_to_spike)
    num_pulses_to_spike = np.flip(num_pulses_to_spike, axis = 1)
    min_pulse_to_spike = np.min(num_pulses_to_spike)
    max_pulse_to_spike = np.max(num_pulses_to_spike)
    
    fig, ax = plt.subplots(1, 1, figsize = (12, 6))
    base_cmap = plt.cm.get_cmap('GnBu', max_pulse_to_spike-min_pulse_to_spike)
    cax = ax.imshow(num_pulses_to_spike, origin = 'lower', cmap = base_cmap, aspect = 'auto') #, extent = [0, np.shape(num_pulses_to_spike)[1], 0, np.shape(num_pulses_to_spike)[0]])
    # create an extended colorbar and appropriate tick labels
    cbar = fig.colorbar(cax, ax = ax, label = 'Number of Pulses to Spikes', extend = "max")
    ticklabels = [str(t) for t in np.arange(min_pulse_to_spike, max_pulse_to_spike)]
    ticklabels[-1] += '+'
    #ticklabels[-1] = ' '
    cbar.set_ticks(np.linspace(min_pulse_to_spike+0.5, max_pulse_to_spike-0.5, max_pulse_to_spike-min_pulse_to_spike))
    cbar.set_ticklabels(ticklabels)
    x_ticks = np.linspace(0, N_freq-1, 6, dtype = int)
    ax.set_xticks(ticks = x_ticks, labels = np.round(multipulse_freq[x_ticks], 2))
    ax.set_xlabel("Pulse Frequency [Hz]", fontsize = 16)
    ax.set_ylabel("Neuron number", fontsize = 16)
    ax.set_title("Number of pulses to produce a spike", fontsize = 20)

    return fig, ax


def subsequent_spikes(spike_times, pulse_starts, multipulse_freq):
    """
    Will return a binary heatmap of frequencies which produce 2 subsequent spikes
    to highlight the 'firing resonance' of the neuron.

    Args:
        spike_times:        ndarray (N_sims, N_spikes)
                            times for each spike.
        pulse_starts:       ndarray (N_freq, N_pulses)
                            pulse start times.
        multipulse_freq:    array (N_freq, )
                            corresponding frequency for each row of pulse_starts.
    
    
    """
    N_sims = np.shape(spike_times)[0]
    N_freq = np.shape(pulse_starts)[0]
    N_neurons = int(N_sims/N_freq)

    # first_spike time
    first_spike = spike_times[:, 0]
    first_spike = np.nan_to_num(first_spike, nan = 0.0)
    first_spike = first_spike[:, np.newaxis]

    second_spike = spike_times[:, 1]
    second_spike = np.nan_to_num(second_spike, nan = 0.0)
    second_spike = second_spike[:, np.newaxis]

    subsequent_spikes = np.zeros((N_neurons, N_freq), dtype = bool)
    for n in range(N_sims): # each loop is over a specific frequency
        #Get the time of the first pulse.
        neuron_num = n%N_neurons
        freq_num = n//N_neurons
        first_pulse = pulse_starts[freq_num, pulse_starts[freq_num, :] < first_spike[n]]
        if len(first_pulse) == 0:
            first_pulse = np.nan
        else:
            first_pulse = first_pulse[-1]

        # Get the time of the second pulse
        second_pulse = pulse_starts[freq_num, pulse_starts[freq_num, :] < second_spike[n]]
        if len(second_pulse) == 0:      # if no second pulse, then no subsequent spikes
            continue
        else:       # at least 2 spikes
            second_pulse = second_pulse[-1] # start time of the second pulse
            if np.argwhere(pulse_starts[freq_num, :] == first_pulse)[0]+1 == np.argwhere(pulse_starts[freq_num, :] == second_pulse)[0]:
                subsequent_spikes[neuron_num, freq_num] = True

    # Plot the heatmap...
    fig, ax = plt.subplots(1, 1, figsize = (12, 6))
    
    base_cmap = plt.cm.get_cmap("Greys", 2) # 2 colour map
    cax = ax.imshow(subsequent_spikes, origin = 'lower', cmap = base_cmap, aspect = 'auto') #, extent = [0, np.shape(num_pulses_to_spike)[1], 0, np.shape(num_pulses_to_spike)[0]])
    cbar = fig.colorbar(cax, ax = ax, shrink = 0.5)
    ticklabels = ["False", "True"]
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(ticklabels)

    #x_ticks = np.linspace(0, N_freq-1, 12, dtype = int)
    #ax.set_xticks(ticks = x_ticks, labels = np.round(multipulse_freq[x_ticks], 2))
    ax.set_xlabel("Pulse Frequency [Hz]", fontsize = 16)
    ax.set_ylabel("Neuron number", fontsize = 16)
    ax.set_title("Whether the second pulse after a spike, produces a spike", fontsize = 20)

    # define grid
    ax.set_xticks(np.arange(0, np.shape(subsequent_spikes)[1], 5), minor = True)
    ax.set_xticks(np.arange(0, np.shape(subsequent_spikes)[1], 25))
    ax.set_yticks(np.arange(0, np.shape(subsequent_spikes)[0], 1), minor = True)
    ax.set_yticks(np.arange(0, np.shape(subsequent_spikes)[0], 5))
    ax.grid(alpha = 0.4, which = 'minor')
    ax.grid(alpha = 0.8, which = 'major')
    
    return fig, ax

def subsequent_spikes_by_params(spike_times, pulse_starts, multipulse_freq, param_df, params):
    """
    Will return a binary heatmap of frequencies which produce 2 subsequent spikes
    to highlight the 'firing resonance' of the neuron.

    Args:
        spike_times:        ndarray (N_sims, N_spikes)
                            times for each spike.
        pulse_starts:       ndarray (N_freq, N_pulses)
                            pulse start times.
        multipulse_freq:    array (N_freq, )
                            corresponding frequency for each row of pulse_starts.
        params_df:         pd.DataFrame
                            dataframe of neuron parameters
        params:             list
                            list of autapse parameter keys to order the df
    
    
    """
    N_sims = np.shape(spike_times)[0]
    N_freq = np.shape(pulse_starts)[0]
    N_neurons = int(N_sims/N_freq)

    # first_spike time
    first_spike = spike_times[:, 0]
    first_spike = np.nan_to_num(first_spike, nan = 0.0)
    first_spike = first_spike[:, np.newaxis]


    second_spike = spike_times[:, 1]
    second_spike = np.nan_to_num(second_spike, nan = 0.0)
    second_spike = second_spike[:, np.newaxis]

    subsequent_spikes = np.zeros((N_neurons, N_freq), dtype = bool)
    for n in range(N_sims): # each loop is over a specific frequency
        #Get the time of the first pulse.
        neuron_num = n%N_neurons
        freq_num = n//N_neurons
        first_pulse = pulse_starts[freq_num, pulse_starts[freq_num, :] < first_spike[n]]
        if len(first_pulse) == 0:
            first_pulse = np.nan
        else:
            first_pulse = first_pulse[-1]

        # Get the time of the second pulse
        second_pulse = pulse_starts[freq_num, pulse_starts[freq_num, :] < second_spike[n]]
        if len(second_pulse) == 0:      # if no second pulse, then no subsequent spikes
            continue
        else:       # at least 2 spikes
            second_pulse = second_pulse[-1] # start time of the second pulse
            if np.argwhere(pulse_starts[freq_num, :] == first_pulse)[0]+1 == np.argwhere(pulse_starts[freq_num, :] == second_pulse)[0]:
                subsequent_spikes[neuron_num, freq_num] = True

    # order the array by autapse param.
    param_values = param_df[params[0]].values
    unique_params = np.unique(param_values)
    
    num_neurons_with_param = np.zeros(len(unique_params))
    for n, val in enumerate(unique_params):
        num_neurons_with_param[n] = np.sum(param_values == val)
    cumsum = np.cumsum(num_neurons_with_param)

    # order the autapse
    ordered_df = param_df.sort_values(by = params)
    sorted_indices = np.array(ordered_df.index.tolist(), dtype = int)


    # Plot the heatmap...
    fig, ax = plt.subplots(1, 1, figsize = (12, 6))
    base_cmap = plt.cm.get_cmap("Greys", 2) # 2 colour map
    cax = ax.imshow(subsequent_spikes[sorted_indices, :], origin = 'lower', cmap = base_cmap, aspect = 'auto', extent = [0, np.shape(subsequent_spikes)[1], 0, np.shape(subsequent_spikes)[0]])
    cbar = fig.colorbar(cax, ax = ax, shrink = 0.5)
    ticklabels = ["False", "True"]
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(ticklabels)
    x_ticks = np.linspace(0, N_freq-1, 6, dtype = int)
    ax.set_xticks(ticks = x_ticks, labels = np.round(multipulse_freq[x_ticks], 2))
    ax.set_xlabel("Pulse Frequency [Hz]", fontsize = 16)
    ax.set_ylabel(f"Neuron number ordered by {params[0]}", fontsize = 16)
    ax.set_title(f"Whether the pulse after a spike produces a spike (ordered by {params})", fontsize = 20)

    cumsum = np.insert(cumsum, 0, 0.)
    tick_loc = (cumsum[:-1] + cumsum[1:])/2.0
    ax.set_yticks(tick_loc, labels = np.round(unique_params, 2))

    xmin = 0.0
    xmax = np.shape(subsequent_spikes)[1]
    #ax.set_xlim((xmin, xmax))
    ax.hlines(y = np.cumsum(num_neurons_with_param), xmin = xmin, xmax = xmax, colors = "grey", linestyles = "dashed")
    
    return fig, ax

def plot_total_spikes(num_spikes, multipulse_freq, param_df, param_order = []):
    """
    Plots the number of spikes produced at each frequency for each neuron as a heatmap.
    
    Args:
        num_spikes:         ndarray (N_neurons, N_freq)
                            total number of spikes produced by each neuron at each frequency
        multipulse_freq:    ndarray (N_freq, )
                            pulse frequencies sampled.
    """
    N_freq = np.shape(multipulse_freq)[0]

    min_spikes = np.min(num_spikes)
    max_spikes = np.max(num_spikes)

    fig, ax = plt.subplots(1, 1, figsize = (12, 6))



    # might need to order the neurons
    if len(param_order) != 0:       # if an order list is provided, order the array
        param_values = param_df[param_order[0]].values
        unique_params = np.unique(param_values)
        
        num_neurons_with_param = np.zeros(len(unique_params))
        for n, val in enumerate(unique_params):
            num_neurons_with_param[n] = np.sum(param_values == val)
        cumsum = np.cumsum(num_neurons_with_param)

        # order the autapse
        ordered_df = param_df.sort_values(by = param_order)
        sorted_indices = np.array(ordered_df.index.tolist(), dtype = int)
        ax.set_title(f"Number of pulses to produce a spike ordered by {param_order}", fontsize = 20)

        cumsum = np.insert(cumsum, 0, 0.)
        tick_loc = (cumsum[:-1] + cumsum[1:])/2.0
        ax.set_yticks(tick_loc, labels = np.round(unique_params, 2))
        ax.set_ylabel(f"Neuron number ordered by {param_order[0]}", fontsize = 16)

        xmin = 0.0
        xmax = np.shape(num_spikes)[1]
        #ax.set_xlim((xmin, xmax))
        ax.hlines(y = np.cumsum(num_neurons_with_param), xmin = xmin, xmax = xmax, colors = "grey", linestyles = "dashed")
    else:
        sorted_indices = np.array(param_df.index.tolist(), dtype = int)
        ax.set_title("Number of pulses to produce a spike", fontsize = 20)
        ax.set_ylabel("Neuron number", fontsize = 16)


    base_cmap = plt.cm.get_cmap('GnBu', max_spikes-min_spikes)
    cax = ax.imshow(num_spikes, origin = 'lower', cmap = base_cmap, aspect = 'auto') #, extent = [0, np.shape(num_pulses_to_spike)[1], 0, np.shape(num_pulses_to_spike)[0]])
    # create an extended colorbar and appropriate tick labels
    cbar = fig.colorbar(cax, ax = ax, label = 'Number of Spikes', extend = "max")
    ticklabels = [str(t) for t in np.arange(min_spikes, max_spikes)]
    ticklabels[-1] += '+'
    cbar.set_ticks(np.linspace(min_spikes+0.5, max_spikes-0.5, max_spikes-min_spikes))
    cbar.set_ticklabels(ticklabels)
    x_ticks = np.linspace(0, N_freq-1, 6, dtype = int)
    ax.set_xticks(ticks = x_ticks, labels = np.round(multipulse_freq[x_ticks], 2))
    ax.set_xlabel("Pulse Frequency [Hz]", fontsize = 16)

    return fig, ax