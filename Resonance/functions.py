
import sys
sys.path.append("..\\")

import numpy as np
from batchAQUA_general import batchAQUA
from AQUA_general import AQUA
from stimulus import *
import matplotlib.pyplot as plt




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

    I_inj = np.array([i*np.ones(N_iter) for i in I_list], dtype = np.float32)

    # batch list of params
    params = [neuron_params for n in range(N_neurons)]

    x_start = np.full((N_neurons, 3), fill_value = np.array([neuron_params["c"], 0., 0.], dtype=np.float32))

    t_start = np.zeros(N_neurons, dtype=np.float32)

    neurons = batchAQUA(params)
    neurons.Initialise(x_start, t_start)

    _, _, spikes = neurons.update_batch(dt, N_iter, I_inj)

    idx_threshold = np.argwhere(np.isnan(spikes[:, 0]).flatten())[-1]

    sub_threshold = I_list[idx_threshold]
    above_threshold = I_list[idx_threshold + 1]

    #redefine I_inj
    I_list_thresh = np.linspace(sub_threshold, above_threshold, N_neurons, dtype=np.float32)
    I_inj_thresh = np.array([i*np.ones(N_iter, dtype = np.float32) for i in I_list_thresh], dtype = np.float32)

    neurons.Initialise(x_start, t_start)
    X, _, spikes = neurons.update_batch(dt, N_iter, I_inj_thresh)

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

    I_inj = np.array([spikes_constant(N_iter, dt, threshold, 0.0, 1, i, pulse_duration, delay) for i in I_list], dtype = np.float32)

    # batch list of params
    params = [neuron_params for n in range(N_neurons)]

    x_start = np.full((N_neurons, 3), fill_value = x_ini, dtype = np.float32)
    t_start = np.zeros(N_neurons, dtype = np.float32)

    neurons = batchAQUA(params)
    neurons.Initialise(x_start, t_start)

    _, _, spikes = neurons.update_batch(dt, N_iter, I_inj)
    
    idx_threshold = np.argwhere(~np.isnan(spikes[:, 0]).flatten())[0]
    pulse_height_low = I_list[idx_threshold]        # first estimate of the pulse height
    pulse_height_high = I_list[idx_threshold + 1]   # second estimate of the pulse height

    # refine the search
    I_list_refine = np.linspace(pulse_height_low, pulse_height_high, N_neurons, dtype = np.float32)
    I_inj_refine = np.array([spikes_constant(N_iter, dt, threshold, 0.0, 1, i, pulse_duration, delay) for i in I_list_refine], dtype = np.float32)

    # re-initialise the neurons
    neurons.Initialise(x_start, t_start)
    _, _, spikes = neurons.update_batch(dt, N_iter, I_inj_refine)

    idx_threshold = np.argwhere(~np.isnan(spikes[:, 0]).flatten())[0]
    pulse_height = I_list_refine[idx_threshold][0]     # close estimate of the pulse height

    pulse_end = delay + pulse_duration - dt
    spike_time = spikes[idx_threshold, 0] - pulse_end
    
    return pulse_height, spike_time

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
    I_inj = threshold*np.ones((N_neurons, N_iter), dtype = np.float32)
    pulse1_start = int(delay/dt)
    pulse1_end = int((delay+pulse_duration)/dt)
    I_inj[:, pulse1_start:pulse1_end] += first_pulse_height # create the first pulse
    
    pulse2_start = pulse1_end + int((time_to_spike + resonant_ISI)/dt)
    pulse2_end = pulse2_start + int(pulse_duration/dt)
    for n, i in enumerate(I_list):      # define the second pulse as 
        I_inj[n, pulse2_start:pulse2_end] += i  


    # batch list of params
    params = [neuron_params for n in range(N_neurons)]

    x_start = np.full((N_neurons, 3), fill_value = x_ini, dtype = np.float32)
    t_start = np.zeros(N_neurons, dtype = np.float32)

    neurons = batchAQUA(params)
    neurons.Initialise(x_start, t_start)

    _, _, spikes = neurons.update_batch(dt, N_iter, I_inj)
    print(spikes)

    idx_threshold = np.argwhere(~np.isnan(spikes[:, 1]).flatten())[0]
    pulse_height_low = I_list[idx_threshold]            # first estimate of the pulse height
    pulse_height_high = I_list[idx_threshold + 1]       # second estimate of the pulse height

    # refine the search
    I_list_refine = np.linspace(pulse_height_low, pulse_height_high, N_neurons, dtype = np.float32)
    I_inj_refine = threshold*np.ones((N_neurons, N_iter), dtype = np.float32)
    I_inj_refine[:, pulse1_start:pulse1_end] += first_pulse_height
    for n, i in enumerate(I_list_refine):      # define the second pulse as 
        I_inj_refine[n, pulse2_start:pulse2_end] += i
    
    # re-initialise the neurons
    neurons.Initialise(x_start, t_start)
    _, _, spikes = neurons.update_batch(dt, N_iter, I_inj_refine)

    idx_threshold = np.argwhere(~np.isnan(spikes[:, 1]).flatten())[0]
    pulse_height = I_list_refine[idx_threshold][0]     # close estimate of the pulse height
    
    return pulse_height


def get_resonance_bands(resonant_f, spike_boolean):
    """
    Returns the resonance bands given the frequencies tested and number of spikes produced
    
    """
    band_limits = np.diff(spike_boolean)

    band_starts = np.argwhere(band_limits == 1).flatten()
    band_ends = np.argwhere(band_limits == -1).flatten()

    frequency_bands = np.zeros((len(band_starts), 2))
    for n in range(len(band_starts)):
        if n > len(band_ends) - 1:   # if there are more starts than ends
            frequency_bands[n] = np.array([resonant_f[band_starts[n]], 0.0])
        else:
            frequency_bands[n] = np.array([resonant_f[band_starts[n]], resonant_f[band_ends[n]]])
    
    return frequency_bands


def plot_resonance_map(frequencies, bands):
    """
    Returns a heat map of the resonance frequencies of the neurons

    IN:
        frequencies:    array (N_neurons, N_frequencies)
                        list of frequencies tested

        bands:          ndarray (N_neurons, N_frequencies)
                        resonance bands for each neuron
    
    """

    fig, ax = plt.subplots(1, 1, figsize = (8, 8))

    ax.imshow(bands, origin = 'lower', cmap = 'Greys')

    ax.set_xlabel('Frequency (Hz)', fontsize = 16)
    ax.set_ylabel('Neuron index', fontsize = 16)

    ax.set_title('Resonance Map of neurons', fontsize = 20)

    return fig, ax

def plot_resonance_for_autapse_param(parameters, frequencies, bands, autapse_param):
    """
    Plots the resonance properties of neurons for a given autapse parameter

    IN:
        parameters:     DataFrame
                        dataframe of all neuron parameters tested

        frequencies:    array (N_neurons, N_frequencies)
                        list of frequencies tested

        bands:          ndarray (N_neurons, N_frequencies)
                        resonance bands for each neuron

        autapse_param:  str
                        name of the autapse param to plot against resonance
    """

    param_values = parameters[autapse_param].values
    unique_params = np.unique(param_values)

    argsort_params = np.argsort(param_values)       # indices that would sort the array

    fig, ax = plt.subplots(1, 1, figsize = (8, 8))

    ax.imshow(bands[argsort_params, :], origin = 'lower', cmap = 'Greys', aspect = 'auto', extent = [frequencies[0], frequencies[-1], 0, len(param_values)])

    ax.set_xlabel('Frequency (Hz)', fontsize = 16)
    ax.set_ylabel(f'{autapse_param} value', fontsize = 16)
    ax.set_yticks(np.linspace(np.min(param_values), np.max(param_values), len(unique_params)), labels = np.round(unique_params, 2))

    ax.set_title(f'Resonance Map of neurons for varying {autapse_param}', fontsize = 20)
    ax.legend()

    return fig, ax

# Plots of the spike to pulse ratio from simulation 2
def plot_spike_to_pulse_ratio(multipulse_freq, spike_to_pulse_ratio):
    """
    Plots the spike to pulse ratio for each neuron tested

    IN:
        spike_to_pulse_ratio:   array (N_neurons, N_isi)
                                ratio of spikes to pulses for each neuron at each isi
    """

    fig, ax = plt.subplots(1, 1, figsize = (8, 8))

    cax = ax.imshow(spike_to_pulse_ratio, origin = 'lower', cmap = 'viridis', aspect = 'auto', extent = [0, np.shape(spike_to_pulse_ratio)[1], 0, np.shape(spike_to_pulse_ratio)[0]])
    fig.colorbar(cax, ax = ax, label = 'Spike to Pulse Ratio')

    ax.set_xlabel('Pulse frequency [Hz]', fontsize = 16)
    ax.set_xticks(np.linspace(0, np.shape(spike_to_pulse_ratio)[1], np.shape(multipulse_freq)[1]), labels = np.round(multipulse_freq[0, :], 2))
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
    ax.set_xticks(np.linspace(0, np.shape(spike_to_pulse_ratio)[1], np.shape(multipulse_freq)[1]), labels = np.round(multipulse_freq[0, :], 2))
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
    ax.set_xticks(np.linspace(0, np.shape(spike_to_pulse_freq_ratio)[1], np.shape(multipulse_freq)[1]), labels = np.round(multipulse_freq[0, :], 2))
    ax.set_ylabel('Neuron index', fontsize = 16)

    ax.set_title('Spike/Pulse Frequency for each neuron', fontsize = 20)

    return fig, ax

def plot_spike_to_pulse_ratio_for_param(parameters, multipulse_freq, spike_to_pulse_freq_ratio, autapse_param):
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
    ax.set_xticks(np.linspace(0, np.shape(spike_to_pulse_freq_ratio)[1], np.shape(multipulse_freq)[1]), labels = np.round(multipulse_freq[0, :], 2))
    ax.set_ylabel(f'{autapse_param} value', fontsize = 16)
    ax.set_yticks(np.linspace(np.min(param_values), np.max(param_values), len(unique_params)), labels = np.round(unique_params, 2))

    ax.set_title(f'Spike to Pulse Ratio for each neuron, sorted by the {autapse_param} parameter', fontsize = 20)

    return fig, ax