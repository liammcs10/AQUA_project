# append AQUA directory to sys.path
import sys
sys.path.append("..\\") # parent directory
from AQUA_class import AQUA
from plotting_functions import *

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from tqdm import tqdm   # for a progress bar
import seaborn as sns
sns.set_theme(style = "white")

# local imports
import CLI
import CF
from functions import *

"""
For a given set of neuron parameters:
    We need to identify the threshold current.
    Apply a set of stimuli just below this threshold.

"""


def sim(args, conf):
    """
    Implements the simulation run for data generation.
    Will use a few autapse params but mainly test different step parameters.

    IN:
        args:   arguments passed to main in terminal
                config filename, output filename, save boolean

        conf:   dict
                configuration parameters for the simulation
    
    OUTPUT:
        df:     pandas dataframe
                saved as a .csv and includes run params

    """

    # neuron params to be used for every simulation below
    params = cast_to_float(conf["Neuron"])

    # extract autapse params
    f_vals   = np.array(conf["Autapse"]["f"].split(", "), dtype = np.float64)
    e_vals   = np.array(conf["Autapse"]["e"].split(", "), dtype = np.float64)
    tau_vals = np.array(conf["Autapse"]["tau"].split(", "), dtype = np.float64)

    # calculate the number of neurons
    N_neurons = 1 + len(f_vals) * len(e_vals) * len(tau_vals)
    # params arr will store all the parameter dictionaries in 1 place.
    params_arr = []
    params_arr.append(params) # index 0, will be the reference non-autaptic neuron
    for f in f_vals:
        for e in e_vals:
            for tau in tau_vals:
                temp_dict = params
                temp_dict["f"] = f
                temp_dict["e"] = e
                temp_dict["tau"] = tau
                params_arr.append(temp_dict)

    ## 1st test: 2 pulses, each produce a spike. What is the timing of the second pulse?
    # frequencies:      shape (N_neurons, N_freq), frequencies of the pulse relative to the spike
    # bands:            shape (N_neurons, N_freq), for each neuron, when 2 spikes were emitted
    frequencies, bands, pulse_height, time_to_spike = first_test(conf, N_neurons, params_arr)
    resonance_bands_base_neuron = get_resonance_bands(frequencies[0, :], bands[0, :])
    largest_band_idx = np.argmax(np.diff(resonance_bands_base_neuron, axis = 0))
    peak_resonance = np.mean(resonance_bands_base_neuron[largest_band_idx])

    ## 2nd test: a train of pulses all equally spaced. We want to see if a spike occurrs on subsequent pulses
    N_pulses, multipulse_freq, num_spikes, spike_freq = second_test(conf, N_neurons, params_arr, peak_resonance, pulse_height, time_to_spike)
    
    spike_to_pulse_ratio = num_spikes / N_pulses
    spike_to_pulse_freq_ratio = spike_freq / multipulse_freq


    # At this stage just save the outputs
    results_dict = {"Parameters": params_arr,
                    "Simulation 1": {"frequencies": frequencies, "bands": bands},
                    "Simulation 2": {"multipulse_freq": multipulse_freq, "num_spikes": num_spikes, "spike_freq": spike_freq}}

    # save the results dict as a pickle
    with open(args.outfile, 'wb') as f:
        pickle.dump(results_dict, f)
    

""" - - - - SIMULATION FUNCTIONS - - - - """

def first_test(conf, N_neurons, params_arr):
    """
    Simulates 2 pulses that generate a spike alone. Extracts the resonance bands for each neuron type.

    RETURNS:
        resonant_f:         The frequency of the pulse relative to the time of the spike
        bands:              When 2 spikes were generated or not, index matched to resonant_ISI
    
    """
    print("SIMULATION 1")
    # Define the simulation params
    T = float(conf['Simulation 1']['T'])        # needs to be in ms
    dt = float(conf['Simulation 1']['dt'])      # ms
    N_iter = int(T/dt)                          # number of iterations

    print("Finding threshold...")
    threshold, x_ini = find_threshold(params_arr[0], np.linspace(0, 500, 100), T, dt)
    
    # extract frequencies to investigate
    freq_start = float(conf["Simulation 1"]["freq_start"])
    freq_stop = float(conf["Simulation 1"]["freq_stop"])
    N_freq = int(conf["Simulation 1"]["N_freq"])
    freq_range = np.linspace(freq_start, freq_stop, N_freq)

    pulse_duration = float(conf["Simulation 1"]["pulse_duration"])

    N_pulses = 2    
    delay = float(conf["Simulation 1"]["delay"])          # ms
    pulse1_end = delay + pulse_duration - dt

    print("Finding pulse height...")
    # the height of the pulse which produces a spike and the relative timing of the spike.
    pulse_height, time_to_spike = find_pulse_height(params_arr[0], np.linspace(100, 1000, 100), threshold, x_ini, pulse_duration)
    
    # define the frequency of the pulses relative to the spike timing.
    ISI_range = time_to_spike + 1000/freq_range
    print(np.shape(ISI_range))

    # Define the injected current
    # structure: 0:N_neurons @ a given isi, increment isi and define N_neurons more neurons...
    I_inj = np.array([spikes_constant(N_iter, dt, threshold, isi, N_pulses, pulse_height, pulse_duration, delay) for isi in ISI_range for n in range(N_neurons)])
    N_sims = np.shape(I_inj)[0]     # N_neurons x N_isis

    # define batch now that we know how many simulations are run
    batch1_params = []
    for i in range(N_freq):
        batch1_params += params_arr
    x_start = np.full((N_sims, 3), fill_value = x_ini)
    t_start = np.zeros(N_sims)
    # define and initialize

    batch1 = batchAQUA(batch1_params)
    batch1.Initialise(x_start, t_start)

    X, T, spikes = batch1.update_batch(dt, N_iter, I_inj)

    pulse2_start = np.zeros(N_sims)
    spike_boolean = np.zeros(N_sims)

    for n in range(N_sims):
        if len(spikes[n, np.isnan(spikes[n])]) != 1:        # if 2 spikes generated
            spike_boolean[n] = 1
        pulse_times = np.argwhere(I_inj[n, :] >= threshold + 1) * dt  # pulse times in ms
        pulse2_start[n] = pulse_times[np.where(pulse_times > pulse1_end)][0] # start time of the second pulse, ms

    InterPulse_Intervals = pulse2_start - pulse1_end
    first_spike_time = pulse1_end + time_to_spike

    resonant_ISI = (pulse2_start - first_spike_time).reshape((N_neurons, N_freq), order = 'F')
    resonant_f = 1000/resonant_ISI

    bands = spike_boolean.reshape((N_neurons, N_freq), order = 'F')

    return resonant_f, bands, pulse_height, time_to_spike

def second_test(conf, N_neurons, params_arr, peak_resonance, pulse_height, time_to_spike):
    """
    The second test is a train of equally spaced, identical pulses. 
    We want to extract information about how reliable the response spiking is

    RETURNS:
        N_pulses:           int
                            number of pulses injected
        multipulse_freq:    arr,  (N_neurons, N_sims)
                            the frequency 
        num_spikes:         array, (N_neurons, N_sims)
                            the total number of spikes produced in a given simulation.
        spike_freq:         array, (N_neurons, N_sims)
                            the mean frequency of spiking in a given simulation.  
    """
    # Get the number of neuron models
    N_neurons = np.shape(params_arr)[0]

    # Define the simulation params
    T = float(conf['Simulation 2']['T'])        # needs to be in ms
    dt = float(conf['Simulation 2']['dt'])      # ms
    N_iter = int(T/dt)                          # number of iterations

    print("Finding threshold...")
    threshold, x_ini = find_threshold(params_arr[0], np.linspace(0, 500, 100), T, dt)

    pulse_frequency = peak_resonance   # optimal resonance of base neuron model... good assumption?
    pulse_ISI = 1000/pulse_frequency        # ms
    pulse2_heights = np.linspace(10, pulse_height, 200)
    pulse_duration = float(conf["Simulation 2"]["pulse_duration"])

    # get the minimum pulse height that induces a spike if it arrives at the peak resonance frequency.
    pulse2_height = find_2nd_pulse_height(params_arr[0], pulse2_heights, threshold, x_ini, pulse_duration, pulse_height, time_to_spike, pulse_ISI)

    pulse_isi_low = float(conf["Simulation 2"]["isi_low"])
    pulse_isi_high = float(conf["Simulation 2"]["isi_high"])
    N_isi = float(conf["Simulation 2"]["N_isi"])

    multipulse_ISIs = np.linspace(pulse_isi_low, pulse_isi_high, N_isi)
    multipulse_freq = 1000/multipulse_ISIs

    N_pulses = int(conf["Simulation 2"]["N_pulses"])
    delay = float(conf["Simulation 2"]["delay"])

    # define the injected current array
    I_multi = np.array([spikes_constant(N_iter, dt, threshold, isi, N_pulses, pulse2_height, pulse_duration, delay) for isi in multipulse_ISIs for n in range(N_neurons)])
    print(np.shape(I_multi))
    N_sims = np.shape(I_multi)[0]       # N_neurons * N_isis

    # Can now setup the batch
    batch_params = []
    for i in range(N_isi):
        batch_params += params_arr
    x_start = np.full((N_sims, 3), fill_value = x_ini)
    t_start = np.zeros(N_sims)

    # define and initialize
    batch = batchAQUA(batch_params)
    batch.Initialise(x_start, t_start)

    #simulate
    X, T, spikes = batch.update_batch(dt, N_iter, I_multi)

    # get the mean frequency of spiking (to compare with mean frequency of pulses)
    spike_ISI = np.diff(spikes, axis = 1)
    spike_freq = 1000/np.nanmean(spike_ISI, axis = 1)
    spike_freq = np.nan_to_num(spike_freq, nan = 0.0)
    spike_freq = spike_freq.reshape((N_neurons, N_isi), order = 'F')

    # Get the total number of spikes (to compare with the total number of pulses)
    num_spikes = np.zeros((N_neurons, N_isi), dtype = int)
    for i in range(N_sims):
        num_spikes[i] = len(spikes[i, ~np.isnan(spikes[i])])
    
    num_spikes = num_spikes.reshape((N_neurons, N_isi), order = 'F')

    return N_pulses, multipulse_freq, num_spikes, spike_freq



""" - - - - HELPER FUNCTIONS - - - - """

def cast_to_float(data_dict):
    """
    Casts the values of a dictionary to float if conversion is possible.
    Otherwise, the original value is retained.
    """
    new_dict = {
        key: float(value) 
        if isinstance(value, (int, str)) and value not in ('', None) and is_float(value)
        else value
        for key, value in data_dict.items()
    }
    return new_dict

def is_float(value):
    """Helper function to safely check if a string can be converted to float."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False