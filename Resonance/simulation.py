# append AQUA directory to sys.path
import sys
import os
sys.path.append("..\\") # parent directory
from AQUA_general import AQUA
from batchAQUA_general import batchAQUA
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
    print("- - - - - - - - -")
    for f in f_vals:
        for e in e_vals:
            for tau in tau_vals:
                temp_dict = params.copy()
                temp_dict["f"] = f
                temp_dict["e"] = e
                temp_dict["tau"] = tau
                params_arr.append(temp_dict)

    ## 1st test: 2 pulses, each produce a spike. What is the timing of the second pulse?
    # frequencies:      shape (N_freq, ), frequencies of the pulse relative to the spike
    # bands:            shape (N_neurons, N_freq), for each neuron, when 2 spikes were emitted
    frequencies, bands, pulse_height, time_to_spike = first_test(conf, N_neurons, params_arr)

    resonance_bands_base_neuron = get_resonance_bands(frequencies, bands[0, :]) # idx 0 is the base neuron

    largest_band_idx = np.argmax(np.diff(resonance_bands_base_neuron, axis = 1))
    peak_resonance = np.mean(resonance_bands_base_neuron[largest_band_idx])
    print(f"Peak Resonance: {peak_resonance}")

    ## 2nd test: a train of pulses all equally spaced. We want to see if a spike occurrs on subsequent pulses
    N_pulses, multipulse_freq, num_spikes, spike_freq, spike_times, pulse_starts = second_test(conf, N_neurons, params_arr, peak_resonance, pulse_height+10, time_to_spike)
    
    spike_to_pulse_ratio = num_spikes / N_pulses
    spike_to_pulse_freq_ratio = spike_freq / multipulse_freq


    # At this stage just save the outputs
    results_dict = {"Parameters": params_arr,
                    "Simulation 1": {"frequencies": frequencies, "bands": bands},
                    "Simulation 2": {"multipulse_freq": multipulse_freq, "num_spikes": num_spikes, "spike_freq": spike_freq, "spike_times": spike_times, "pulse_starts": pulse_starts}}

    # save the results dict as a pickle
    output_directory = conf["Output"]["out_dir"]
    output_data_file = conf["Output"]["data_file"]
    output_path = output_directory + "\\" + output_data_file
    with open(output_path, 'wb') as f:
        pickle.dump(results_dict, f)
    

""" - - - - SIMULATION FUNCTIONS - - - - """

"""  TWIN PULSES  """

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
    N_iter = int(1000*T/dt)                          # number of iterations

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
    pulse1_end = delay + pulse_duration

    print("Finding pulse height...")
    # the height of the pulse which produces a spike and the relative timing of the spike.
    pulse_height, pulse_height2, time_to_spike = find_pulse_height(params_arr[0], np.linspace(100, 1000, 100), threshold, x_ini, pulse_duration)
    #pulse_height2 = pulse_height - 2 # second pulse is slightly weaker
    pulse_heights = [pulse_height, pulse_height2]

    # define the frequency of the pulses relative to the spike timing.
    ISI_range = time_to_spike + 1000/freq_range

    # Define the injected current
    # structure: 0:N_neurons @ a given isi, increment isi and define N_neurons more neurons...
    I_inj = np.array([spikes_constant(N_iter, dt, threshold, isi, N_pulses, pulse_heights, pulse_duration, delay) for isi in ISI_range for _ in range(N_neurons)])
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

    pulse2_start  = np.zeros(N_sims)
    spike_boolean = np.zeros(N_sims)

    for n in range(N_sims):
        if len(spikes[n, np.isnan(spikes[n])]) != 1:        # if 2 spikes generated
            spike_boolean[n] = 1
        pulse_times = np.argwhere(I_inj[n, :] > threshold) * dt                     # pulse times in ms
        pulse2_start[n] = pulse_times[np.where(pulse_times > pulse1_end)][0]        # start time of the second pulse, ms

    #InterPulse_Intervals = pulse2_start - pulse1_end
    first_spike_time = pulse1_end + time_to_spike

    resonant_ISI = (pulse2_start - first_spike_time).reshape((N_neurons, N_freq), order = 'F')
    resonant_ISI = resonant_ISI[0, :]   # only take the first row as all rows are identical
    resonant_f = 1000/resonant_ISI # represents the frequency of the pulses w.r.t the spike time
    
    bands = spike_boolean.reshape((N_neurons, N_freq), order = 'F') # where 2 spikes were generated. Maps to the frequencies.
    
    
    return resonant_f, bands, pulse_height, time_to_spike


"""  MULTIPULSE TEST  """

def second_test(conf, N_neurons, params_arr, peak_resonance, pulse_height, time_to_spike):
    """
    The second test is a train of equally spaced, identical pulses. 
    We want to extract information about how reliable the response spiking is.
    When setting parameters, need to make sure that all pulses will fit into the
    simulation.

    RETURNS:
        N_pulses:           int
                            number of pulses injected
        multipulse_freq:    arr,  (N_neurons, N_freq)
                            the frequency 
        num_spikes:         array, (N_neurons, N_freq)
                            the total number of spikes produced in a given simulation.
        spike_freq:         array, (N_neurons, N_freqa)
                            the mean frequency of spiking in a given simulation.
        spike_times:        ndarray (N_neurons, Inh.)
                            spike times for each neuron.
        pulse_start:        ndarray (N_neurons, N_pulses)
                            start times of each pulse. Pulse end is +pulse_duration.
    """

    # Get the number of neuron models
    N_neurons = np.shape(params_arr)[0]

    # Define the simulation params
    T = float(conf['Simulation 2']['T'])        # needs to be in ms
    dt = float(conf['Simulation 2']['dt'])      # ms
    N_iter = int(1000*T/dt)                          # number of iterations

    # import simulation config.
    N_pulses = int(conf["Simulation 2"]["N_pulses"])
    delay = float(conf["Simulation 2"]["delay"])
    pulse_duration = float(conf["Simulation 2"]["pulse_duration"])
    
    pulse_isi_low = float(conf["Simulation 2"]["isi_low"])
    pulse_isi_high = float(conf["Simulation 2"]["isi_high"])
    N_isi = int(conf["Simulation 2"]["N_isi"])

    assert T*1000 >= delay + pulse_duration + N_pulses*pulse_isi_high, "Multipulse simulation parameters don't fit in the time frame."

    print("Finding threshold...")
    threshold, x_ini = find_threshold(params_arr[0], np.linspace(0, 500, 100), T, dt)

    pulse_frequency = peak_resonance   # optimal resonance of base neuron model... good assumption?
    pulse_ISI = 1000/pulse_frequency        # ms
    pulse2_heights = np.linspace(10, 1000, 200)

    # get the minimum pulse height that induces a spike if it arrives at the peak resonance frequency.
    pulse2_height = find_2nd_pulse_height(params_arr[0], pulse2_heights, threshold, x_ini, pulse_duration, pulse_height, time_to_spike, pulse_ISI)
    pulse2_height += 10

    multipulse_ISIs = np.linspace(pulse_isi_low, pulse_isi_high, N_isi) # (N_isi, )
    multipulse_freq = 1000/multipulse_ISIs

    # get the pulse start times (in ms). Pulse ends is just += pulse_duration
    pulse_starts = np.array([[delay + i*(isi) for i in range(N_pulses)] for isi in multipulse_ISIs])

    # define the injected current array
    I_multi = np.array([spikes_constant(N_iter, dt, threshold, isi, N_pulses, pulse2_height, pulse_duration, delay) for isi in multipulse_ISIs for n in range(N_neurons)])
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
    num_spikes = np.zeros(N_sims, dtype = int)
    for i in range(N_sims):
        num_spikes[i] = len(spikes[i, ~np.isnan(spikes[i])])
    
    num_spikes = num_spikes.reshape((N_neurons, N_isi), order = 'F')

    return N_pulses, multipulse_freq, num_spikes, spike_freq, spikes, pulse_starts



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