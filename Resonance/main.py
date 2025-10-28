"""
Perform a large parameter search over the space of autapse parameters.


"""

import sys
import os
sys.path.append("..\\") # parent directory

from AQUA_general import AQUA
from batchAQUA_general import batchAQUA
from plotting_functions import *
from stimulus import *

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

import CLI
import CF
import simulation
from functions import *


def main():
    """
    This file simulates the response of the IB neuron with different autaptic params
    to a fixed injected current. 

    The results are saved as .pickle files and plotted as heatmaps.

    Arguments
    ----------
    sim     :   action
                whether to generate data or not
    config  :   str
                config file name
    pickle  :   str
                data filename if no sim run.
    out     :   str
                output filename
    save    :   action
                whether to save the output plots    
    
    Outputs
    ----------
    pickle_file :  pickle containing the simulation results for all parameters

    ef_map  :      plot
                   shows information entropy of the spike train for different e and f params
    
    etau_map  :    plot
                   shows information entropy of the spike train for different e and tau params
    
    ftau_map  :    plot
                   shows information entropy of the spike train for different f and tau params
    """

    args = CLI.command_line()

    # Check for config file
    if args.config is None:
        print("No config file passed... exiting")
        quit()
    else:
        conf = CF.read_conf(args.config)
        print("Config extracted...")

    # Create the output directory if needed
    output_directory = conf["Output"]["out_dir"]
    try:
        os.mkdir(output_directory)
        print(f"Creating output directory: {output_directory}")
    except FileExistsError:
        print("Output folder already exists")
        pass
    
    """- - - - SIMULATION - - - -"""
    if args.sim:    # Run the whole simulation if --sim is passed
        print("Running simulation...")

        simulation.sim(args, conf)
        print("Simulation complete.")
    

    """- - - - PLOTTING - - - -"""
    # import data
    data_file = conf["Output"]["data_file"]
    with open(f"{output_directory}\\{data_file}", 'rb') as f:
        dict = pickle.load(f)
    # neurons tested
    parameters = dict["Parameters"]

    param_df = pd.DataFrame(parameters)
    neuron_name = conf["Neuron"]["name"]
    
    # Simulation 1 outputs
    frequencies = dict["Simulation 1"]["frequencies"]
    bands = dict["Simulation 1"]["bands"]

    # Simulation 2 outputs
    N_pulses = int(conf["Simulation 2"]["N_pulses"])
    multipulse_freq = dict["Simulation 2"]["multipulse_freq"]
    num_spikes = dict["Simulation 2"]["num_spikes"]
    spike_freq = dict["Simulation 2"]["spike_freq"]
    spike_times = dict["Simulation 2"]["spike_times"]
    pulse_starts = dict["Simulation 2"]["pulse_starts"]

    print(np.shape(num_spikes))

    spike_to_pulse_ratio = num_spikes/N_pulses
    spike_to_pulse_frequency = spike_freq/multipulse_freq

    """ Test 1 plots """

    ## Now plot the data...
    fig1, ax1 = plot_resonance_map(frequencies, bands)
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_resonator_unordered.png")
    plt.show()

    # plot ordered by 'e'
    fig2, ax2 = plot_resonance_for_autapse_param(param_df, frequencies, bands, ["e"])
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_resonator_by_e.png")

    
    # plot ordered by 'f'
    fig3, ax3 = plot_resonance_for_autapse_param(param_df, frequencies, bands, ["f"])
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_resonator_by_f.png")


    # plot ordered by 'tau'
    fig4, ax4 = plot_resonance_for_autapse_param(param_df, frequencies, bands, ["tau"])
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_resonator_by_tau.png")


    fig4, ax4 = plot_resonance_for_autapse_param(param_df, frequencies, bands, ["e", "f"])
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_resonator_by_e_f.png")


    """ Test 2 plots """

    #Spike to pulse ratios
    #fig5, ax5 = plot_spike_to_pulse_ratio(multipulse_freq, spike_to_pulse_ratio)
    #if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_spike_pulse.png")

    # spike to pulse frequency (gives a sense of how many pulses are needed per spike.)
    #fig6, ax6 = plot_spike_to_pulse_frequency(multipulse_freq, spike_to_pulse_frequency)
    #if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_spike_freq.png")

    fig, ax = pulses_to_spike(spike_times, pulse_starts, multipulse_freq)
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_pulsesToSpike.png")

    fig, ax = subsequent_spikes(spike_times, pulse_starts, multipulse_freq)
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_subSpikes_unordered.png")

    fig, ax = subsequent_spikes_by_params(spike_times, pulse_starts, multipulse_freq, param_df, ["e", "f"])
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_subSpikes_by_e_f.png")

    fig, ax = subsequent_spikes_by_params(spike_times, pulse_starts, multipulse_freq, param_df, ["f", "e"])
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_subSpikes_by_f_e.png")

    fig, ax = subsequent_spikes_by_params(spike_times, pulse_starts, multipulse_freq, param_df, ["tau"])
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_subSpikes_by_tau.png")

    fig, ax = plot_total_spikes(num_spikes, multipulse_freq, param_df, param_order = ["e", "f"])
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_totalSpikes_by_e_f.png")

    fig, ax = plot_total_spikes(num_spikes, multipulse_freq, param_df, param_order = ["f", "e"])
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_totalSpikes_by_f_e.png")

    fig, ax = plot_total_spikes(num_spikes, multipulse_freq, param_df, param_order = ["tau"])
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_totalSpikes_by_tau.png")


    # Can then organise by parameter values...

main()