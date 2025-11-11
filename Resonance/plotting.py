"""
Pipeline for plotting the results produced by simulation.py

"""
import sys
sys.path.append("..\\")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from functions import *
from plotting_functions import *




def plot(args, conf):

    # import data
    output_directory = conf["Output"]["out_dir"]
    data_file = conf["Output"]["data_file"]
    with open(f"{output_directory}\\{data_file}", 'rb') as f:
        dict = pickle.load(f)
    
    # neuron parameters
    params = dict["Parameters"]
    param_df = pd.DataFrame(params)


    # test 1
    test1_plots(dict, conf, param_df, args)

    # 3 pulses
    three_pulse_plots(dict, conf, param_df, args)

    # test 2
    test2_plots(dict, conf, param_df, args)


def test1_plots(dict, conf, param_df, args):
    """
    Plotting the resonance bands from test 1.
    Test 1 involves 2 pulses: first pulse generates a spike, second pulse is slightly below
    threshold. Only resonance frequencies produce 2 spikes.

    """
    neuron_name = conf["Neuron"]["name"]
    output_directory = conf["Output"]["out_dir"]

    # Simulation 1 outputs
    frequencies1 = dict["Simulation 1"]["frequencies"]
    bands1 = dict["Simulation 1"]["bands"]

    ## Now plot the data...
    fig, ax, _ = plot_resonance_map(frequencies1, bands1)
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_resonator_unordered.png")
    plt.close(fig)

    # plot ordered by 'e'
    fig, ax, _ = plot_resonance_for_autapse_param(param_df, frequencies1, bands1, ["e"])
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_resonator_by_e.png")
    plt.close(fig)
    
    # plot ordered by 'f'
    fig, ax, _ = plot_resonance_for_autapse_param(param_df, frequencies1, bands1, ["f"])
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_resonator_by_f.png")
    plt.close(fig)

    # plot ordered by 'tau'
    fig, ax, _ = plot_resonance_for_autapse_param(param_df, frequencies1, bands1, ["tau"])
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_resonator_by_tau.png")
    plt.close(fig)

    fig, ax, _ = plot_resonance_for_autapse_param(param_df, frequencies1, bands1, ["e", "f"])
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_resonator_by_e_f.png")
    plt.close(fig)


def three_pulse_plots(dict, conf, param_df, args):

    """
    Plot the results for the 3 pulse protocol: first pulse produces a spike, subsequent 
    2 pulses are below threshold and test the resonance frequency.
        
    """
    neuron_name = conf["Neuron"]["name"]
    output_directory = conf["Output"]["out_dir"]

    # Three pulse output
    frequencies2 = dict["Three pulses"]["frequencies"]
    bands2 = dict["Three pulses"]["bands"]
    num_spikes_three_pulses = dict["Three pulses"]["num_spikes"]


    fig, ax, _ = plot_resonance_map(frequencies2, bands2, title = 'Resonance map for three-pulse protocol')
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_resonator_3pulses_unordered.png")
    plt.close(fig)

    fig, ax, _ = plot_resonance_for_autapse_param(param_df, frequencies2, bands2, ["e", "f"], title = 'Resonance map for three-pulse protocol ordered by')
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_resonator_3pulses_by_e_f.png")
    plt.close(fig)

    fig, ax, im = plot_resonance_map(frequencies2, num_spikes_three_pulses, title = 'Number of spikes emitted by \nthe 3-pulse protocol')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number of spikes produced")
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_resonator_num_spikes_unordered.png")
    plt.close(fig)

    fig, ax, im = plot_resonance_for_autapse_param(param_df, frequencies2, num_spikes_three_pulses, ['e', 'f'], title = 'Number of spikes emitted by \nthe 3-pulse protocol')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number of spikes produced")
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_resonator_num_spikes_by_e_f.png")
    plt.close(fig)

    fig, ax, im = plot_resonance_for_autapse_param(param_df, frequencies2, num_spikes_three_pulses, ['tau'], title = 'Number of spikes emitted by \nthe 3-pulse protocol')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number of spikes produced")
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_resonator_num_spikes_by_tau.png")
    plt.close(fig)


def test2_plots(dict, conf, param_df, args):
    """
    Plot the results for test 2: a chain of subthreshold pulses at fixed frequencies.
    
    """

    neuron_name = conf["Neuron"]["name"]
    output_directory = conf["Output"]["out_dir"]

    # Simulation 2 outputs
    N_pulses = int(conf["Simulation 2"]["N_pulses"])
    multipulse_freq = dict["Simulation 2"]["multipulse_freq"]
    num_spikes = dict["Simulation 2"]["num_spikes"]
    spike_freq = dict["Simulation 2"]["spike_freq"]
    spike_times = dict["Simulation 2"]["spike_times"]
    pulse_starts = dict["Simulation 2"]["pulse_starts"]

    #Spike to pulse ratios
    #fig5, ax5 = plot_spike_to_pulse_ratio(multipulse_freq, spike_to_pulse_ratio)
    #if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_spike_pulse.png")

    # spike to pulse frequency (gives a sense of how many pulses are needed per spike.)
    #fig6, ax6 = plot_spike_to_pulse_frequency(multipulse_freq, spike_to_pulse_frequency)
    #if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_spike_freq.png")

    fig, ax = pulses_to_spike(spike_times, pulse_starts, multipulse_freq)
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_pulsesToSpike.png")
    plt.close(fig)

    fig, ax = subsequent_spikes(spike_times, pulse_starts, multipulse_freq)
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_subSpikes_unordered.png")
    plt.close(fig)

    fig, ax = subsequent_spikes_by_params(spike_times, pulse_starts, multipulse_freq, param_df, ["e", "f"])
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_subSpikes_by_e_f.png")
    plt.close(fig)

    fig, ax = subsequent_spikes_by_params(spike_times, pulse_starts, multipulse_freq, param_df, ["f", "e"])
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_subSpikes_by_f_e.png")
    plt.close(fig)

    fig, ax = subsequent_spikes_by_params(spike_times, pulse_starts, multipulse_freq, param_df, ["tau"])
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_subSpikes_by_tau.png")
    plt.close(fig)

    fig, ax = plot_total_spikes(num_spikes, multipulse_freq, param_df, param_order = ["e", "f"])
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_totalSpikes_by_e_f.png")
    plt.close(fig)

    fig, ax = plot_total_spikes(num_spikes, multipulse_freq, param_df, param_order = ["f", "e"])
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_totalSpikes_by_f_e.png")
    plt.close(fig)

    fig, ax = plot_total_spikes(num_spikes, multipulse_freq, param_df, param_order = ["tau"])
    if args.save: plt.savefig(f"{output_directory}\\{neuron_name}_totalSpikes_by_tau.png")
    plt.close(fig)
    

def plot_subset(args, conf):
    """Plot traces from each trial and select neurons.
    
    Plot example traces... Need select frequencies... Maybe input all these things in 
    the config file??
    
    """

    indices = np.array(conf["Subset"]["indices"].split(", "), dtype = np.int64)
    plot_freq = np.array(conf["Subset"]["frequencies"].split(", "), dtype = np.float64)

    # Get the data file.
    output_directory = conf["Output"]["out_dir"]
    output_data_file = conf["Output"]["data_file"]
    out_filename = output_data_file[:-7] + "_subset" + output_data_file[-7:]
    output_path = output_directory + "\\" + out_filename

    with open(output_path, 'rb') as f:          # open the data generated from subset
        subset_dict = pickle.load(f)

    print(subset_dict.keys())

    with open(output_directory + "\\" + output_data_file, 'rb') as f:          # open data generated from main simulation
        main_dict = pickle.load(f)

    params = subset_dict["Parameters"]

    freq_test1 = main_dict["Simulation 1"]["frequencies"]
    freq_3pulses = main_dict["Three pulses"]["frequencies"]
    freq_test2 = main_dict["Simulation 2"]["multipulse_freq"]

    plot_all_indices(subset_dict, "Simulation 1", params, indices, plot_freq, freq_test1, output_directory)
    plot_all_indices(subset_dict, "Three pulses", params, indices, plot_freq, freq_3pulses, output_directory)
    plot_all_indices(subset_dict, "Simulation 2", params, indices, plot_freq, freq_test2, output_directory)


def plot_all_indices(dict, trial_label, params, indices, plot_freq, ref_freq, out_dir):
    """
    Loops over the indices and plots the membrane potential versus injected
    
    """
    N_neurons = np.shape(params)[0]

    X = dict[trial_label]["X"]
    T = dict[trial_label]["T"]
    I = dict[trial_label]["I"]


    for n, i in enumerate(indices):
        # i is the neuron index
        e = params[i]["e"]
        f = params[i]["f"]
        tau = params[i]["tau"]

        for m, freq in enumerate(plot_freq):
            # f is the value of the plotting frequency
            # find the index and value of the frequency closest to the desired frequency to plot
            freq_diff = np.abs(ref_freq - freq)
            freq_idx = np.argmin(freq_diff)
            freq_actual = ref_freq[freq_idx]

            row_idx = N_neurons * freq_idx + i
            fig, ax = plot_potential_versus_injected(X[row_idx], T, I[row_idx])
            fig.suptitle("Membrane potential versus Injected Current", fontweight = 'bold', y = 0.95)
            ax[0].set_title(f"autapse: e = {e}, f = {f}, \u03c4 = {tau}; frequency = {freq_actual}", fontsize = 10)
            plt.savefig(out_dir + f"\\{trial_label.replace(" ", "_")}_idx{i}_f{freq}.png")

