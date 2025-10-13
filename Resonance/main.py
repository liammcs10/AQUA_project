"""
Perform a large parameter search over the space of autapse parameters.


"""

import sys
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


    if args.sim:    # Run the whole simulation if --sim is passed
        print("Running simulation...")
        
        # check for a config file
        if args.config is None:
            print("No config file passed... exiting")
            quit()

        # check for an output file
        if args.outfile is None:
            print("No data file given... exiting")
            quit()

        # extract params from config file and store in dictionary
        conf = CF.read_conf(args.config)
        print("Config extracted...")
        df = simulation.sim(args, conf)
        print("Simulation complete...")
    
    
    # import data
    with open(args.outfile, 'rb') as f:
        dict = pickle.load(f)
    # neurons tested
    parameters = dict["Parameters"]
    df = pd.Dataframe(parameters)
    
    # Simulation 1 outputs
    frequencies = dict["Simulation 1"]["frequencies"]
    bands = dict["Simulation 1"]["bands"]

    # Simulation 2 outputs
    multipulse_freq = dict["Simulation 2"]["multipulse_freq"]
    num_spikes = dict["Simulation 2"]["num_spikes"]
    spike_freq = dict["Simulation 2"]["spike_freq"]

    ## Now plot the data...
    fig1, ax1 = plot_resonance_map(frequencies, bands)
    plt.show()
    

main()