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
import plotting
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
    plot    :   action
                whether to generate plots
    subset  :   str
                string with indices of subset of params, separated by a comma.
    config  :   str
                config file name
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
    if args.sim and not args.subset:    # Run the whole simulation if --sim is passed, but not --subset
        print("Running simulation...")
        simulation.sim(args, conf)
        print("Simulation complete.")
    

    """- - - - PLOTTING - - - -"""
    if args.plot:
        print("Plotting outputs...")
        plotting.plot(args, conf)
        print("Plotting complete.")

    """Re-run on subset of parameters"""
    if args.subset:
        print("Extracting subset...")
        if args.sim:
            simulation.sim(args, conf, SUBSET = True)

        plotting.plot_subset(args, conf)



main()