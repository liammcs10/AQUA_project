## The main file for running and plotting the firing transition plots.
# append AQUA directory to sys.path
import sys
sys.path.append("..\\") # parent directory
from AQUA_class import AQUA
from plotting_functions import *

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


#local imports
import CLI
import CF
import simulation
import functions



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
    
    if args.sim: # run the simulation if --sim is passed
        if args.config is None:
            print("No config file passed... exiting")
            quit()
        # get parameters from configuration file.
        conf = CF.read_conf(args.config)

        df = simulation.sim(args, conf) # DataFrame saved as pickle and returne
    
    else: # else import the pickle filename for plotting
        if args.pickle is None:
            print("No data file passed... exiting")
            quit()
        
        with open(args.pickle, 'rb') as file:
            df = pickle.load(file)
        
    
    #print(df[df['f'] == 3.0])

    functions.plot_heatmap(df, 'e', 'f', conf['Neuron']['name'])
    functions.plot_heatmap(df, 'e', 'tau', conf['Neuron']['name'])
    functions.plot_heatmap(df, 'f', 'tau', conf['Neuron']['name'])


main()