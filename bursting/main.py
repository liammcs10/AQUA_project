"""

- - - DEFAULT RESEARCH TEMPLATE - - -

    - - - main.py

template to evaluate different metrics over a grid of autapse parameters

The exact metric and analysis is flexible.

Will make the process of creating a large batch of neurons automatic and the resulting
simulations relatively fast.

"""


from aqua.AQUA_general import AQUA
from aqua.batchAQUA_general import batchAQUA
from aqua.plotting_functions import *
from aqua.stimulus import *

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

    # parse arguments from the command line
    args = CLI.command_line()

    if args.sim:    # Run the whole simulation if --sim is passed
        print("Running simulation...")
        
        # check for a config file
        if args.config is None:
            print("No config file passed... exiting")
            quit()

        # extract params from config file and store in dictionary
        conf = CF.read_conf(args.config)
        print("Config extracted...")
        df = simulation.sim(args, conf)
        print("Simulation complete...")
    


main()