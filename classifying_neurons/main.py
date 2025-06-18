"""
Implementing test to characterise autaptic neuron's computational properties.

"""



import sys
sys.path.append("..\\") # parent directory containing AQUA, etc...
from AQUA_class import AQUA
from batchAQUA import batchAQUA
from stimulus import *
from plotting_functions import *
from functions import *
import CLI
import CF


import numpy as np
import matplotlib.pyplot as plt


def main():
    # Simulation protocol

    # import neuron parameters.
    args = CLI.command_line()
    
    if args.sim: # run the simulation if --sim is passed
        if args.config is None:
            print("No config file passed... exiting")
            quit()
        # get parameters from configuration file.
        conf = CF.read_conf(args.config)

        df = simulation.sim(args, conf) # DataFrame saved as pickle and returned
    
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





if __name__ == '__main__':
    main()

