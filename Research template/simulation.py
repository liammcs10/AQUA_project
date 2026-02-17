"""

- - - DEFAULT RESEARCH TEMPLATE - - -

    - - - simulation.py


template to evaluate different metrics over a grid of autapse parameters

The exact metric and analysis is flexible.

Will make the process of creating a large batch of neurons automatic and the resulting
simulations relatively fast.

"""




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

    if conf['define_explicit'] == True:     # if passed as True, the specific autapse parameter values were given
        f_vals   = np.array(conf["Autapse"]["f"].split(", "), dtype = np.float64)
        e_vals   = np.array(conf["Autapse"]["e"].split(", "), dtype = np.float64)
        tau_vals = np.array(conf["Autapse"]["tau"].split(", "), dtype = np.float64)
    else:
        # 0 - start value, 1 - end value, 2 - number of samples
        f_arr = np.array(conf["Autapse"]["f"].split(", "), dtype = np.float64)
        f_vals = np.linspace(f_arr[0], f_arr[1], f_arr[2])
        e_arr = np.array(conf["Autapse"]["e"].split(", "), dtype = np.float64)
        e_vals = np.linspace(e_arr[0], e_arr[1], e_arr[2])
        tau_arr = np.array(conf["Autapse"]["tau"].split(", "), dtype = np.float64)
        tau_vals = np.linspace(tau_arr[0], tau_arr[1], tau_arr[2])


    # calculate the number of neurons
    N_neurons = 1 + len(f_vals) * len(e_vals) * len(tau_vals)
    # params arr will store all the parameter dictionaries in 1 place.
    params_arr = []
    params_arr.append(params) # index 0, will be the reference non-autaptic neuron
    for f in f_vals:
        for e in e_vals:
            for tau in tau_vals:
                temp_dict = params.copy()
                temp_dict["f"] = f
                temp_dict["e"] = e
                temp_dict["tau"] = tau
                params_arr.append(temp_dict)
        
    # convert params_arr to DataFrame
    params_df = pd.DataFrame(params_df)

    # We won't define the batch just yet as it may be more useful to have 

    """ RUN THE ANALYSES BELOW - define functions at the end of this script/ in a different script"""
    
    # Test 1


    # Test 2
    

    # Test 3


    # At this stage just save the outputs
    results_dict = {"Parameters": params_arr,
                    "Simulation 1": {"frequencies": frequencies, "bands": bands},
                    "Simulation 2": {"multipulse_freq": multipulse_freq, "num_spikes": num_spikes, "spike_freq": spike_freq}}

    # save the results dict as a pickle
    with open(args.outfile, 'wb') as f:
        pickle.dump(results_dict, f)
    

""" - - - - SIMULATION FUNCTIONS - - - - """




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