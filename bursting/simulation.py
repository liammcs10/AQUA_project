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


from AQUA_general import AQUA
from batchAQUA_general import batchAQUA
from plotting_functions import *
from stimulus import step_current

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
    print(bool(conf["Autapse"]["define_explicit"]) == 'False')
    if conf["Autapse"]["define_explicit"] == 'True':     # if True, the specific autapse parameter values were given
        f_vals   = np.array(conf["Autapse"]["f"].split(", "), dtype = np.float64)
        e_vals   = np.array(conf["Autapse"]["e"].split(", "), dtype = np.float64)
        tau_vals = np.array(conf["Autapse"]["tau"].split(", "), dtype = np.float64)
    else:
        # 0 - start value, 1 - end value, 2 - number of samples
        f_arr = np.array(conf["Autapse"]["f"].split(", "), dtype = np.float64)
        f_vals = np.linspace(f_arr[0], f_arr[1], int(f_arr[2]))
        e_arr = np.array(conf["Autapse"]["e"].split(", "), dtype = np.float64)
        e_vals = np.linspace(e_arr[0], e_arr[1], int(e_arr[2]))
        tau_arr = np.array(conf["Autapse"]["tau"].split(", "), dtype = np.float64)
        tau_vals = np.linspace(tau_arr[0], tau_arr[1], int(tau_arr[2]))


    # calculate the number of neurons
    N_neurons = 1 + len(f_vals) * len(e_vals) * len(tau_vals)
    print(f"N_neurons: {N_neurons}")
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
    params_df = pd.DataFrame(params_arr)
    print(f"params_df is {sys.getsizeof(params_df)/1000} kBytes")
    # We won't define the batch just yet as it may be more useful to have 

    """ RUN THE ANALYSES BELOW - define functions at the end of this script/in a different script"""
    
    # Test 1 - gain modulation on the AQUA batch
    out_df = gain_modulation(params_df, conf)

    # Test 2 - gain modulation on biexponential autapse in brian2
    

    # Test 3


    # At this stage just save the outputs

    # save the results dict as a pickle
    with open(args.outfile, 'wb') as file:
        pickle.dump(out_df, file)




""" - - - - SIMULATION FUNCTIONS - - - - """



def gain_modulation(params_df, conf):
    """
    Analysis of the effect of the autapse on gain modulation of the neuron. Step currents
    Injected and either the instantaneous or steady state beahviours are extracted.

    params:
        params_df:      DataFrame containing all autapse parameters
        conf:           parsed config file containing simulation info.
        instant:        whether to take instantaneous response.

    """
    print("- - - GAIN MODULATION - - -")

    # convert config values to float
    conf["Gain"] = cast_to_float(conf["Gain"])
    conf["Gain"]["N_I"] = int(conf["Gain"]["N_I"])
    conf["Gain"]["N_per_loop"] = int(conf["Gain"]["N_per_loop"])

    N_per_loop = conf["Gain"]["N_per_loop"]     # number of neurons per loop (to address memory issues)
    N_neurons = len(params_df)                  # number of different neuron parameters
    
    # time
    T = float(conf["Gain"]["T"])
    dt = float(conf["Gain"]["dt"])
    N_iter = int(T/dt)

    # range of injected currents values
    I_range = np.linspace(conf["Gain"]["I_start"], conf["Gain"]["I_stop"], conf["Gain"]["N_I"])
    # build injected current array
    delay = conf["Gain"]["delay"]
    y_0 = conf["Gain"]["y_0"]
    I_inj = np.array([step_current(N_iter, dt, y_0, delay, I_h) for I_h in I_range for n in range(N_neurons)])

    # number of simulations that ultimately need to be run
    N_sims = N_neurons * conf["Gain"]["N_I"]
    print(f"N_sims: {N_sims}")
    # Need to scale up parameter dict
    sim_params = pd.DataFrame(data = [], columns = params_df.keys())
    for i in range(conf["Gain"]["N_I"]):
        sim_params = pd.concat([sim_params, params_df], ignore_index = True)


    # somewhere to store the outputs
    cols = ['e', 'f', 'tau', 'I_h', 'F_instant', 'F_steady']
    output_df = pd.DataFrame(data = [], columns = cols)     # will be a list of dictionaries

    # start looping over the simulations
    N_loops = N_sims // N_per_loop
    for n in range(N_loops):
        if n == N_loops - 1:
            N_in_loop = N_sims - (N_loops - 1)*N_per_loop
        else:
            N_in_loop = N_per_loop

        # get proper indices
        idx_start = n * N_per_loop
        idx_end = idx_start + N_in_loop

        # initialise
        x_start = np.full((N_in_loop, 3), fill_value = np.array([conf["Neuron"]["c"], 0, 0]))
        t_start = np.zeros(N_in_loop)

        # create batch
        batch = batchAQUA(sim_params[idx_start:idx_end])
        batch.Initialise(x_start, t_start)
        # simulate
        _, _, spikes = batch.update_batch(dt, N_iter, I_inj[idx_start:idx_end, :])

        """ - - - from this point analyse from spike times and start building output df - - - """
        # quantifying autapse values
        autapse_current = list(batch.get_net_autapse_currents())
        autapse_delay = list(batch.get_mean_autapse_delays())

        F_instant = get_F(spikes, instant = True)
        F_steady = get_F(spikes, instant = False)

        out_dict = {"e":   list(sim_params['e'][idx_start:idx_end]),
                    "f":   list(sim_params['f'][idx_start:idx_end]),
                    "tau": list(sim_params['tau'][idx_start:idx_end]),
                    "autapse current": autapse_current,
                    "autapse delay": autapse_delay,
                    "I_h": list(I_inj[idx_start:idx_end, -1]),
                    "F_instant": list(F_instant),
                    "F_steady": list(F_steady)
                    }


        small_df = pd.DataFrame(out_dict)   # convert dictionary to DataFrame
        output_df = pd.concat([output_df, small_df])


    return output_df



def gain_modulation_biexponential(params_df, conf):
    """ 
    Same as the function above except that here we will make use of the 
    brian2 model with a biexponential autapse function. 

    params_df needs to be updated to include

    params:
        params_df:      DataFrame containing all autapse parameters
        conf:           parsed config file containing simulation info.
        instant:        whether to take instantaneous response.


    """




""" - - - - HELPER FUNCTIONS - - - - """

def get_F(spikes, instant = False):
    """
    Returns an array of the desired firing frequency
    
    :param spikes: array of spike times
    :param instant: boolean, whether to get instantaneous firing frequency or not

    """
    N_neurons = len(spikes)
    F = np.zeros(N_neurons)

    for n in range(N_neurons):
        if np.isnan(spikes[n]).all() or np.sum(~np.isnan(spikes[n])) <= 3:      # if no spikes or 1 spike
            F[n] = np.nan
        else:
            if instant:     # get instant firing frequency
                F[n] = 1000/(spikes[n][1] - spikes[n][0])           # first and second spikes
            else:           # get steady firing frequency (might be same as initial)
                spike_times = spikes[n][~np.isnan(spikes[n])]
                n_spikes = len(spike_times)
                ceil = np.ceil(n_spikes/2)
                freq = 1000/(np.ediff1d(spike_times[-int(ceil):]))
                F[n] = np.max(freq)     # largest firing frequency in the steady-state

    return F


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