"""

- - - DEFAULT RESEARCH TEMPLATE - - -

    - - - simulation.py


template to evaluate different metrics over a grid of autapse parameters

The exact metric and analysis is flexible.

Will make the process of creating a large batch of neurons automatic and the resulting
simulations relatively fast.

"""


from aqua.AQUA_general import AQUA
from aqua.batchAQUA_general import *
from aqua.plotting_functions import *
from aqua.stimulus import step_current, filtered_white_noise_fast
from aqua.utils import STA

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from brian2 import *
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
    print("- - - SIMULATION - - -")
    # neuron params to be used for every simulation below
    params = cast_to_float(conf["Neuron"])

    # extract autapse params
    if conf["Autapse"]["mode"] == 'explicit':     # mode = 'explicit' - autapse parameters are pre-defined and comma-separated
        f_vals   = np.array(conf["Autapse"]["f"].split(", "), dtype = np.float64)
        e_vals   = np.array(conf["Autapse"]["e"].split(", "), dtype = np.float64)
        tau_vals = np.array(conf["Autapse"]["tau"].split(", "), dtype = np.float64)
    elif conf["Autapse"]["mode"] == 'uniform':    # mode = 'uniform' - the range is supplied and the number of params
        # 0 - start value, 1 - end value, 2 - number of samples
        f_arr = np.array(conf["Autapse"]["f"].split(", "), dtype = np.float64)
        f_vals = np.linspace(f_arr[0], f_arr[1], int(f_arr[2]))
        e_arr = np.array(conf["Autapse"]["e"].split(", "), dtype = np.float64)
        e_vals = np.linspace(e_arr[0], e_arr[1], int(e_arr[2]))
        tau_arr = np.array(conf["Autapse"]["tau"].split(", "), dtype = np.float64)
        tau_vals = np.linspace(tau_arr[0], tau_arr[1], int(tau_arr[2]))
    elif conf["Autapse"]["mode"] == 'normal':       # mode = 'normal' - params sampled from a normal distribution with mean, std, N_samples
        # 0 - mean, 1 - std, 2 - N_samples
        f_arr = np.array(conf["Autapse"]["f"].split(", "), dtype = np.float64)
        f_vals = np.random.normal(f_arr[0], f_arr[1], int(f_arr[2]))
        e_arr = np.array(conf["Autapse"]["e"].split(", "), dtype = np.float64)
        e_vals = np.random.normal(e_arr[0], e_arr[1], int(e_arr[2]))
        tau_arr = np.array(conf["Autapse"]["tau"].split(", "), dtype = np.float64)
        tau_vals = np.random.normal(tau_arr[0], tau_arr[1], int(tau_arr[2]))
    elif conf["Autapse"]["mode"] == 'log-normal':       # mode = 'log-normal' - sample from log-normal distribution with mean, std, N_samples
        # need to convert mean and std to those of the underlying normal distribution
        # f
        f_arr = np.array(conf["Autapse"]["f"].split(", "), dtype = np.float64)
        f_mean = np.log(f_arr[0]**2/ (np.sqrt(f_arr[0]**2 + f_arr[1]**2)))
        f_std = np.log(1 + (f_arr[1]**2/f_arr[0]**2))
        f_vals = np.random.Generator.lognormal(f_mean, f_std, int(f_arr[2]))
        # e
        e_arr = np.array(conf["Autapse"]["e"].split(", "), dtype = np.float64)
        e_mean = np.log(e_arr[0]**2/ (np.sqrt(e_arr[0]**2 + e_arr[1]**2)))
        e_std = np.log(1 + (e_arr[1]**2/e_arr[0]**2))
        e_vals = np.random.Generator.lognormal(e_mean, e_std, int(e_arr[2]))
        # tau
        tau_arr = np.array(conf["Autapse"]["tau"].split(", "), dtype = np.float64)
        tau_mean = np.log(tau_arr[0]**2/ (np.sqrt(tau_arr[0]**2 + tau_arr[1]**2)))
        tau_std = np.log(1 + (tau_arr[1]**2/tau_arr[0]**2))
        tau_vals = np.random.Generator.lognormal(tau_mean, tau_std, int(tau_arr[2]))

    
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
    gain_modulation(params_df, conf)

    # Test 2 - gain modulation on biexponential autapse in brian2
    #out_df = gain_modulation_biexponential(params_df, conf)

    # Test 3 - STA
    #calculate_STA(params_df, conf)





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
    print("- - - GAIN MODULATION- - -")

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
    cols = ['e', 'f', 'tau', 'autapse current', 'autapse delay', 'I_h', 'F_instant', 'F_steady']
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

    # save the results dict as a pickle
    name = conf['Neuron']['name']
    mode = conf['Autapse']['mode']
    file_sign = conf['Gain']['outfile']
    filepath = f"{name}//{name}_{mode}_{file_sign}"
    with open(filepath, 'wb') as file:
        pickle.dump(output_df, file)



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

    print("- - - GAIN MODULATION: BIEXPONENTIAL - - -")

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
    # Need to scale up parameter dict to match N_sims
    sim_params = pd.DataFrame(data = [], columns = params_df.keys())
    for i in range(conf["Gain"]["N_I"]):
        sim_params = pd.concat([sim_params, params_df], ignore_index = True)

    # biexponential autapse - fix rise time
    t_a1_arr = np.ones(N_sims)
    t_a2_arr = 1/np.array(sim_params['e'])
    t_a2_arr[t_a2_arr == np.inf] = 2.       # remove infinities where there is no autapse

    print("- - - biexponential - - -")
    print(t_a2_arr)
    
    # somewhere to store the outputs
    cols = ['e', 'f', 'tau', 'autapse current', 'autapse delay', 'I_h', 'F_instant', 'F_steady']
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

        # create batch and initialise
        batch = batchAQUA(sim_params[idx_start:idx_end])
        batch.Initialise(x_start, t_start)


        # convert injected currents to brian2
        I_injTA = TimedArray(values = I_inj[idx_start:idx_end, :].T, dt = dt*ms, name = 'I_injTA')    # inputs as a TimedArray

        # convert batch to brian2
        G, autapses = batch.meetBrian(stimulus_name = I_injTA, biexponential = True, t_a1 = t_a1_arr[idx_start:idx_end], t_a2 = t_a2_arr[idx_start:idx_end])

        # simulation timestep
        defaultclock.dt = dt*ms
        M_v = StateMonitor(G, 'v', record = 0)
        M_w = StateMonitor(G, 'w', record = 0)
        spikemon = SpikeMonitor(G, record = True)
        net = Network(G, autapses, M_v, M_w, spikemon)

        # run simulation
        net.run(T*ms)

        spikes = convert_spikes_to_aqua(spikemon.spike_trains())

        """ - - - from this point analyse from spike times and start building output df - - - """
        # quantifying autapse values -> don't correspond to biexponential autapse but still differentiate all neurons...
        autapse_current = list(batch.get_net_autapse_currents())
        autapse_delay = list(batch.get_mean_autapse_delays())

        # brian2 output needs to be converted here...
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

    # save the results dict as a pickle
    name = conf['Neuron']['name']
    mode = conf['Autapse']['mode']
    file_sign = conf['Gain']['outfile']
    filepath = f"{name}//{name}_{mode}_{file_sign}"
    with open(filepath, 'wb') as file:
        pickle.dump(output_df, file)


def calculate_STA(params, conf):
    """
    Calculate the STA for each neuron and autapse combination.
    Calculated from a filtered white noise input.

    """

    conf['STA'] = cast_to_float(conf['STA'])
    T = conf['STA']['T']        # ms
    dt = conf['STA']['dt']      # ms
    N_iter = int(T/dt)          # number of iterations

    # define the injected currents...
    N_neurons = len(params)                                 # number of neurons
    N_F = int(conf['STA']['N_F'])                                # number of frequency filters
    N_per_F = int(conf['STA']['N_per_F'])                          # number of repetitions of each filter
    F_start = conf['STA']['F_start']                        # lowest frequency filter
    F_stop = conf['STA']['F_stop']                          # highest frequency filter
    filter_freq = np.linspace(F_start, F_stop, N_F)   # actual frequencies
    amplitude = conf['STA']['amplitude']                    # amplitude of the signal
    window = int(conf['STA']['window'])


    batch_thresh = batchAQUA(params)     # create a batch with only the non-autaptic neuron
    thresh, steady_state = batch_thresh.get_threshold(idx = 0)
    print("- - - - - - -")
    print(thresh)
    print(steady_state)

    STA_dict = {}
    # Loop over each frequency for memory reasons
    for f in filter_freq:
        print(f"freq: {f}")
        # get the filtered white noise
        N_sims = N_per_F * N_neurons
        print(f"N_sims: {N_sims}")
        I_noise = np.array([filtered_white_noise_fast(T/1000., dt, amplitude = amplitude, cutoff = f) for i in range(N_per_F)])     # very slow
        I_fwn = np.zeros((N_sims, N_iter))
        
        # Need to scale up parameter dict to match N_sims
        sim_params = pd.DataFrame(data = [], columns = params.keys())
        for i in range(N_per_F):
            I_fwn[i*N_neurons:(i+1)*N_neurons] = I_noise[i]
            sim_params = pd.concat([sim_params, params], ignore_index = True)

        I_inj = thresh + I_fwn      # bring to threshold

        # store the data
        sta = np.zeros((N_sims, window))

        # Need to break up the simulation
        N_per_loop = int(conf['STA']['N_per_loop'])
        N_loops = N_sims // N_per_loop
        for n in range(N_loops):
            if n == N_loops - 1:
                N_in_loop = N_sims - (N_loops - 1)*N_per_loop
            else:
                N_in_loop = N_per_loop

            # get proper indices
            idx_start = n * N_per_loop
            idx_end = idx_start + N_in_loop

            # create batch
            batch = batchAQUA(sim_params[idx_start:idx_end])
            # initialise
            x_start = np.full((N_in_loop, 3), fill_value = steady_state)
            t_start = np.zeros(N_in_loop)
            batch.Initialise(x_start, t_start)

            _, _, spikes = batch.update_batch(dt, N_iter, I_inj[idx_start:idx_end])

            # calculate the STA
            sta[idx_start:idx_end] = STA(spikes, I_inj[idx_start:idx_end], dt, window = window)
        # average the STA for each neuron
        neuron_dict = {}
        for i in range(N_neurons):
            sta_mean = np.mean(sta[i::N_neurons], axis = 0)
            neuron_dict[i] = sta_mean
        
        STA_dict[f] = neuron_dict
    
    # save the results dict as a pickle
    name = conf['Neuron']['name']
    mode = conf['Autapse']['mode']
    file_sign = conf['STA']['outfile']
    filepath = f"{name}//{name}_{mode}_{file_sign}"
    with open(filepath, 'wb') as file:
        pickle.dump(STA_dict, file)

