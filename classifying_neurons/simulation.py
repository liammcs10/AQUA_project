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


#local imports
import CLI
import CF



def sim(args, conf):
    """
    Will generate the firing pattern transition for the input neuron
    based on different autaptic parameter values.
    
    Arguments
    ----------
    config  :   str
                config file name
    out     :   str
                output filename
    save    :   action
                whether to save the output plots
    
    Parameters
    ----------
    args :      arguments passed to main in terminal

    conf :      configuration dict from CF.read_conf()


    Output
    ---------
    dataframe :  saved as pickle file. 
                 contains run params (I_inj, e, f, tau, ISI_values, ISI_counts)
 
    
    """


    # Define the neuron
    T = float(conf['Simulation']['t']) # needs to be in ms
    dt = float(conf['Simulation']['dt'])
    N_iter = int(T/dt) # number of iterations
    
    #injected current 
    I_height = 10
    I_inj = I_height*np.ones(N_iter)

    # starting values will stay fixed
    x_start = np.array([float(conf['Neuron']['c']), 0, 0])
    t_start = np.array([0])

    ## Arrays of autaptic parameters
    e_vals = np.arange(float(conf['Autapse']['e_start']), 
                       float(conf['Autapse']['e_stop']), 
                       float(conf['Autapse']['de']))
    
    f_vals = np.arange(float(conf['Autapse']['f_start']), 
                       float(conf['Autapse']['f_stop']), 
                       float(conf['Autapse']['df']))
    
    tau_vals = np.arange(float(conf['Autapse']['tau_start']), 
                         float(conf['Autapse']['tau_stop']), 
                         float(conf['Autapse']['dtau']))

    max_val = len(e_vals)*len(f_vals)*len(tau_vals)
    # data array to be save in a dataframe
    cols = ['I_inj', 'e', 'f', 'tau', 'ISI_values', 'ISI_counts']
    data_array = np.zeros((1, len(cols)))

    with tqdm(total = max_val, desc = 'Progress') as pbar:
        for e in e_vals:
            for f in f_vals:
                for tau in tau_vals:
                    neuron = AQUA(float(conf['Neuron']['a']),
                                float(conf['Neuron']['b']),
                                float(conf['Neuron']['c']),
                                float(conf['Neuron']['d']),
                                e, f, tau) # autapse params
                    
                    neuron.Initialise(x_start, t_start)

                    _, _, spikes = neuron.update_RK2(dt, N_iter, I_inj)

                    ISI = np.ediff1d(spikes[-int(0.5*len(spikes)):])
                    isi_vals, isi_counts = np.unique(np.round(ISI, decimals = 4), return_counts = True)

                    for j in range(len(isi_vals)):
                        df_row = np.array([I_height, e, f, tau, isi_vals[j], isi_counts[j]])
                        data_array = np.vstack((data_array, df_row))
                
                    pbar.update(1)
                    
    
    data_array = np.delete(data_array, 0, axis = 0)
    df = pd.DataFrame(data = data_array, columns = cols)
    print(df)
    with open(str(conf['Output']['pickle_file']), "wb") as filename:
        pickle.dump(df, filename)

    return df