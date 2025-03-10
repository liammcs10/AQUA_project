
## The FS neuron undergoes bifurcations as the size of the current step is increased
# Here we run the AQUA simulations for each injected current step in the hopes it runs faster

# For step current between 1 to 25
# get steady-state frequencies and plot them. 
# Lines representing the transitions are useful to visualise.


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from AQUA_class import AQUA
from plotting_functions import *



def bifurcation_simulation():
    # output dataframe
    cols = ["I_inj", "frequency"]
    df = pd.DataFrame(columns = cols)

    # Neuron setup

    # FS neuron
    a = 0.1
    b = 0.2
    c = -65
    d = 2

    #autaptic parameters
    e = 0.14    # Bacci et al. 2003
    f = -4      # negative for inhibitory interneuron
    tau = 10    # ms (Bacci et al. 2003)

    #simulation parameters
    T = 5       # s
    dt = 0.01   # ms
    N_iter = int(T*1000/dt)

    # Initial values
    x_start = np.array([-70., 0., 0.])
    t_start = np.array([0.])

    neuron = AQUA(a, b, c, d, e, f, tau)

    #Initialization variables, X_start must contain 3-elements
    x_start = np.array([-70, -14, 0]) # [v(0), u(0), w(0)]
    t_start = np.array([0])

    # Run again bu start from 0
    I_heights = np.arange(0, 25.5, 0.5)

    for i in I_heights:

        # Re-initialise for each run.
        neuron.Initialise(x_start, t_start)

        # Define step current
        I_inj = np.concatenate([np.zeros(int(0.20*N_iter)), i*np.ones(int(0.80*N_iter))])    # injected current.

        #Run simulation on step current...
        X, T_aut, spikes_autapse = neuron.update_RK2(dt, N_iter, I_inj)
        
        # Save the spike times and calc frequencies
        freq_autapse = 1000/(np.ediff1d(spikes_autapse))

        # get the unique frequencies of the last 500 spikes.
        unique_vals = np.unique(np.round(freq_autapse[int(-0.2*len(freq_autapse)):], decimals = 4))

        for val in unique_vals:
            temp_df = pd.DataFrame(data = [[i, val]], columns = cols)
            df = pd.concat([df, temp_df])
    
    return df

        