"""

Perform some kind of gri-search on parameter values around RS neurons.

 (1) will start with basic RS_integrator and RS_resonator parameter values

    - for each neuron type, vary the autapse parameters 
            e between 0.5 and 0.1 (2ms and 10ms decay)
            f between 50 to 250 pA
            tau between 0-4 ms
    - can calculate the input-output gain (slope of F-I curve)
            also the instantaneous I/O gain.



"""


import sys
sys.path.append("../")

# local imports
from batchAQUA_general import batchAQUA
from AQUA_general import AQUA
from stimulus import *
from plotting_functions import *

# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq




RS_int = {'name': 'RS_integrator', 'C': 100, 'k': 0.7, 'v_r': -60, 'v_t': -40, 'v_peak': 35,
     'a': 0.03, 'b': -2, 'c': -50, 'd': 100, 'e': 0., 'f': 0., 'tau': 0.}

RS_middle = {'name': 'RS_middle', 'C': 100, 'k': 0.7, 'v_r': -60, 'v_t': -40, 'v_peak': 35,
     'a': 0.03, 'b': 0.5, 'c': -50, 'd': 100, 'e': 0., 'f': 0., 'tau': 0.}

RS_res = {'name': 'RS_resonator', 'C': 100, 'k': 0.7, 'v_r': -60, 'v_t': -40, 'v_peak': 35,
     'a': 0.03, 'b': 5, 'c': -50, 'd': 100, 'e': 0., 'f': 0., 'tau': 0.}







neuron_params = [RS_int, RS_middle, RS_res]

def main():
    
    T = 500    # ms
    dt = 0.1    # ms
    N_iter = int(T/dt)


    """ prep simulation parameter values"""
    # autapse values
    e_vals = 1/np.linspace(2, 10, 5)   # 10 values btw 2-10 ms
    f_vals = np.linspace(50, 450, 20)
    tau_vals = np.linspace(0, 4, 5)
    print(tau_vals)
    N_neurons = len(e_vals)*len(f_vals)*len(tau_vals) + 1

    N_I = 100     # the number of current steps tested

    N_sims = N_neurons * N_I
    print(N_sims)

    I_range = np.linspace(50, 500, N_I)
    delay = 100     # ms, delay before step current onset
    # total number of injected currents is N_sims
    I_inj = np.array([step_current(N_iter, dt, 0., delay, i) for i in I_range for n in range(N_neurons)])
    print(np.shape(I_inj))

    # loop over neuron parameters
    for param in neuron_params:

        # create parameter dataframe and initial conditions
        x_start = np.full((N_sims, 3), np.array([-60, 0., 0.]))
        t_ini = np.zeros(N_sims)
        
        params = []
        for i in range(N_sims):
            params.append(param)

        params_df = pd.DataFrame(params)

        print(params_df.head())
        print(params_df.count())






if __name__ == "__main__":
    main()