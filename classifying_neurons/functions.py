### General functions to simulate simple tests for the Hodgkin classification of neurons. 

import sys
sys.path.append("..\\")
from AQUA_general import AQUA
from batchAQUA_general import batchAQUA
from stimulus import *
from plotting_functions import *


import numpy as np
import matplotlib.pyplot as plt



def FI_from_steps(batch, I_Range, N_iter, dt):
    # will setup an array of step current according to the range above. 
    # Assumes that the batch of neurons has been initialised.
    N_models = batch.N_models

    I_0 = 0.0
    I_delay =  500 # ms
    I_h = np.linspace(I_Range[0], I_Range[1], N_models) # range of step heights
    I_inj = np.array([step_current(N_iter, dt, I_0, I_delay, i) for i in I_h])

    # X has shape (N_models, 3, N_iter)
    X, T, spikes = batch.update_batch(dt, N_iter, I_inj)

    # Determine the FI curve. 
    freq = 1000/np.diff(spikes, axis = 1)

    steady_state = np.zeros(N_models)
    for i in range(N_models):
        mean_val = np.mean(freq[i, ~np.isnan(freq[i])][-10:])
        if ~np.isnan(mean_val):
            steady_state[i] = mean_val
    
    # Need to plot the steady-state frequency versus injected current.
    return I_h, steady_state


def instant_FI_from_steps(batch, I_Range, N_iter, dt):
    # will setup an array of step current according to the range above. 
    # Assumes that the batch of neurons has been initialised.
    N_models = batch.N_models

    I_0 = 0.0
    I_delay =  500 # ms
    I_h = np.linspace(I_Range[0], I_Range[1], N_models) # range of step heights
    I_inj = np.array([step_current(N_iter, dt, I_0, I_delay, i) for i in I_h])

    # X has shape (N_models, 3, N_iter)
    X, T, spikes = batch.update_batch(dt, N_iter, I_inj)

    # Determine the FI curve. 
    freq = 1000/np.diff(spikes, axis = 1)

    instant_freq = np.zeros(N_models)
    for i in range(N_models):
        val = np.mean(freq[i, ~np.isnan(freq[i])][:1])
        if ~np.isnan(val):
            instant_freq[i] = val
    
    # Need to plot the steady-state frequency versus injected current.
    return I_h, instant_freq


def FI_from_ramps(neuron, I_range, N_iter, dt, delay = 500):
    """
    apply a slow ramp stimulus to a neuron, only single neuron needed.
    only apply to the first neuron in the batch for efficiency.

    returns the instantaneous frequency during the trial and the corresponding injected current.
    """

    I_0 = I_range[0]    # start value
    I_delay = delay       # ms
    I_f = I_range[1]    # end value

    I_inj = np.array(ramp(N_iter, dt, I_0, I_delay, I_f))

    X, T, spikes = neuron.update_RK2(dt, N_iter, I_inj)    # assumes neuron is already initialised.

    freq = 1000/np.diff(spikes)
    I_spike = I_inj[(spikes[:-1]/dt).astype(int)]

    return I_spike, freq, X, T, I_inj


    




