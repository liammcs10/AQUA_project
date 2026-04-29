''' import aqua '''
from aqua.batchAQUA_general import batchAQUA
from aqua.AQUA_general import AQUA
from aqua.utils import * 
from aqua.plotting_functions import *

'''general imports''' 
import numpy as np
import pandas as pd
from brian2 import *
import matplotlib.pyplot as plt
import seaborn as sns
import pyspike as spk
from scipy.signal import convolve, windows

from FT_metrics import *


I_neuron = {'name': 'FS', 'C': 20, 'k': 1, 'v_r': -55, 'v_t': -40, 'v_peak': 25,
     'a': 0.2, 'b': -2, 'c': -45, 'd': 0, 'e': 0.2, 'f': 0., 'tau': 0.}

# strong autaptic neuron on RS resonator...
E1_neuron = {'name': 'RS', 'C': 100, 'k': 0.7, 'v_r': -60, 'v_t': -40, 'v_peak': 35,
     'a': 0.03, 'b': 5, 'c': -50, 'd': 100, 'e': 0.2, 'f': 250., 'tau': 0.}     # instantaneous autapse bc all synapses are instant.

# non-autaptic neuron - RS resonator
E2_neuron = {'name': 'RS', 'C': 100, 'k': 0.7, 'v_r': -60, 'v_t': -40, 'v_peak': 35,
     'a': 0.03, 'b': 5, 'c': -50, 'd': 100, 'e': 0., 'f': 0., 'tau': 0.}


'''- - - model equations - - - '''

syn_eq = """
    dSyn_exc/dt = -(Syn_exc/t_exc)/ms : 1 
    t_exc : 1
    dSyn_inh/dt = -(Syn_inh/t_inh)/ms : 1 
    t_inh : 1
    g_total = Syn_exc + Syn_inh : 1 

"""

model_exc = '''w_exc : 1'''
model_inh = '''w_inh : 1'''
syn_on_pre_exc = '''Syn_exc += w_exc'''     # excitatory presynaptic neuron
syn_on_pre_inh = '''Syn_inh += w_inh'''     # inhibitory presynaptic neuron


model_stdp = '''
w_inh : 1
w_exc : 1
dApre/dt = -Apre / taupre : 1 (event-driven)
dApost/dt = -Apost / taupost : 1 (event-driven)
'''
# if pre- is inhibitory
on_pre_stdp_inh = '''
Syn_inh += w_inh
Apre += dApre
w_inh = clip(w_inh + Apost, -w_max, 0)
''' 
# if pre- is excitatory
on_pre_stdp_exc = '''
Syn_exc += w_exc
Apre += dApre
w_exc = clip(w_exc + Apost, 0, w_max)
''' 
on_post_stdp = '''
Apost += dApost
w_inh = clip(w_inh + Apre, 0, w_max)
w_exc = clip(w_exc + Apre, 0, w_max)
'''



def main():
    """
    Quick analysis over parameters to see if the autapse extends the range of synchrony.

    Vary driving current into autaptic neuron and weight to/from the autaptic neuron.

    Calculate some simple measures of synchrony and return it all in a dataframe.


    CAN PROBABLY RUN EVERYTHING IN ONE GO AND USE GPU OPTIMIZATION THIS WAY TOO!
    
    """

    ### Store the simulation parameters below. 

    # DRIVING CURRENTS
    INPUT_E1 = np.linspace(120, 280, 20)
    INPUT_E2 = 150

    # SYNAPSE STRENGTH
    E1_TO_E2 = np.linspace(20, 100, 10)   
    E2_TO_E1 = np.linspace(20, 100, 10)

    #  INHIBITORY PARAMETERS
    THRESHOLD_OFFSET = 0
    W_MAX = 100

    # simulation parameters
    T = 5000 # ms
    dt = 0.1
    N_iter = int(T/dt)

