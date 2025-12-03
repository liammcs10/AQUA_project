"""
Docstring for AQUAmeetBrian

Generate the necessary elements to run the AQUA model in Brian2. Will predominantly
make use of the brian2 interface. This class is aimed at setting up 

"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from brian2 import *

EQS = '''
dv/dt = ((1/C)*(k *(v-v_r)*(v-v_t) - u + I))/ms : 1
du/dt = (a * (b*(v-v_r) - u))/ms : 1
dw/dt = (-e_a*w)/ms : 1
C : 1
k : 1
v_r : 1
v_t : 1
I : 1
a : 1
b : 1
c : 1
d : 1
e_a : 1
f : 1
'''

RESET = '''
v = c
u += d
'''

class AQUAMeetBrian:


    def __init__(self, params_df):
        """
        Docstring for __init__
        
        :param self: Description
        :param params_df: pd.DataFrame containing all neuron parameters
        :param x_ini: initial conditions, shape: (N_neurons, 3)
        :param I_inj: Driving current, shape: (N_neurons, )
        """

        # convert params_list to pandas dataframe
        if not isinstance(params_df, pd.DataFrame):
            params_df = pd.DataFrame(params_df)

        self.N_models = len(params_df)
        self.name = params_df['name'].to_numpy(dtype = str)
        self.isFS = (np.char.find(self.name, "FS")!=-1)     # bool array, where the neuron is of FS type.
        self.k = params_df['k'].to_numpy(dtype = np.float64)
        self.C = params_df['C'].to_numpy(dtype = np.float64)
        self.v_r = params_df['v_r'].to_numpy(dtype = np.float64)
        self.v_t = params_df['v_t'].to_numpy(dtype = np.float64)
        self.v_peak = params_df['v_peak'].to_numpy(dtype = np.float64)
        self.a = params_df['a'].to_numpy(dtype = np.float64)
        self.b = params_df['b'].to_numpy(dtype = np.float64)
        self.c = params_df['c'].to_numpy(dtype = np.float64)
        self.d = params_df['d'].to_numpy(dtype = np.float64)
        self.e = params_df['e'].to_numpy(dtype = np.float64)
        self.f = params_df['f'].to_numpy(dtype = np.float64)
        self.tau = params_df['tau'].to_numpy(dtype = np.float64)*ms
        #self.E_syn = np.array([p['E_syn'] for p in params_list])


    def createBrian(self, x_ini, I_inj):
        
        
        G = NeuronGroup(self.N_models, EQS, threshold = 'v >= v_peak', reset = RESET, method = 'rk2')

        # Intialise variables
        G.v = x_ini[:, 0]           # initialise membrane potential
        G.u = x_ini[:, 1]           # initialise adaptation current
        G.w = x_ini[:, 2]           # initialise autapse current
        G.C = self.C
        G.k = self.k
        G.v_r = self.v_r
        G.v_t = self.v_t
        G.a = self.a
        G.b = self.b
        G.I = I_inj        # the driving current is static in this case
        G.c = self.c
        G.d = self.d
        G.e_a = self.e
        G.f = self.f

        autapses = Synapses(G, G, on_pre = 'w += f')
        autapses.connect(condition = 'i == j')
        autapses.delay = self.tau


        return G, autapses