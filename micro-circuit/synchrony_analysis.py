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
import pickle
import gc
import tracemalloc
from scipy.signal import convolve, windows

from FT_metrics import *

# run everything on GPU
# import brian2cuda
# set_device("cuda_standalone")



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

def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plt.plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plt.plot([0, 1], [i, j], '-k')
    plt.xticks([0, 1], ['Source', 'Target'])
    plt.ylabel('Neuron index')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-1, max(Ns, Nt))
    plt.subplot(122)
    plt.plot(S.i, S.j, 'ok')
    plt.xlim(-1, Ns)
    plt.ylim(-1, Nt)
    plt.xlabel('Source neuron index')
    plt.ylabel('Target neuron index')
    plt.show()


def main():

    # DRIVING CURRENTS
    INPUT_E1 = np.linspace(120, 280, 20)# 15
    INPUT_E2 = 150

    # SYNAPSE STRENGTH
    E1_TO_E2 = np.linspace(20, 100, 10)# 10   
    E2_TO_E1 = np.linspace(20, 100, 10)# 10


    # simulate the autaptic network
    simulate(E1_neuron, E2_neuron, I_neuron, INPUT_E1, INPUT_E2, E1_TO_E2, E2_TO_E1, "synch_analysis_aut.pickle")

    # simulate the non-autaptic network
    simulate(E2_neuron, E2_neuron, I_neuron, INPUT_E1, INPUT_E2, E1_TO_E2, E2_TO_E1, "synch_analysis_naut.pickle")

    # Can now simulate networks with different autapse types.
    # Also need to decide autapse delivery mode...






def simulate(E1_neuron, E2_neuron, I_neuron, INPUT_E1, INPUT_E2, E1_TO_E2, E2_TO_E1, outfile, autapse_type = 'standard', t1 = None, t2 = None, I_peak = None):
    """
    Quick analysis over parameters to see if the autapse extends the range of synchrony.

    Vary driving current into autaptic neuron and weight to/from the autaptic neuron.

    Calculate some simple measures of synchrony and return it all in a dataframe.


    CAN PROBABLY RUN EVERYTHING IN ONE GO AND USE GPU OPTIMIZATION THIS WAY TOO!
    
    """
    start_scope()
    tracemalloc.start()
    snap1 = tracemalloc.take_snapshot()

    ### Store the simulation parameters below. 
    N_SIMS = len(INPUT_E1) * len(E1_TO_E2) * len(E2_TO_E1)

    #  INHIBITORY PARAMETERS
    THRESHOLD_OFFSET = 0
    W_MAX = 100

    # simulation parameters
    T = 5000 # ms
    dt = 0.1
    N_iter = int(T/dt)

    
    ''' - - - define the excitatory populations - - - '''
    # neuron parameters, 2 populations for each neuron...
    params_E1 = [E1_neuron for _ in range(N_SIMS)]      # 2 neurons
    params_E2 = [E2_neuron for _ in range(N_SIMS)]      # 2 neurons


    x_start = np.full(shape = (N_SIMS, 3), fill_value = np.array([-60, 0, 0]))
    t_start = np.zeros(N_SIMS)

    # create the batch E1
    batch_E1 = batchAQUA(params_E1)
    batch_E1.Initialise(x_start, t_start)

    # create the batch E1
    batch_E2 = batchAQUA(params_E2)
    batch_E2.Initialise(x_start, t_start)

    # create the input current - STEP CURRENT
    N_at_each_current = N_SIMS // len(INPUT_E1)       # number of simulations at each current (number of total weight combinations)
    I1 = np.array([i * np.ones((N_at_each_current, N_iter)) for i in INPUT_E1])        # stronger driving current to E1s
    I1 = I1.reshape(-1, N_iter)

    # same driving current for all E2
    I2 = INPUT_E2 * np.ones((N_SIMS, N_iter))        # weaker current to E2

    # create a Timed Arrays
    I1_TA = TimedArray(values = I1.T, dt = dt*ms, name = 'I1_TA')   
    I2_TA = TimedArray(values = I2.T, dt = dt*ms, name = 'I2_TA')

    # convert to brian2 with the standard autapse model
    E1, aut_E1 = batch_E1.meetBrian(stimulus_name = I1_TA, synapse_eq = syn_eq, autapse_type = autapse_type, t_a1 = t1, t_a2 = t2, I_peak = I_peak)
    E2, aut_E2 = batch_E2.meetBrian(stimulus_name = I2_TA, synapse_eq = syn_eq)     # no autapse (defaults to standard)


    ''' - - - define the inhibitory neuron - - - '''
    param_I = [I_neuron for _ in range(N_SIMS)]
    x_start = np.array([[-60, 0, 0]])
    t_start = np.array([0.])

    # create batch 
    batch_I = batchAQUA(param_I)
    batch_I.Initialise(x_start, t_start)

    # input current will be just subthreshold
    threshold, _ = batch_I.get_threshold(idx = 0)
    I_inh = np.array((threshold - THRESHOLD_OFFSET)*np.ones((N_SIMS, N_iter)))
    I_inhTA = TimedArray(values = I_inh.T, dt = dt*ms, name = 'I_inhTA')

    # create brian objects, no effective autapse here.
    I, aut_I = batch_I.meetBrian(stimulus_name = I_inhTA, synapse_eq = syn_eq)

    ''' - - - CREATE SYNAPSES - - - '''
    """ - - STDP - - """
    w_max = W_MAX # maximum allowed current through an inhibitory synapse
    taupre = taupost = 20 * ms
    dApre = 20      # the maximum change in the weight in one step
    dApost = -dApre * (taupre/taupost) * 1.05

    '''- - exc. synapses - -'''
    # fully connect excitatory neurons (ignoring autapses)
    syn_E1 = Synapses(E1, E2, 
                model = model_exc,
                on_pre = syn_on_pre_exc,
                method = 'rk2')
    syn_E2 = Synapses(E2, E1, 
                model = model_exc,
                on_pre = syn_on_pre_exc,
                method = 'rk2')
    
    syn_E1.connect(condition = 'i == j')     # 1-1 connection
    syn_E2.connect(condition = 'i == j')     # 1-1 connection

    ## Set exc. synapse variables here...
    E1.Syn_exc = 0            # pA
    E2.Syn_exc = 0            # pA
    E1.t_exc = 5              # ms
    E2.t_exc = 5              # ms
    I.t_exc = 5               # ms

    E1.t_inh = 5              # ms
    E2.t_inh = 5              # ms
    I.t_inh = 5               # ms


    N_w = len(E1_TO_E2) * len(E2_TO_E1)
    N_e1 = len(E1_TO_E2)
    for l in range(len(INPUT_E1)):
        for m, w1 in enumerate(E1_TO_E2):
            for n, w2 in enumerate(E2_TO_E1):
                idx = l * N_w + m * N_e1 + n
                syn_E1.w_exc[idx, idx] = w1    # pA, weight from E1 -> E2
                syn_E2.w_exc[idx, idx] = w2    # pA, weight from E2 -> E1


    ''' - - E1 and E2 to I synapses - - '''
    syn_E1_I = Synapses(E1, I,
                model = model_stdp,
                on_pre = on_pre_stdp_exc,
                on_post = on_post_stdp,
                method = 'rk2')
    syn_E1_I.connect(condition = 'i == j')         # both excitatory neurons connect to I

    syn_E2_I = Synapses(E2, I,
                model = model_stdp,
                on_pre = on_pre_stdp_exc,
                on_post = on_post_stdp,
                method = 'rk2')
    syn_E2_I.connect(condition = 'i == j')         # both excitatory neurons connect to I

    ## set inh. synapse variables for post-syn population
    syn_E2_I.w_exc[:, :] = 50   # pA, weight from I -> E1
    syn_E2_I.w_exc[:, :] = 50   # pA, weight from I -> E2


    ''' - - I to E1 and E2 synapses - - '''
    syn_I_E1 = Synapses(I, E1,
                model = model_stdp,
                on_pre = on_pre_stdp_exc,
                on_post = on_post_stdp,
                method = 'rk2')
    syn_I_E1.connect(condition = 'i == j')         # both excitatory neurons connect to I

    syn_I_E2 = Synapses(I, E2,
                model = model_stdp,
                on_pre = on_pre_stdp_exc,
                on_post = on_post_stdp,
                method = 'rk2')
    syn_I_E2.connect(condition = 'i == j')         # both excitatory neurons connect to I

    ## set inh. synapse variables for post-syn population
    syn_I_E2.w_exc[:, :] = 50   # pA, weight from I -> E1
    syn_I_E2.w_exc[:, :] = 50   # pA, weight from I -> E2

    snap2 = tracemalloc.take_snapshot()
    stats = snap2.compare_to(snap1, 'lineno')
    print(f'Before Network Creation: {stats[0].size_diff / 10**6:.2f} MB')

    ''' - - simulation - - '''
    # set simulation parameters
    defaultclock.dt = dt*ms
    # Monitors

    # Monitors for the autaptic network
    M_v_E1 = StateMonitor(E1, ['v', 'Syn_exc', 'Syn_inh', 'w'], record = True)
    M_v_E2 = StateMonitor(E2, ['v', 'Syn_exc', 'Syn_inh', 'w'], record = True)
    M_v_I = StateMonitor(I, ['v', 'Syn_exc', 'Syn_inh'], record = True)
    spikemon_E1 = SpikeMonitor(E1, record = True)
    spikemon_E2 = SpikeMonitor(E2, record = True)
    spikemon_I = SpikeMonitor(I, record = True)

    
    # M_syn_EI_aut = StateMonitor(syn_EI, 'w_exc', record = True)
    # M_syn_IE_aut = StateMonitor(syn_IE, 'w_inh', record = True)
    # M_syn_E = StateMonitor(syn_E, 'Syn_exc', record = True)   # record the current through the synapse
    # M_syn_IE = StateMonitor(syn_IE, 'Syn_inh', record = True)   
    # M_syn_EI = StateMonitor(syn_EI, 'Syn_exc', record = True)   

    # create networks
    net = Network(E1, E2, I, aut_E1, aut_E2, aut_I, syn_E1, syn_E2, syn_E1_I, syn_I_E1, syn_E2_I, syn_I_E2, 
                    M_v_E1, M_v_E2, M_v_I, spikemon_E1, spikemon_E2, spikemon_I) 
    
    snap2 = tracemalloc.take_snapshot()
    stats = snap2.compare_to(snap1, 'lineno')
    print(f'Before Simulation: {stats[0].size_diff / 10**6:.2f} MB')
    net.run(T*ms)
    snap2 = tracemalloc.take_snapshot()
    stats = snap2.compare_to(snap1, 'lineno')
    print(f'After Simulation: {stats[0].size_diff / 10**6:.2f} MB')

    ''' - - - CALCULATE METRICS - - - '''

    cols = ['I_inj', 'w_e1_e2', 'w_e2_e1', 'FT_distance', 'ISI_distance', 'SPIKE_distance', 'SPIKE_synchrony', 'spike_directionality']
    results = pd.DataFrame(columns = cols)

    # set values from the simulation
    results['I_inj'] = I1[:, 0]                   # Current into E1
    results['w_e1_e2'] = syn_E1.w_exc[:]         # Synapse weight from e1 to e2
    results['w_e2_e1'] = syn_E2.w_exc[:]         # Synapse weight from e2 to e1


    #visualise_connectivity(syn_E1_I_naut)

    #visualise_connectivity(syn_I_E2_naut)

    ## Get spike trains
    spike_train_E1 = spikemon_E1.spike_trains()
    spike_train_E2 = spikemon_E2.spike_trains()
    #spike_train_I = spikemon_I_aut.spike_trains()


    # convert to aqua spikes
    spikes_E1 = convert_spikes_to_aqua(spike_train_E1)
    spikes_E2 = convert_spikes_to_aqua(spike_train_E2)
    #spikes_I = convert_spikes_to_aqua(spike_train_I)



    ''' - - FT metric - - '''
    bin_E1 = binarise_spikes(spikes_E1, dt, N_iter)
    bin_E2 = binarise_spikes(spikes_E2, dt, N_iter)

    # filter - can vary the std for different measures
    gauss = windows.gaussian(M = 10000, std = 100)
    gauss /= gauss.sum()

    edges = [0, T]     # edges for pyspike

    # store the metrics
    FT_dist = np.zeros(N_SIMS)
    ISI_dist = np.zeros(N_SIMS)
    SPIKE_dist = np.zeros(N_SIMS)
    SPIKE_synch = np.zeros(N_SIMS)
    spike_directionality = np.zeros(N_SIMS)

    for i in range(N_SIMS):

        '''FT distance'''
        # calculate FFT
        _, freq = calculate_FT(bin_E1[0, :], dt, gauss)
        fft_E1, _ = calculate_FT(bin_E1[i, :], dt, gauss)
        fft_E2, _ = calculate_FT(bin_E2[i, :], dt, gauss)

        FT_dist[i] = calculate_FT_diff(fft_E1, fft_E2, freq)

        '''- - PYSPIKE metrics - - '''
        # create pyspike spike_trains
        spk_E1 = spk.SpikeTrain(spikes_E1[i, :], edges)
        spk_E2 = spk.SpikeTrain(spikes_E2[i, :], edges)

        '''- - ISI distance - -'''
        ISI_dist[i] = spk.isi_profile(spk_E1, spk_E2).avrg()

        '''- - SPIKE distance - -'''
        SPIKE_dist[i] = spk.spike_profile(spk_E1, spk_E2).avrg()

        '''- - SPIKE synchrony - -'''
        SPIKE_synch[i] = spk.spike_sync_profile(spk_E1, spk_E2).avrg()

        '''- - SPIKE directionality - -'''
        spike_directionality[i] = spk.spike_directionality(spk_E1, spk_E2)

    # append to dataframes
    results['FT_distance'] = FT_dist
    results['ISI_distance'] = ISI_dist
    results['SPIKE_distance'] = SPIKE_dist
    results['SPIKE_synchrony'] = SPIKE_synch
    results['spike_directionality'] = spike_directionality



    with open(outfile, 'wb') as file:
        pickle.dump(results, file)

    snap2 = tracemalloc.take_snapshot()
    stats = snap2.compare_to(snap1, 'lineno')
    print(f'Before Deletion: {stats[0].size_diff / 10**6:.2f} MB')
    gc.collect()
    snap2 = tracemalloc.take_snapshot()
    stats = snap2.compare_to(snap1, 'lineno')
    print(f'After Deletion: {stats[0].size_diff / 10**6:.2f} MB')


    

if __name__ == "__main__":
    main()