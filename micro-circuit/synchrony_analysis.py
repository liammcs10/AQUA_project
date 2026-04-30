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
    """
    Quick analysis over parameters to see if the autapse extends the range of synchrony.

    Vary driving current into autaptic neuron and weight to/from the autaptic neuron.

    Calculate some simple measures of synchrony and return it all in a dataframe.


    CAN PROBABLY RUN EVERYTHING IN ONE GO AND USE GPU OPTIMIZATION THIS WAY TOO!
    
    """

    ### Store the simulation parameters below. 

    # DRIVING CURRENTS
    INPUT_E1 = np.linspace(120, 280, 14)
    INPUT_E2 = 150

    # SYNAPSE STRENGTH
    E1_TO_E2 = np.linspace(20, 100, 10)   
    E2_TO_E1 = np.linspace(20, 100, 10)

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
    params_E2 = [E2_neuron for _ in range(N_SIMS)]


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
    E1_aut, aut_E1 = batch_E1.meetBrian(stimulus_name = I1_TA, synapse_eq = syn_eq)
    E2_aut, aut_E2 = batch_E2.meetBrian(stimulus_name = I2_TA, synapse_eq = syn_eq)
    
    # create the non-autaptic version of the network
    E1_naut, naut_E1 = batch_E1.meetBrian(stimulus_name = I1_TA, synapse_eq = syn_eq)
    E2_naut, naut_E2 = batch_E2.meetBrian(stimulus_name = I2_TA, synapse_eq = syn_eq)


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
    I_aut, aut_I = batch_I.meetBrian(stimulus_name = I_inhTA, synapse_eq = syn_eq)
    I_naut, naut_I = batch_I.meetBrian(stimulus_name = I_inhTA, synapse_eq = syn_eq)

    ''' - - - CREATE SYNAPSES - - - '''
    """ - - STDP - - """
    w_max = W_MAX # maximum allowed current through an inhibitory synapse
    taupre = taupost = 20 * ms
    dApre = 20      # the maximum change in the weight in one step
    dApost = -dApre * (taupre/taupost) * 1.05

    '''- - exc. (aut) synapses - -'''
    # fully connect excitatory neurons (ignoring autapses)
    syn_E1_aut = Synapses(E1_aut, E2_aut, 
                model = model_exc,
                on_pre = syn_on_pre_exc,
                method = 'rk2')
    syn_E2_aut = Synapses(E2_aut, E1_aut, 
                model = model_exc,
                on_pre = syn_on_pre_exc,
                method = 'rk2')
    
    syn_E1_aut.connect(condition = 'i == j')     # fully connected minus autapses
    syn_E2_aut.connect(condition = 'i == j')     # fully connected minus autapses

    ## Set exc. synapse variables here...
    E1_aut.Syn_exc = 0            # pA
    E2_aut.Syn_exc = 0            # pA
    E1_aut.t_exc = 5              # ms
    E2_aut.t_exc = 5              # ms
    I_aut.t_exc = 5               # ms
    I_naut.t_exc = 5              # ms

    E1_aut.t_inh = 5              # ms
    E2_aut.t_inh = 5              # ms
    I_aut.t_inh = 5               # ms
    I_naut.t_inh = 5              # ms

    for n in range(N_SIMS):
        for w1 in E1_TO_E2:
            for w2 in E2_TO_E1:
                syn_E1_aut.w_exc[n, n] = w1    # pA, weight from E1 -> E2
                syn_E2_aut.w_exc[n, n] = w2    # pA, weight from E2 -> E1

    '''- - exc. (naut) synapses - - '''
    # fully connect excitatory neurons (ignoring autapses)
    syn_E1_naut = Synapses(E1_naut, E2_naut, 
                model = model_exc,
                on_pre = syn_on_pre_exc,
                method = 'rk2')
    syn_E2_naut = Synapses(E2_naut, E1_naut, 
                model = model_exc,
                on_pre = syn_on_pre_exc,
                method = 'rk2')
    
    syn_E1_naut.connect(condition = 'i == j')     # fully connected minus autapses
    syn_E2_naut.connect(condition = 'i == j')     # fully connected minus autapses

    ## Set exc. synapse variables here...
    E1_naut.Syn_exc = 0            # pA
    E2_naut.Syn_exc = 0            # pA
    E1_naut.t_exc = 5              # ms
    E2_naut.t_exc = 5              # ms

    E1_naut.t_inh = 5              # ms
    E2_naut.t_inh = 5              # ms

    for n in range(N_SIMS):
        for w1 in E1_TO_E2:
            for w2 in E2_TO_E1:
                syn_E1_naut.w_exc[n, n] = w1    # pA, weight from E1 -> E2
                syn_E2_naut.w_exc[n, n] = w2    # pA, weight from E2 -> E1

    ''' - - E1 and E2 to I synapses (aut.) - - '''
    syn_E1_I_aut = Synapses(E1_aut, I_aut,
                model = model_stdp,
                on_pre = on_pre_stdp_exc,
                on_post = on_post_stdp,
                method = 'rk2')
    syn_E1_I_aut.connect(condition = 'i == j')         # both excitatory neurons connect to I

    syn_E2_I_aut = Synapses(E2_aut, I_aut,
                model = model_stdp,
                on_pre = on_pre_stdp_exc,
                on_post = on_post_stdp,
                method = 'rk2')
    syn_E2_I_aut.connect(condition = 'i == j')         # both excitatory neurons connect to I

    ## set inh. synapse variables for post-syn population
    syn_E2_I_aut.w_exc[:, :] = 50   # pA, weight from I -> E1
    syn_E2_I_aut.w_exc[:, :] = 50   # pA, weight from I -> E2

    ''' - - E1 and E2 to I synapses (Naut.) - - '''
    syn_E1_I_naut = Synapses(E1_naut, I_naut,
                model = model_stdp,
                on_pre = on_pre_stdp_exc,
                on_post = on_post_stdp,
                method = 'rk2')
    syn_E1_I_naut.connect(condition = 'i == j')         # both excitatory neurons connect to I

    syn_E2_I_naut = Synapses(E2_naut, I_naut,
                model = model_stdp,
                on_pre = on_pre_stdp_exc,
                on_post = on_post_stdp,
                method = 'rk2')
    syn_E2_I_naut.connect(condition = 'i == j')         # both excitatory neurons connect to I

    ## set inh. synapse variables for post-syn population
    syn_E2_I_naut.w_exc[:, :] = 50   # pA, weight from I -> E1
    syn_E2_I_naut.w_exc[:, :] = 50   # pA, weight from I -> E2


    ''' - - I to E1 and E2 synapses (aut.) - - '''
    syn_I_E1_aut = Synapses(I_aut, E1_aut,
                model = model_stdp,
                on_pre = on_pre_stdp_exc,
                on_post = on_post_stdp,
                method = 'rk2')
    syn_I_E1_aut.connect(condition = 'i == j')         # both excitatory neurons connect to I

    syn_I_E2_aut = Synapses(I_aut, E2_aut,
                model = model_stdp,
                on_pre = on_pre_stdp_exc,
                on_post = on_post_stdp,
                method = 'rk2')
    syn_I_E2_aut.connect(condition = 'i == j')         # both excitatory neurons connect to I

    ## set inh. synapse variables for post-syn population
    syn_I_E2_aut.w_exc[:, :] = 50   # pA, weight from I -> E1
    syn_I_E2_aut.w_exc[:, :] = 50   # pA, weight from I -> E2


    ''' - - I to E1 and E2 synapses (Naut.) - - '''
    syn_I_E1_naut = Synapses(I_naut, E1_naut,
                model = model_stdp,
                on_pre = on_pre_stdp_exc,
                on_post = on_post_stdp,
                method = 'rk2')
    syn_I_E1_naut.connect(condition = 'i == j')         # both excitatory neurons connect to I

    syn_I_E2_naut = Synapses(I_naut, E2_naut,
                model = model_stdp,
                on_pre = on_pre_stdp_exc,
                on_post = on_post_stdp,
                method = 'rk2')
    syn_I_E2_naut.connect(condition = 'i == j')         # both excitatory neurons connect to I

    ## set inh. synapse variables for post-syn population
    syn_I_E2_naut.w_exc[:, :] = 50   # pA, weight from I -> E1
    syn_I_E2_naut.w_exc[:, :] = 50   # pA, weight from I -> E2



    ''' - - simulation - - '''
    # set simulation parameters
    defaultclock.dt = dt*ms
    # Monitors

    # Monitors for the autaptic network
    M_v_E1_aut = StateMonitor(E1_aut, ['v', 'Syn_exc', 'Syn_inh', 'w'], record = True)
    M_v_E2_aut = StateMonitor(E2_aut, ['v', 'Syn_exc', 'Syn_inh', 'w'], record = True)
    M_v_I_aut = StateMonitor(I_aut, ['v', 'Syn_exc', 'Syn_inh'], record = True)
    spikemon_E1_aut = SpikeMonitor(E1_aut, record = True)
    spikemon_E2_aut = SpikeMonitor(E2_aut, record = True)
    spikemon_I_aut = SpikeMonitor(I_aut, record = True)

    # Monitors for the NON-autaptic network
    M_v_E1_naut = StateMonitor(E1_naut, ['v', 'Syn_exc', 'Syn_inh', 'w'], record = True)
    M_v_E2_naut = StateMonitor(E2_naut, ['v', 'Syn_exc', 'Syn_inh', 'w'], record = True)
    M_v_I_naut = StateMonitor(I_naut, ['v', 'Syn_exc', 'Syn_inh'], record = True)
    spikemon_E1_naut = SpikeMonitor(E1_naut, record = True)
    spikemon_E2_naut = SpikeMonitor(E2_naut, record = True)
    spikemon_I_naut = SpikeMonitor(I_naut, record = True)
    
    # M_syn_EI_aut = StateMonitor(syn_EI, 'w_exc', record = True)
    # M_syn_IE_aut = StateMonitor(syn_IE, 'w_inh', record = True)
    # M_syn_E = StateMonitor(syn_E, 'Syn_exc', record = True)   # record the current through the synapse
    # M_syn_IE = StateMonitor(syn_IE, 'Syn_inh', record = True)   
    # M_syn_EI = StateMonitor(syn_EI, 'Syn_exc', record = True)   

    # create networks
    net_aut = Network(E1_aut, E2_aut, I_aut, aut_E1, aut_E2, aut_I, syn_E1_aut, syn_E2_aut, syn_E1_I_aut, syn_I_E1_aut, syn_E2_I_aut, syn_I_E2_aut, 
                    M_v_E1_aut, M_v_E2_aut, M_v_I_aut, spikemon_E1_aut, spikemon_E2_aut, spikemon_I_aut) 
    
    net_naut = Network(E1_naut, E2_naut, I_naut, naut_E1, naut_E2, naut_I, syn_E1_naut, syn_E2_naut, syn_E1_I_naut, syn_I_E1_naut, syn_E2_I_naut, syn_I_E2_naut, 
                    M_v_E1_naut, M_v_E2_naut, M_v_I_naut, spikemon_E1_naut, spikemon_E2_naut, spikemon_I_naut) 

    net_aut.run(T*ms)

    net_naut.run(T*ms)

    ''' - - - CALCULATE METRICS - - - '''

    cols = ['I_inj', 'w_e1_e2', 'w_e2_e1', 'ft_distance', 'ISI_distance', 'SPIKE_distance', 'SPIKE_synchrony', 'spike_directionality']
    results_aut = pd.DataFrame(columns = cols)
    results_naut = pd.DataFrame(columns = cols)

    # set values from the simulation
    results_aut['I_inj'] = I1[:, 0]                   # Current into E1
    results_naut['I_Inj'] = I1[:, 0]   

    results_aut['w_e1_e2'] = syn_E1_aut.w_exc[:]         # Synapse weight from e1 to e2
    results_naut['w_e1_e2'] = syn_E1_naut.w_exc[:]

    results_aut['w_e2_e1'] = syn_E2_aut.w_exc[:]         # Synapse weight from e2 to e1
    results_naut['w_e2_e1'] = syn_E2_naut.w_exc[:]


    visualise_connectivity(syn_E1_I_naut)

    visualise_connectivity(syn_I_E2_naut)

    ## Get spike trains
    spike_train_E1_aut = spikemon_E1_aut.spike_trains()
    spike_train_E2_aut = spikemon_E2_aut.spike_trains()
    #spike_train_I_aut = spikemon_I_aut.spike_trains()

    spike_train_E1_naut = spikemon_E1_naut.spike_trains()
    spike_train_E2_naut = spikemon_E2_naut.spike_trains()
    #spike_train_I_naut = spikemon_I_naut.spike_trains()

    # convert to aqua spikes
    spikes_E1_aut = convert_spikes_to_aqua(spike_train_E1_aut)
    spikes_E2_aut = convert_spikes_to_aqua(spike_train_E2_aut)
    #spikes_I_aut = convert_spikes_to_aqua(spike_train_I_aut)

    spikes_E1_naut = convert_spikes_to_aqua(spike_train_E1_naut)
    spikes_E2_naut = convert_spikes_to_aqua(spike_train_E2_naut)
    #spikes_I_naut = convert_spikes_to_aqua(spike_train_I_naut)

    ''' - - FT metric - - '''
    bin_E1_aut = binarise_spikes(spikes_E1_aut, dt, N_iter)
    bin_E2_aut = binarise_spikes(spikes_E2_aut, dt, N_iter)

    bin_E1_naut = binarise_spikes(spikes_E1_naut, dt, N_iter)
    bin_E2_naut = binarise_spikes(spikes_E2_naut, dt, N_iter)

    # filter - can vary the std for different measures
    gauss = windows.gaussian(M = 10000, std = 100)
    gauss /= gauss.sum()

    edges = [0, T]     # edges for pyspike

    # store the metrics
    FT_dist_aut = np.zeros(N_SIMS)
    ISI_dist_aut = np.zeros(N_SIMS)
    SPIKE_dist_aut = np.zeros(N_SIMS)
    SPIKE_synch_aut = np.zeros(N_SIMS)
    spike_directionality_aut = np.zeros(N_SIMS)

    FT_dist_naut = np.zeros(N_SIMS)
    ISI_dist_naut = np.zeros(N_SIMS)
    SPIKE_dist_naut = np.zeros(N_SIMS)
    SPIKE_synch_naut = np.zeros(N_SIMS)
    spike_directionality_naut = np.zeros(N_SIMS)

    for i in range(N_SIMS):

        '''FT distance'''
        # calculate FFT
        _, freq = calculate_FT(bin_E1_aut[0, :], dt, gauss)
        fft_E1_aut = calculate_FT(bin_E1_aut[i, :], dt, gauss)[0]
        fft_E2_aut = calculate_FT(bin_E2_aut[i, :], dt, gauss)[0]

        fft_E1_naut = calculate_FT(bin_E1_naut[i, :], dt, gauss)[0]
        fft_E2_naut = calculate_FT(bin_E2_naut[i, :], dt, gauss)[0]

        FT_dist_aut[i] = calculate_FT_diff(fft_E1_aut, fft_E2_aut, freq)
        FT_dist_naut[i] = calculate_FT_diff(fft_E1_naut, fft_E2_naut, freq)

        '''- - PYSPIKE metrics - - '''
        # create pyspike spike_trains
        spk_E1_aut = spk.SpikeTrain(spikes_E1_aut, edges)
        spk_E2_aut = spk.SpikeTrain(spikes_E2_aut, edges)

        spk_E1_naut = spk.SpikeTrain(spikes_E1_naut, edges)
        spk_E2_naut = spk.SpikeTrain(spikes_E2_naut, edges)

        '''- - ISI distance - -'''
        ISI_dist_aut[i] = spk.isi_profile(spk_E1_aut, spk_E2_aut).avrg()
        ISI_dist_naut[i] = spk.isi_profile(spk_E1_naut, spk_E2_naut).avrg()

        '''- - SPIKE distance - -'''
        SPIKE_dist_aut[i] = spk.spike_profile(spk_E1_aut, spk_E2_aut).avrg()
        SPIKE_dist_naut[i] = spk.spike_profile(spk_E1_naut, spk_E2_naut).avrg()

        '''- - SPIKE synchrony - -'''
        SPIKE_synch_aut[i] = spk.spike_sync_profile(spk_E1_aut, spk_E2_aut).avrg()
        SPIKE_synch_naut[i] = spk.spike_sync_profile(spk_E1_naut, spk_E2_naut).avrg()

        '''- - SPIKE directionality - -'''
        spike_directionality_aut[i] = spk.spike_sync_profile(spk_E1_aut, spk_E2_aut).avrg()
        spike_directionality_naut[i] = spk.spike_sync_profile(spk_E1_naut, spk_E2_naut).avrg()

    # append to dataframes
    results_aut['FT_distance'] = FT_dist_aut 
    results_aut['ISI_distance'] = ISI_dist_aut 
    results_aut['SPIKE_distance'] = SPIKE_dist_aut 
    results_aut['SPIKE_synchrony'] = SPIKE_synch_aut 
    results_aut['spike directionality'] = spike_directionality_aut

    results_naut['FT_distance'] = FT_dist_naut 
    results_naut['ISI_distance'] = ISI_dist_naut 
    results_naut['SPIKE_distance'] = SPIKE_dist_naut 
    results_naut['SPIKE_synchrony'] = SPIKE_synch_naut 
    results_naut['spike directionality'] = spike_directionality_naut


    with open("synch_analysis_aut.pickle", 'wb') as file:
        pickle.dump(results_aut, file)

    with open("synch_analysis_naut.pickle", 'wb') as file:
        pickle.dump(results_naut, file)
    

if __name__ == "__main__":
    main()