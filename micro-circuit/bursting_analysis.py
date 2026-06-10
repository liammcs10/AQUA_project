'''
The idea here is to replicate the simulation in synchrony analysis but instead of varying external parameters
like the driving current or synaptic weights. We will fix these and only vary the autapse parameters (maybe only the peak
of the autapse current). 

We want to quantify if the autapse is bursting or not (multiple frequencies in the output), average frequency, etc...

Output metrics would be average synchrony (e.g. EMD or FT distance) and regularity (CV_isi).

GOAL: demonstrate that non-linearity due to the autapse is producing these changes in behaviour.


'''


''' import aqua '''
from aqua.batchAQUA_general import batchAQUA
from aqua.AQUA_general import AQUA
from aqua.utils import * 
from aqua.plotting_functions import *
from aqua.stimulus import OU_current

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
from scipy.stats import wasserstein_distance

from FT_metrics import *
from functions import *

# run everything on GPU
# import brian2cuda
# set_device("cuda_standalone")



I_neuron = {'name': 'FS', 'C': 20, 'k': 1, 'v_r': -55, 'v_t': -40, 'v_peak': 25,
     'a': 0.2, 'b': -2, 'c': -45, 'd': 0, 'e': 0.2, 'f': 0., 'tau': 0.}


# non-autaptic neuron - RS resonator
E_neuron = {'name': 'RS', 'C': 100, 'k': 0.7, 'v_r': -60, 'v_t': -40, 'v_peak': 35,
     'a': 0.03, 'b': 5, 'c': -50, 'd': 100, 'e': 0., 'f': 0., 'tau': 0.}


''' - - - model equations - - - '''

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

    # DRIVING CURRENTS
    INPUT_E = 200

    # SYNAPSE STRENGTH
    W = 56 

    # autapse params
    e = 0.2
    tau = 2.0
    f_vals = np.linspace(10, 450, 100)      # range of autapse parameters.

    # simulate the autaptic network identical conditions (no noise)
    print('- - SIMULATION 1 - - ')
    simulate(E_neuron, I_neuron, INPUT_E, W, e, f_vals, tau, "burst_analysis_identical.pickle")

    # simulate the autaptic network - different noise   
    print('- - SIMULATION 2 - - ')
    INPUT_NOISE = {'theta': 0.2, 'sigma': 5}
    #simulate(E_neuron, I_neuron, INPUT_E, W, e, f_vals, tau, "burst_analysis_OU_noise.pickle", INPUT_NOISE)


    # Can now simulate networks with different aut apse types.
    # Also need to decide autapse delivery mode...




def simulate(E_neuron, I_neuron, INPUT_E, W, e_val, f_vals, tau_val, outfile, INPUT_NOISE = {'theta': 0, 'sigma': 0}, autapse_type = 'standard', t1 = None, t2 = None, I_peak = None, autapse_mode = 'standard', p1 = None, p2 = None):
    """
    Quick analysis over parameters to see if the autapse extends the range of synchrony.

    Vary driving current into autaptic neuron and weight to/from the autaptic neuron.

    Calculate some simple measures of synchrony and return it all in a dataframe.


    CAN PROBABLY RUN EVERYTHING IN ONE GO AND USE GPU OPTIMIZATION THIS WAY TOO!
    
    """
    start_scope()
    tracemalloc.start()
    snap1 = tracemalloc.take_snapshot()

    # number of neurons/simulations
    N_SIMS = len(f_vals) + 1

    #  INHIBITORY PARAMETERS
    THRESHOLD_OFFSET = 0
    W_MAX = 100

    # simulation parameters
    T = 5000 # ms
    dt = 0.1
    N_iter = int(T/dt)

    
    ''' - - - define the excitatory populations - - - '''
    # neuron parameters, 2 populations for each neuron...
    params_E1 = []
    params_E1.append(E_neuron)
    for f_value in f_vals:
        temp = E_neuron.copy()
        temp['e'] = e_val
        temp['f'] = f_value
        temp['tau'] = tau_val
        params_E1.append(temp)

    E1_df = pd.DataFrame(params_E1)
    f_values = E1_df['f'].unique()        # all f values, len = N_SIMS
    params_E2 = [E_neuron for _ in range(N_SIMS)]      # 2 neurons
    E2_df = pd.DataFrame(params_E2)

    x_start = np.full(shape = (N_SIMS, 3), fill_value = np.array([-60, 0, 0]))
    t_start = np.zeros(N_SIMS)

    # create the batch E1
    batch_E1 = batchAQUA(E1_df)
    batch_E1.Initialise(x_start, t_start)

    # create the batch E1
    batch_E2 = batchAQUA(E2_df)
    batch_E2.Initialise(x_start, t_start)

    # create the input current - STEP CURRENT
    I_E1 = INPUT_E * np.ones((N_SIMS, N_iter))
    I_E2 = INPUT_E * np.ones((N_SIMS, N_iter))
    #I_E2 = 150. * np.ones((N_SIMS, N_iter))

    if INPUT_NOISE['sigma'] > 0:
        theta = INPUT_NOISE['theta'] # ms
        sigma = INPUT_NOISE['sigma']
        I_noise1 = OU_current(N_iter, dt, INPUT_E, theta, I_E1[0, :], sigma)
        I_noise2 = OU_current(N_iter, dt, INPUT_E, theta, I_E2[0, :], sigma)
        I_E1 = np.full((N_SIMS, N_iter), fill_value = I_noise1)
        I_E2 = np.full((N_SIMS, N_iter), fill_value = I_noise2)


    # create a Timed Arrays
    IE1_TA = TimedArray(values = I_E1.T, dt = dt*ms, name = 'IE1_TA')   
    IE2_TA = TimedArray(values = I_E2.T, dt = dt*ms, name = 'IE2_TA') 
    
    # convert to brian2 with the standard autapse model
    E1, aut_E1 = batch_E1.meetBrian(stimulus_name = IE1_TA, synapse_eq = syn_eq, 
                                    autapse_type = autapse_type, t_a1 = t1, t_a2 = t2, I_peak = I_peak,     # autapse waveform
                                    autapse_mode = autapse_mode, p1 = p1, p2 = p2)                          # randomise autapse delivery
    E2, aut_E2 = batch_E2.meetBrian(stimulus_name = IE2_TA, synapse_eq = syn_eq)                            # no autapse (defaults to standard)


    ''' - - - define the inhibitory neuron - - - '''
    param_I = [I_neuron for _ in range(N_SIMS)]
    x_start = np.array([[-60, 0, 0]])
    t_start = np.array([0.])

    # create batch 
    batch_I = batchAQUA(param_I)
    batch_I.Initialise(x_start, t_start)

    # input current will be just subthreshold
    #threshold, _ = batch_I.get_threshold(idx = 0)
    threshold = 71.26
    print(f"THRESHOLD = {threshold}")
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

    # set synapse strength
    syn_E1.w_exc[:, :] = W
    syn_E2.w_exc[:, :] = W


    ''' - - E1 and E2 to I synapses (adaptive) - - '''
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


    ''' - - I to E1 and E2 synapses (adaptive) - - '''
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


    ''' - - simulation - - '''
    # set simulation parameters
    defaultclock.dt = dt*ms

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
    

    net.run(T*ms)


    ''' - - - CALCULATE METRICS - - - '''

    cols = ['sim_number', 'neuron_label', 'e', 'f', 'tau', 'I_inj', 'W', 'FT_distance', 'FT_EMD', 'SPIKE_distance',
            'SPIKE_synchrony', 'spike_directionality', 'schreiber similarity', 'peak_isi', 'peak_number', 'mean', 'std', 'count_sum', 'CV_isi', 'LV']
    results_df = pd.DataFrame(columns = cols)


    ## Get spike trains
    spike_train_E1 = spikemon_E1.spike_trains()
    spike_train_E2 = spikemon_E2.spike_trains()
    #spike_train_I = spikemon_I_aut.spike_trains()

    # convert to aqua spikes
    spikes_E1 = convert_spikes_to_aqua(spike_train_E1)
    spikes_E2 = convert_spikes_to_aqua(spike_train_E2)
    #spikes_I = convert_spikes_to_aqua(spike_train_I)

    # get the isis
    isi_E1 = np.diff(spikes_E1, axis = 1)
    isi_E2 = np.diff(spikes_E2, axis = 1)

    # calculate the local variation
    local_variation_E1 = isi_local_variation(spikes_E1)
    local_variation_E2 = isi_local_variation(spikes_E2)

    ''' - - FT metric - - '''
    bin_E1 = binarise_spikes(spikes_E1, dt, N_iter)
    bin_E2 = binarise_spikes(spikes_E2, dt, N_iter)

    # filter - can vary the std for different measures
    gauss = windows.gaussian(M = 10000, std = 100)
    gauss /= gauss.sum()

    edges = [0, T]     # edges for pyspike


    # define and store the outputs in a list here
    FT_dist = []
    FT_EMD = []
    SPIKE_dist = []
    SPIKE_synch = []
    spike_directionality = []
    schreiber_sim = []
    neuron_label = []
    e_lst = []
    f_lst = []
    tau_lst = []
    I_inj_lst = []
    W_lst = []
    simulation_number = []
    peak_isi = []
    peak_num = []
    mean = []
    std = []
    count_sum = []
    CV_isi = []
    LV = []

    ''' - - - SAVE MEMBRANE TRACES - - - '''
    trace_dict = {}
    trace_dict['e'] = e_val
    trace_dict['f'] = f_values
    trace_dict['tau'] = tau_val
    trace_dict['time'] = M_v_E1.t/ms
    trace_dict['X_E1'] = M_v_E1.v
    trace_dict['X_E2'] = M_v_E2.v
    trace_dict['I_E1'] = I_E1
    trace_dict['I_E2'] = I_E2
    trace_dict['spikes_E1'] = spikes_E1
    trace_dict['spikes_E2'] = spikes_E2

    # SAVE TO PICKLE
    filename = outfile[:-7] + 'TRACES' + outfile[-7:]
    with open(filename, 'wb') as file:
        pickle.dump(trace_dict, file)


    for i in range(N_SIMS):     # loop over each neuron setting

        '''FT distance'''
        # calculate FFT with no filter
        _, freq = calculate_FT(bin_E1[0, :], dt)
        n_freq = len(freq)//2
        fft_E1, _ = calculate_FT(bin_E1[i, :], dt)
        fft_E2, _ = calculate_FT(bin_E2[i, :], dt)

        # calculate the distance metrics once here. 
        ''' custom FT distance '''
        FT_distance = calculate_FT_diff(fft_E1, fft_E2, freq)
        ''' Earth mover's distance '''
        FT_wasserstein = wasserstein_distance(freq[:n_freq], freq[:n_freq], np.abs(fft_E1[:n_freq]), np.abs(fft_E2[:n_freq]))

        '''- - PYSPIKE metrics - - '''
        # create pyspike spike_trains
        spk_E1 = spk.SpikeTrain(spikes_E1[i, :], edges)
        spk_E2 = spk.SpikeTrain(spikes_E2[i, :], edges)

        '''- - SPIKE distance - -'''
        SPIKE_distance = spk.spike_profile(spk_E1, spk_E2).avrg()
        '''- - SPIKE synchrony - -'''
        SPIKE_synchrony = spk.spike_sync_profile(spk_E1, spk_E2).avrg()
        '''- - SPIKE directionality - -'''
        spike_direct = spk.spike_directionality(spk_E1, spk_E2)
        '''- - Schreiber Similarity - -'''
        schreiber = schreiber_similarity(bin_E1[i], bin_E2[i], sigma = 15, dt = dt)

        '''- - ISI histogram metrics - -'''
        bins = 100
        x_range = (0, 150)
    
        counts_E1, bin_edges_E1 = np.histogram(isi_E1[i, :], bins = bins, range = x_range)
        counts_E2, bin_edges_E2 = np.histogram(isi_E2[i, :], bins = bins, range = x_range)
        results_E1 = analyze_isi_peaks(counts_E1, bin_edges_E1)
        results_E2 = analyze_isi_peaks(counts_E2, bin_edges_E2)

        ''' LOOP THROUGH ALL THE NEURONS AND APPEND TO EACH DATA LIST'''
        # append the data for each identified peak
        for j in range(len(results_E1)):        # loop over all the peaks in E1
            simulation_number.append(i)
            neuron_label.append('E1')
            e_lst.append(e_val)
            f_lst.append(f_values[i])               # only append the f for that neuron 
            tau_lst.append(tau_val)
            I_inj_lst.append(INPUT_E)
            W_lst.append(W)
            peak_isi.append(results_E1[j]['peak_isi'])
            peak_num.append(j+1)
            mean.append(results_E1[j]['mean'])
            std.append(results_E1[j]['std'])
            count_sum.append(results_E1[j]['count_sum'])
            CV_isi.append(results_E1[j]['std']/results_E1[j]['mean'])
            LV.append(local_variation_E1[i])
            # append distance measures
            FT_dist.append(FT_distance)
            FT_EMD.append(FT_wasserstein)
            SPIKE_dist.append(SPIKE_distance)
            SPIKE_synch.append(SPIKE_synchrony)
            spike_directionality.append(spike_direct)
            schreiber_sim.append(schreiber)


        for k in range(len(results_E2)):        # loop over all the peaks in E2
            simulation_number.append(i)
            neuron_label.append('E2')
            e_lst.append(e_val)                 # e of the corresponding autaptic neuron
            f_lst.append(f_values[i])           # f of the corresponding autaptic neuron          
            tau_lst.append(tau_val)             # tau of the corresponding autaptic neuron
            I_inj_lst.append(INPUT_E)
            W_lst.append(W)
            peak_isi.append(results_E2[k]['peak_isi'])
            peak_num.append(k+1)
            mean.append(results_E2[k]['mean'])
            std.append(results_E2[k]['std'])
            count_sum.append(results_E2[k]['count_sum'])
            CV_isi.append(results_E2[k]['std']/results_E2[k]['mean'])
            LV.append(local_variation_E2[k])
            # append distance measures
            FT_dist.append(FT_distance)
            FT_EMD.append(FT_wasserstein)
            SPIKE_dist.append(SPIKE_distance)
            SPIKE_synch.append(SPIKE_synchrony)
            spike_directionality.append(spike_direct)
            schreiber_sim.append(schreiber)


    # append to dataframes, these are comparison metrics between both responses...
    results_df['sim_number'] = simulation_number
    results_df['noise_theta'] = INPUT_NOISE['theta']
    results_df['noise_sigma'] = INPUT_NOISE['sigma']
    results_df['neuron_label'] = neuron_label
    results_df['e'] = e_lst
    results_df['f'] = f_lst
    results_df['tau'] = tau_lst
    results_df['I_inj'] = I_inj_lst
    results_df['W'] = W_lst
    results_df['FT_distance'] = FT_dist
    results_df['FT_EMD'] = FT_EMD
    results_df['SPIKE_distance'] = SPIKE_dist
    results_df['SPIKE_synchrony'] = SPIKE_synch
    results_df['spike_directionality'] = spike_directionality
    results_df['schreiber similarity'] = schreiber_sim
    results_df['peak_isi'] = peak_isi
    results_df['peak_number'] = peak_num
    results_df['mean'] = mean
    results_df['std'] = std
    results_df['count_sum'] = count_sum
    results_df['CV_isi'] = CV_isi
    results_df['LV'] = LV
    results_df['num_peaks'] = results_df.groupby(['sim_number', 'neuron_label'])['neuron_label'].transform('count')


    # save the distance metrics
    with open(outfile, 'wb') as file:
        pickle.dump(results_df, file)


    gc.collect()



if __name__ == "__main__":
    main()