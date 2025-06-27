import sys
sys.path.append("..\\") # parent directory
from AQUA_general import AQUA
from batchAQUA_general import batchAQUA
from stimulus import *
from plotting_functions import *


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.interpolate import LSQUnivariateSpline



"""
Try and see why the response is different to the ramp when an autapse is present.


"""


def phase_response(neuron_params, I_h, T_pert, H_pert):
    """
    this function will create the phase response curve of the neuron.

    Takes neuron to steady state -> step current. Then adds a timed perturbation 
    between spikes and observes the effect on spike timing.
    - can observe the effect at  the first and second spikes.

    1) Will need to determine the base firing frequency (spike times) for a given I.
    2) setup up I_batch with perturbations inserted at appropriate times.
    3) setup a corresponding batch of neurons.
    4) 

    INPUTS:
        neuron_params:      dict
                            contains the neuron parameters for the neuron
        perturbation:       array
                            represents the waveform of the perturbation.
                        
    OUTPUT:
        PRC:                ndarray
                            the phase response curves for different base frequencies.
    
    """

    dt = 0.005


    perturbation = H_pert * np.ones(int(T_pert/dt))

    print(f"A perturbation of height {H_pert} pA and duration {T_pert} ms")
    #I_range = [70, 100, 150, 200, 250, 300] # range of I values.


    # 1) baseline
    # temporary initial conditions
    x_ini = np.array([-80, 0, 0])
    t_ini = 0.0
    duration, t_i, t_f, t_ini, x_ini, w_past, X_base = get_baseline(neuron_params, I_h, x_ini, t_ini, dt)
    print(f"duration: {duration}")  # length of simulation
    print(f"t_i: {t_i}") # 2nd to last spike time
    print(f"t_f: {t_f}") # last spike time
    # t_ini and x_ini will be initial conditions


    # 2) generate batch of injected currents with slight perturbations
    N_pert, pert_times = get_perturbations(dt, t_i, t_f, t_ini)

    print("Number of perturbations: " + str(N_pert))
    t_p = pert_times * dt + t_ini   # convert perturbations times from frames to ms.
    delta_T = t_f - t_i     # period of firing
    print(f"Reference spiking period is {np.round(delta_T, 2)} ms ({np.round(1000/delta_T)} Hz)")


    # 3) setup batch of neurons...
    N_neurons = 1000
    N_sims = int(np.ceil(N_pert / N_neurons))
    print(f"N_sims: {N_sims}")

    data = np.zeros((4, N_pert))    # 0 - I_h, 1 - Hz, 2 - pert time, 3 - PRC

    for i in range(N_sims):
        lower_bound = i * N_neurons
        if (i + 1)*N_neurons > N_pert:
            upper_bound = N_pert
            N = upper_bound - lower_bound
        else:
            upper_bound = (i + 1) * N_neurons
            N = N_neurons
        
        I_inj = get_injected(I_h, pert_times[lower_bound:upper_bound], N, duration, perturbation)

        w_prev = np.array([w_past for _ in range(N)]) # make past w N x delay_steps

        norm_tp, PRC, delta_T, X_prc = get_PRC(N, neuron_params, duration, I_inj, x_ini, t_ini, dt, t_f, t_i, t_p[lower_bound:upper_bound], w_prev)


        data[0, lower_bound:upper_bound] = I_h
        data[1, lower_bound:upper_bound] = np.round(1000/delta_T, 2)
        data[2, lower_bound:upper_bound] = norm_tp
        data[3, lower_bound:upper_bound] = PRC 

    fig, ax = plt.subplots(1, 1, figsize = (5, 10))
    ax.plot(X_base[0, :250])
    ax.plot(X_prc[0, :250])
    plt.show()

    df = pd.DataFrame(data = data.T, columns = ["I_h", "Hz", "Perturbation time", "PRC"])
    """
    # Want to make a spline from the PRC curves - this way the output is a little smoother 
    # and we don't need such high-resolution simulations.

    t = np.linspace(0, 1, 100)
    knots = t[2:-2]

    spl = LSQUnivariateSpline(data[0, :], data[1, :], t = t, k = 3)
    #y_spline = LSQUnivariateSpline(t, data[1, :], knots)

    t_fine = np.linspace(0, 1, 1000)
    y_spl = spl(t)
    
    """

    """
    # plot phase response curve
    plt.plot(data[0, :], data[1, :])
    #plt.plot(x_fit, y_fit) # plotted the perturbation in ms.
    plt.xlabel("Normalized Perturbation Time")
    plt.ylabel("Phase Response [ms]")
    plt.title(f"Phase Response Curve of the {neuron_params['name']} at {I_h} pA driving ({np.round(1000/delta_T, 2)} Hz)")
    plt.savefig(f"phase_response_curve_{neuron_params['name']}_{I_h}.png")
    plt.show()
    """


    return df
    



## STEP 1
def get_baseline(neuron_params, I_h, x_ini, t_ini, dt):
    """ Gets the baseline frequency and spike times for the neuron and driving frequency"""
    neuron = AQUA(neuron_params)

    # sim duration
    T = 2000
    N_iter = int(np.round(T/dt))

    #injected current
    I_inj = I_h*np.ones(N_iter)


    neuron.Initialise(x_ini, t_ini)
    X, T, spikes = neuron.update_RK2(dt, N_iter, I_inj)

    print(f"Number of spikes: {len(spikes)}")

    if len(spikes) <= 3:
        print("Driving current is too weak")
    
    elif len(spikes) < 10:
        t_i = np.round(spikes[-2]/dt)*dt # second to last spike
        t_f = np.round(spikes[-1]/dt)*dt # last spike

        # initial conditions
        t_start = np.round(((spikes[-2] + spikes[-3])/2)/dt)*dt
        x_start = X[:, int(t_start/dt)]
 
    else:
        t_i = np.round(spikes[8]/dt)*dt  # 9th spike
        t_f = np.round(spikes[9]/dt)*dt  # 10th spike

        # initial conditions
        t_start = np.round(((spikes[8] + spikes[7])/2)/dt)*dt  # ms
        #print(X[:, int(t_start/dt)].type())
        x_start = X[:, int(t_start/dt)]
        #print(x_start.type())

    
    duration = (t_f - t_start) + (t_f - t_i)/2    # duration of phase-response sims in ms
    duration = int(np.round(duration/dt)) # duration in time steps

    w_past = X[2, int((t_start - neuron.tau)/dt):int(t_start/dt)]
    
    return duration, t_i, t_f, t_start, x_start, w_past, X[:, int(t_start/dt): int(t_start/dt) + duration]


## STEP 2
def get_perturbations(dt, t_i, t_f, t_start):
    # determines the times of the perturbations
    delta_t = int(np.round((t_f - t_i)/dt)) # spiking period in number of timesteps

    # cap at 500 perturbations
    if delta_t > 500:
        N_pert = 500
    else:
        N_pert = delta_t
    
    pert_times = int(np.round((t_i - t_start)/dt)) + np.linspace(0, delta_t, num = N_pert, dtype = int) # time (indices) of perturbations


    return N_pert, pert_times       # I_inj

def get_injected(I_h, pert_times, N, duration, perturbation):
    """ Will generate the array of injected currents for the smaller sample of simulations.
    INPUTS:
        I_h:            the height of the step current
        pert_times:     sub-sample array of the times of each perturbation
        N:              the number of perturbations for this run
        duration:       the duration of the stimulation
        perturbation:   the trace of the perturbation current
    
    OUT:
        I_inj:          ndarray representing the current traces.
    """
    I_inj = I_h * np.ones((N, duration))

    for n, i in enumerate(pert_times):
        pert = np.zeros(duration)
        stop = len(perturbation) + i
        pert[i:stop] = perturbation # perturbation

        I_inj[n, :] += pert     # add the perturbation to the step
    
    return I_inj

## STEP 3
def get_PRC(N_pert, neuron_params, duration, I_inj, x_ini, t_ini, dt, t_f, t_i, t_p, w_past = []):

    # create initialization arrays for the batch
    params_list = []
    x_start = np.zeros((N_pert, 3))
    t_start = np.zeros(N_pert)
    for i in range(N_pert):
        params_list.append(neuron_params)
        x_start[i, :] = x_ini
        t_start[i] = t_ini

    # define and initialise the batch
    batch = batchAQUA(params_list)
    batch.Initialise(x_start, t_start)

    # run perturbation sims.
    X, _, spikes = batch.update_batch(dt, duration, I_inj, w_past)

    # compare the deviation in the last spike to the time of the perturbation
    
    delta_T = t_f - t_i     # baseline firing period

    norm_tp = (t_p - t_i)/delta_T   # normalized perturbation time

    # here, spikes[:, -1] is the t_f' for all perturbations
    PRC = (delta_T - (spikes[:, -1] - t_i)) # ms

    return norm_tp, PRC, delta_T, X[0, :, :]



    

def main():

    RS = {'name': 'RS', 'C': 100, 'k': 0.7, 'v_r': -60, 'v_t': -40, 'v_peak': 35,
      'a': 0.03, 'b': -2, 'c': -50, 'd': 100, 'e': 0.0, 'f': 0.0, 'tau': 0.0}
    
    RS_E = {'name': 'RS', 'C': 100, 'k': 0.7, 'v_r': -60, 'v_t': -40, 'v_peak': 35,
      'a': 0.03, 'b': -2, 'c': -50, 'd': 100, 'e': 0.03, 'f': 8.0, 'tau': 0.5}
    
    FS = {'name': 'FS', 'C': 20, 'k': 1, 'v_r': -55, 'v_t': -40, 'v_peak': 25,
        'a': 0.2, 'b': -2, 'c': -45, 'd': 0, 'e': 0.0, 'f': 0.0, 'tau': 0.0}


    T_pert = 0.01 # ms, duration of perturbation
    H_pert = 100 # pA, amplitude of perturbation
    #pert = 100*np.ones(10) 

    I_heights = [150]   #[70, 100, 150, 200, 250, 300, 350, 400, 450]

    df = pd.DataFrame(data = [], columns = ["I_h", "Hz", "Perturbation time", "PRC"])


    for I_h in I_heights:
        df = pd.concat([df, phase_response(RS, I_h, T_pert, H_pert)], ignore_index = True)

    sns.lineplot(data = df, x = "Perturbation time", y = "PRC", hue = "Hz")
    #plt.savefig("phase_response_RS_E_allHz_dt-0.005.png")
    plt.show()

if __name__ == "__main__":
    main()