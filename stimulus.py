""" implement a class which will contain standard injected current traces for each neuron

e.g.
    Ornstein-Uhlenbeck process
    Poisson spike trains
    ...


"""
import numpy as np
from collections.abc import Sequence



def OU_current(N_iter, dt, y_0, theta, mu, sigma):
    # implemented using the Euler-Maruyama method.
    # mu is an array representing the time varying long-term mean.
    ys = np.zeros(N_iter)
    ys[0] = y_0
    for i in range(1, N_iter):
        y = ys[i-1]
        dW = np.random.normal(loc = 0.0, scale = np.sqrt(dt))
        ys[i] = y + theta * (mu[i] - y) * dt + sigma * dW
    
    return ys

def ramp(N_iter, dt, y_0, delay, I_end):
    """
    Applies a ramp current starting at delay
    IN  
        y_0: the starting current value
        delay: the delay [ms] before applying the ramp
        I_end: The final current value

    OUT
        I_inj: np.array representing the ramp stimulus
    """
    delay_steps = int(delay / dt)       # the delay in units of timesteps.

    return np.concatenate((y_0 * np.ones(delay_steps), np.linspace(y_0, I_end, (N_iter - delay_steps))))


def step_current(N_iter, dt, y_0, delay, I_h):
    """
    A step current of heigh I_h applied at delay
    IN
        delay: the delay [ms] before the step
        I_h: the height of the step [pA]
    OUT
        I_inj: np.array of length N_iter containing the stimulus trace    
    """
    delay_steps = int(delay / dt)
    
    return np.concatenate((y_0 * np.ones(delay_steps), I_h * np.ones(N_iter - delay_steps)))

def spikes_from_dist(inverse_cdf, N_iter, dt, seed = None):
    """
    generates a spike train with ISIs sampled from a pre-defined distribution.
    NEEDS VERIFYING AND FIXING
    
    Params:
        inverse_cdf:    function, defines the inverse cumulative density function of the desired distribution
        N_iter:         length of the return array containing spikes
        dt:             timestep in ms
        seed;           seed for the random number generator
    
    returns
        I:              array (N_iter, )
                        contains spikes with ISIs distributed according to the desired distribution.
                        spikes have unit height.
    """
    I = np.zeros(N_iter)        # array to store spike train
    T = N_iter * dt             # duration of the stimulus [ms]
    print(T)
    rng = np.random.default_rng(seed = seed)        # specify the seed for reproducibility
    u = rng.uniform(size = N_iter)     # random samples from uniform distribution [0, 1)

    x = inverse_cdf(u) # array of lenght N_iter, element is an ISI in ms

    t_spikes = np.cumsum(x)     # array of spike times [ms]
    t_spikes = t_spikes[np.cumsum(t_spikes) < T] # only keep the spike times in the trial time.

    t_i = (t_spikes / dt).astype(int) # spike times in units of I indices

    I[t_i] += 1

    return I.astype(int), x

def spikes_constant(N_iter, dt, y_0, ISI, N_spikes, spike_heights, spike_duration = 1, delay = 500):
    """
    Create an input spike train (of N_spikes) with a variable ISI and heights
    """
    # check ISI is iterable
    if isinstance(ISI, (Sequence, np.ndarray)) and not isinstance(ISI, str):
        N_isi = len(ISI)
    else:   # if only a scalar is passed
        N_isi = 1
        ISI = np.array([ISI])


    # check pulse heights is an iterable
    if isinstance(spike_heights, Sequence) and not isinstance(spike_heights, str):
        N_heights = len(spike_heights)
    else:   # if only a scalar is passed
        N_heights = 1
        spike_heights = np.array([spike_heights])
    

    I = y_0 * np.ones(N_iter)

    delay_steps = int(delay/dt)
    ISI_steps = (ISI/dt).astype(int)
    spike_duration_steps = int(spike_duration/dt)

    pulse_start_count = 0       # will track where each pulse should start
    for i in range(N_spikes):
        pulse_num = i%N_heights
        isi_num = i%N_isi
        I[delay_steps + pulse_start_count: delay_steps + pulse_start_count + spike_duration_steps] += spike_heights[pulse_num]
        pulse_start_count += ISI_steps[isi_num]

    return I



def sinusoid(N_iter, dt, freq, amp, phase):
    x = np.linspace(0, N_iter * dt, N_iter)

    return amp*np.sin(freq*x + phase), x


