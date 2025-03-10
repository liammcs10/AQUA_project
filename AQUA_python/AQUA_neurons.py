"""
Python implementation of Hazem's Adaptive QUadratic with Autapse (AQUA)

- aqua() is the generic implementation of the Izhikevich model with no predefined parameters
- aqua2() is the Izhikevich cortical neuron with pre-defined params


"""

import numpy as np



def aqua(T, h, I, vpeak, C, vr, vt, k, a, b, c, d, e=0, f=0, tau=0, p=0):
    """
    T = total time [ms]
    h = time step [ms]
    I = injected current array [pA], length N.
    vpeak = threshold potential for spike generation [mV]
    C = membrane conductance [check units]
    vr = resting membrane potential [mV]
    vt = dynamic membrane threshold variable [mV] (see Izhikevich)
    k = conductance [1/Ohm]
    a, b, c, d are izhikevich parameters.
    AUTAPTIC
    e = timescale of autapse variable [ms]
    f = after spike reset of the autapse
    tau = autaptic time delay [ms]
    """

    N = round(T/h) # number of simulation steps

    # variable arrays
    v = vr*np.ones(N)           # membrane potential
    u = np.zeros(N)             # membrane recovery variable
    w = np.zeros(N)             # autaptic current variable
    st = np.full(N, np.nan)     # spike times - array of NaNs length N


    # Euler integration of variables
    for t in range(1 + round(tau/h), N-1):
        #update variables using Euler
        v[t+1] = v[t] + h*(k*(v[t] - vr)*(v[t]-vt) - u[t] + w[t - round(tau/h)] + I[t])/C 
        u[t+1] = u[t] + h*a*(b*(v[t]-vr)-u[t])
        w[t+1] = w[t] - h*e*w[t]

        # Check for spike
        if v[t+1] >= vpeak:
            v[t] = vpeak #padding the spike amplitude
            v[t+1] = c # reset the membrane potential after spike
            u[t+1] += d # update recovery variable
            w[t+1] += f # autapse current update.
            st[np.where(np.isnan(st))[0][0]] = t*h # replace first NaN occurance with spike time.
        
        if t == round(0.9*N): # Not sure what this is for...
            v[t+1] += p

    st = st[~np.isnan(st)] # removes NaNs from the spike times.

    return v, u, w, st



def aqua2(T, h, I, vpeak, a, b, c, d, e=0, f=0, tau=0, p=0):

    N = round(T/h)
    v = -80*np.ones(N)
    u = np.zeros(N)
    w = np.zeros(N)
    st = np.full(N, np.nan)

    for t in range(1+round(tau/h), N-1):
        # Izhikevich cortical neuron model.
        v[t+1] = v[t] + h*(0.04*v[t]**2 + 5*v[t] + 140 - u[t] + w[t - round(tau/h)] + I[t])
        u[t+1] = u[t] + h*a*(b*v[t]-u[t])
        w[t+1] = w[t] - h*e*w[t]

        if v[t+1] >= vpeak: # check for spike and update values
            v[t] = vpeak
            v[t+1] = c
            u[t+1] += d
            w[t+1] += f
            st[np.where(np.isnan(st))[0][0]] = t*h # add spike time to first NaN occurrence
        
        if t == round(0.9*N):
            v[t+1] += p
        
    st = st[~np.isnan(st)]

    return v, u, w, st