## Reproduction of Hazem's Matlab code for the AQUA Neuron

# Adaptive QUadratic with Autapse (AQUA)
# Euler implementation of the generic Izhikevich model with autapse.


import numpy as np
import matplotlib.pyplot as plt




def aqua(T, h, I, vpeak, C, vr, vt, k, a, b, c, d, e=0, f=0, tau=0, p=0):
    """
    T = total time
    h = time step
    I = injected current array of length N
    vpeak = threshold potential for spike generation
    C = membrane conductance
    vr = resting membrane potential
    vt = 
    k = 
    a, b, c, d are izhikevich parameters.
    e = timescale of autapse variable
    f = after spike reset of the autapse
    tau = autaptic time delay
    """

    N = round(T/h) # number of simulation steps

    # intial values
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

"""
# Example usage
T = 1000  # total time
h = 0.1   # time step
I = np.concatenate([np.zeros(int(0.01 * round(T / h))), 
                    158 * np.ones(int(0.0074 * round(T / h))), 
                    10 * np.ones(int(0.9826 * T / h))])
vpeak = 30
C = 50
vr = -60
vt = -40
k = 1.5
a = 0.03
b = 1.0
c = -40
d = 10
e = 0.05
f = 5
tau = 5
# Call the function
v, u, w, st = aqua(T, h, I, vpeak, C, vr, vt, k, a, b, c, d)

plt.plot(v)
plt.xlim([0.05*round(T/h), 0.2*round(T/h)])
plt.show()
"""