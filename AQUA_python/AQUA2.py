## Adaptive QUadratic with Autapse (AQUA) IF model

# AQUA neuron implemented in the izhikevich model of cortical neurons
# parameters pre-installed.

import numpy as np
import matplotlib.pyplot as plt



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
v, u, w, st = aqua2(T, h, I, vpeak, a, b, c, d)

plt.plot(v)
plt.show()
"""