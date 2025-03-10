"""
Simulates responses of AQUA and AQUA2 under different 
parameter regimes and input currents.

An important note, seems like the response is dependent on the timestep
parameter 0.001 creates a burst in the autaptic neuron, but no otherwise.
"""

import numpy as np
import matplotlib.pyplot as plt
from AQUA import aqua
from AQUA2 import aqua2



# Main simulation code
def simulate_neurons():
    # Parameters for the first neuron (IB)
    C = 50
    vr = -60
    vt = -40
    k = 1.5
    a = 0.03
    b = 1
    c = -40
    d = 10
    vpeak = 25  # spike cutoff

    T1 = 10000  # total time (ms)
    h = 0.001  # time step (ms)

    # Create a new figure
    plt.figure(figsize=(12, 8))

    # Input current for first simulation
    I1 = np.concatenate([np.zeros(int(0.01 * round(T1 / h))), 
                         158*np.ones(int(0.0074 * round(T1 / h))), 
                         10*np.ones(int(0.9826 * round(T1 / h)))])

    # No autapse
    e = 0.0; f = 0; tau = 0
    v1, _, _, _ = aqua(T1, h, I1, vpeak, C, vr, vt, k, a, b, c, d, e, f, tau)

    # With autapse
    e = 0.2; f = 100; tau = 0
    v3, _, _, _ = aqua(T1, h, I1, vpeak, C, vr, vt, k, a, b, c, d, e, f, tau)

    plt.subplot(3, 2, 2)
    plt.plot(np.arange(round(T1 / h)) * h / 1000, I1)
    plt.xlim([0.095, 0.25])
    plt.ylim([-5, 165])
    plt.xticks(fontsize=12)
    
    plt.subplot(3, 2, 4)
    plt.plot(np.arange(round(T1 / h)) * h / 1000, v1)
    plt.xlim([0.095, 0.25])
    plt.xticks(fontsize=12)

    plt.subplot(3, 2, 6)
    plt.plot(np.arange(round(T1 / h)) * h / 1000, v3)
    plt.xlim([0.095, 0.25])
    plt.xlabel('Time [sec]')
    plt.xticks(fontsize=12)

    # Parameters for the second neuron (RS)
    a = 0.02
    b = 0.2
    c = -65
    d = 8
    vpeak = 30  # spike cutoff

    T2 = 500  # total time (ms)

    # Input current for second simulation
    I2 = np.concatenate([np.zeros(int(0.1 * round(T2 / h))), 
                         14*np.ones(int(0.9 * round(T2 / h)))])

    # No autapse
    e = 0.0; f = 0; tau = 0
    v1, _, _, _ = aqua2(T2, h, I2, vpeak, a, b, c, d, e, f, tau)

    # With autapse
    e = 0.05; f = 6; tau = 0
    v3, _, _, _ = aqua2(T2, h, I2, vpeak, a, b, c, d, e, f, tau)

    plt.subplot(3, 2, 1)
    plt.plot(np.arange(round(T2 / h)) * h / 1000, I2)
    plt.xlim([0, 0.25])
    plt.ylim([min(I2) - 1, 15])
    plt.ylabel('I [pA]')
    plt.xticks(fontsize=12)


    plt.subplot(3, 2, 3)
    plt.plot(np.arange(round(T2 / h)) * h / 1000, v1)
    plt.xlim([0, 0.25])
    plt.ylabel('V [mV]; no autapse')
    plt.xticks(fontsize=12)


    plt.subplot(3, 2, 5)
    plt.plot(np.arange(round(T2 / h)) * h / 1000, v3)
    plt.xlim([0, 0.25])
    plt.ylabel('V [mV]; with autapse')
    plt.xlabel('Time [sec]')
    plt.xticks(fontsize=12)

    plt.tight_layout()
    plt.show()

# Run the simulation
simulate_neurons()