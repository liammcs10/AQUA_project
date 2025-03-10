
import numpy as np
import matplotlib.pyplot as plt
from AQUA2 import aqua2


# Parameters
a, b, c, d = 0.02, 0.2, -65, 8  # RS neuron
vpeak = 30  # Spike cutoff
T = 1000  # Total time in ms
h = 0.001  # Time step in ms
I = np.concatenate((np.zeros(int(0.01 * T / h)), 10 * np.ones(int(0.99 * T / h))))  # Input current

# Simulations
e, f, tau = 0.0, 0, 0
v1, _, _, _ = aqua2(T, h, I, vpeak, a, b, c, d, e, f, tau)

e, f, tau = 0.05, 5, 5
v2, u2, w2, st2 = aqua2(T, h, I, vpeak, a, b, c, d, e, f, tau)

e, f, tau = 0.05, 5, 0
v3, _, _, st3 = aqua2(T, h, I, vpeak, a, b, c, d, e, f, tau)

# Plot results
time = h * np.arange(0, round(T / h)) / 1000

plt.figure(figsize=(10, 6))
plt.subplot(3, 2, 1)
plt.plot(time, v1)
plt.xlim([0.008 * T / 1000, 0.06 * T / 1000])
plt.title('no autapse')

plt.subplot(3, 2, 3)
plt.plot(time, v2)
plt.xlim([0.008 * T / 1000, 0.06 * T / 1000])
plt.title('delayed autapse')

plt.subplot(3, 2, 5)
plt.plot(time, v3)
plt.xlim([0.008 * T / 1000, 0.06 * T / 1000])
plt.title('instantaneous autapse')
plt.xlabel('time [sec]')
plt.ylabel('membrane potential [mV]')

plt.subplot(3, 2, 2)
plt.plot(time, v1)
plt.xlim([0.9 * T / 1000, T / 1000])

plt.subplot(3, 2, 4)
plt.plot(time, v2)
plt.xlim([0.9 * T / 1000, T / 1000])

plt.subplot(3, 2, 6)
plt.plot(time, v3)
plt.xlim([0.9 * T / 1000, T / 1000])

plt.tight_layout()
plt.show()

# ISI Plot
plt.figure(figsize=(10, 4))
isi2 = np.diff(st2)
isi3 = np.diff(st3)

plt.subplot(1, 2, 1)
plt.plot(isi2[:-1], isi2[1:], '.')
plt.title('ISI - delayed autapse')

plt.subplot(1, 2, 2)
plt.plot(isi3[:-1], isi3[1:], '.')
plt.title('ISI - instantaneous autapse')

plt.tight_layout()
plt.show()