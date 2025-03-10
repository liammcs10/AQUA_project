import numpy as np
import matplotlib.pyplot as plt
from AQUA import aqua


# Neuron parameters
C = 50
vr = -60
vt = -40
k = 1.5
a = 0.03
b = 1
c = -40
d = 10
vpeak = 25

# Simulation parameters
T = 10000  # ms
h = 0.001  # ms
I = np.concatenate([np.zeros(int(0.01 * round(T / h))), 158 * np.ones(int(0.0074 * round(T / h))), 10 * np.ones(int(0.9826 * T / h))])

# Simulate different conditions (no autapse, delayed autapse, instantaneous autapse)
v1, _, _ , _= aqua(T, h, I, vpeak, C, vr, vt, k, a, b, c, d, 0.0, 0, 0)
v2, _, _, st2 = aqua(T, h, I, vpeak, C, vr, vt, k, a, b, c, d, 0.2, 100, 5)
v3, _, _, st3 = aqua(T, h, I, vpeak, C, vr, vt, k, a, b, c, d, 0.2, 100, 0)

# Plot the results
time = np.arange(round(T/h)) * h / 1000  # time in seconds

plt.figure(3, figsize=(10, 8))
plt.subplot(3, 2, 1); plt.plot(time, v1); plt.xlim([.008 * T / 1000, .03 * T / 1000]); plt.ylim([-60, 30]); plt.title('no autapse')
plt.subplot(3, 2, 3); plt.plot(time, v2); plt.xlim([.008 * T / 1000, .03 * T / 1000]); plt.ylim([-60, 30]); plt.title('delayed autapse')
plt.subplot(3, 2, 5); plt.plot(time, v3); plt.xlim([.008 * T / 1000, .03 * T / 1000]); plt.ylim([-60, 30]); plt.title('instantaneous autapse')
plt.subplot(3, 2, 2); plt.plot(time, v1); plt.xlim([.97 * T / 1000, T / 1000]); plt.ylim([-60, 30]); plt.title('no autapse')
plt.subplot(3, 2, 4); plt.plot(time, v2); plt.xlim([.97 * T / 1000, T / 1000]); plt.ylim([-60, 30]); plt.title('delayed autapse')
plt.subplot(3, 2, 6); plt.plot(time, v3); plt.xlim([.97 * T / 1000, T / 1000]); plt.ylim([-60, 30]); plt.title('instantaneous autapse')
plt.xlabel('time [sec]'); plt.ylabel('membrane potential [mV]')
plt.tight_layout()

# Plot ISI (Interspike Interval) analysis
plt.figure(4, figsize=(10, 4))
isi2 = np.diff(st2)
isi3 = np.diff(st3)
plt.subplot(1, 2, 1); plt.plot(isi2[-1000+1:-1], isi2[-1000+2:],'b.'); plt.title('ISI Delayed Autapse')
plt.subplot(1, 2, 2); plt.plot(isi3[-1000+1:-1], isi3[-1000+2:], 'r.'); plt.title('ISI Instantaneous Autapse')
plt.tight_layout()

plt.show()