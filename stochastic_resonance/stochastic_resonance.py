"""Stochastic resonance in autaptic RS neurons (see PROTOCOL.md).

Each neuron is driven at (threshold - THRESH_OFFSET) plus zero-mean noise of a
given variance; the firing rate over the whole trial is recorded.
"""
import os
import numpy as np
import pandas as pd
from aqua.batchAQUA_general import batchAQUA
from plot_SR import plot_sr

# ---------------- parameters ----------------
# neuron model (RS integrator)
RS = {'name': 'RS_ref', 'C': 100, 'k': 0.7, 'v_r': -60, 'v_t': -40, 'v_peak': 35,
      'a': 0.03, 'b': -2, 'c': -50, 'd': 100, 'e': 0., 'f': 0., 'tau': 0.}

E = 0.2                              # autapse decay rate (1/ms)
TAU = 2.                             # autapse delay (ms)
F_VALUES = np.arange(0, 401, 10)     # autapse strengths (pA)

THRESH_OFFSET = 1.                   # pA below threshold for the DC current
N_NOISE = 11
NOISE_VARS = np.linspace(0, 100, N_NOISE)   # noise variance (pA^2)
N_TRIALS = 10                        # noise realisations per (f, variance)

NOISE_TYPE = os.environ.get('NOISE_TYPE', 'white')   # 'white' or 'OU'
WN_FREQ = 50.                        # white noise: sample-and-hold rate (Hz)
OU_TAU = 10.                         # OU: correlation time (ms)

T = 5000.                            # ms
DT = 0.1                             # ms
SEED = 0

PLOT_DIR, DATA_DIR = './plots/', './data/'
# --------------------------------------------

N_ITER = int(T / DT)


def make_noise(var, n_traces, rng):
    """Zero-mean noise traces with stationary variance `var`, shape (n_traces, N_ITER)."""
    if NOISE_TYPE == 'white':
        hold = int(round(1000. / WN_FREQ / DT))           # steps per sample
        n_samples = -(-N_ITER // hold)
        samples = rng.normal(0., np.sqrt(var), (n_traces, n_samples))
        return np.repeat(samples, hold, axis=1)[:, :N_ITER]
    if NOISE_TYPE == 'OU':
        theta = 1. / OU_TAU
        sigma = np.sqrt(2 * theta * var)                  # gives stationary variance = var
        y = np.empty((n_traces, N_ITER))
        y[:, 0] = rng.normal(0., np.sqrt(var), n_traces)  # start in stationary distribution
        dW = rng.normal(0., np.sqrt(DT), (n_traces, N_ITER - 1))
        for i in range(1, N_ITER):
            y[:, i] = y[:, i-1] - theta * y[:, i-1] * DT + sigma * dW[:, i-1]
        return y
    raise ValueError(f'Unknown NOISE_TYPE {NOISE_TYPE!r}')


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # self-check: noise variance matches target
    test = make_noise(50., 20, rng)
    assert abs(test.var() / 50. - 1) < 0.2, f'noise variance {test.var():.1f} != 50'

    # threshold of the f = 0 neuron, reused for every f
    params = pd.DataFrame([{**RS, 'e': E, 'f': f, 'tau': TAU} for f in F_VALUES])
    th0, x0 = batchAQUA([{**RS, 'e': E, 'f': 0., 'tau': TAU}]).get_threshold(0)
    thresholds = np.full(len(F_VALUES), np.ravel(th0)[0])   # get_threshold returns a length-1 array
    x_ss = np.tile(x0, (len(F_VALUES), 1))

    # one batch over every (f, variance, trial)
    fi, vi, ti = [g.ravel() for g in np.meshgrid(np.arange(len(F_VALUES)), np.arange(N_NOISE),
                                                 np.arange(N_TRIALS), indexing='ij')]
    I_dc = thresholds[fi] - THRESH_OFFSET
    I_inj = I_dc[:, None] + np.vstack([make_noise(NOISE_VARS[v], N_TRIALS, rng)
                                       for f in range(len(F_VALUES)) for v in range(N_NOISE)])

    neurons = batchAQUA(params.iloc[fi].reset_index(drop=True))
    neurons.Initialise(x_ss[fi], np.zeros(len(fi)))
    spikes = neurons.update_batch(DT, N_ITER, I_inj, SAVE_ALL=False)
    n_spikes = (~np.isnan(spikes)).sum(axis=1) if spikes.size else np.zeros(len(fi), int)

    df = pd.DataFrame({'f': F_VALUES[fi], 'e': E, 'tau': TAU, 'threshold': thresholds[fi],
                       'I_dc': I_dc, 'noise_type': NOISE_TYPE, 'noise_var': NOISE_VARS[vi],
                       'trial': ti, 'n_spikes': n_spikes, 'rate_Hz': n_spikes / (T / 1000.),
                       'T': T, 'dt': DT,
                       'noise_param': WN_FREQ if NOISE_TYPE == 'white' else OU_TAU})
    tag = f'SR_{NOISE_TYPE}_var{NOISE_VARS.min():g}-{NOISE_VARS.max():g}'
    df.to_csv(os.path.join(DATA_DIR, tag + '.csv'), index=False)

    silent = df[(df.noise_var == 0) & (df.n_spikes > 0)].f.unique()
    if len(silent):
        print(f'WARNING: neurons fire without noise for f = {silent}')

    figs = plot_sr(os.path.join(DATA_DIR, tag + '.csv'), os.path.join(PLOT_DIR, tag + '.png'))
    print(f'saved {DATA_DIR}{tag}.csv and', *figs)


if __name__ == '__main__':
    main()
