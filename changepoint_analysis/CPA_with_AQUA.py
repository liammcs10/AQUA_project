"""
Here we will try to implement a simple change point analysis protocol using the AQUA model.

Injected current will contain 1 change point at a fixed time where the mean of the injected current will change. There will be a super imposed noise.

The question: Based on the spiking activity alone, is it possible to estimate the change point? How significant does the change point have to be? 

** Can the autapse help distinguish smaller changes in activity?

- What happens if multiple change points are present? 

ALGORITHMS:
    - OFFLINE:
        - FLUSS, SST, ESST...
    - ONLINE:
        - CUSUM, or neuron-based versions?


PARAMETERS TO VARY:
- The change in the mean (** CAN ALSO ADJUST THE STARTING CURRENT VALUE **)
- Variance of the noise (** WE CAN ALSO VARY THE TYPE OF NOISE **)
- compare across autapse parameters
- window size of the fluss_detector (How does it fare with more or less data to compare...?)

ALT. QUESTIONS:
    - what happens if multiple change points are present?
    - window size of the FLUSS detector

OUTPUTS:

Psychometric curve - does the autapse shift the curve to the left or right?
Distribution of change point times - Does the autapse give a better estimate overall?

"""
import warnings
warnings.filterwarnings("ignore", module = "stumpy")


import aqua

from aqua.batchAQUA_general import batchAQUA
from aqua.utils import *
from aqua.stimulus import *
from CP_functions import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from scipy.signal import convolve, windows
from tqdm import tqdm
import pickle
import seaborn as sns
sns.set_theme()


from changepoynt.algorithms.fluss import FLUSS
from changepoynt.visualization.score_plotting import plot_data_and_score

import stumpy
from joblib import Parallel, delayed



# neuron model (RS integrator)
RS = {'name': 'RS_ref', 'C': 100, 'k': 0.7, 'v_r': -60, 'v_t': -40, 'v_peak': 35,
     'a': 0.03, 'b': -2, 'c': -50, 'd': 100, 'e': 0., 'f': 0., 'tau': 0.}




def main():
    '''
    Run the estimate on many different parameter combinations.
    
    Parameters to vary : autapse (can make it narrower than that in other sims), height of change point, noise,
    window of the FLUSS analyser.

    We can also compare values for the raw spiking time series and also the ISI distribution.  

    '''

    # for now only vary the peak autapse current? 
    f_values = np.linspace(0, 300, 5) # 15
    f_values = np.array([0, 100, 150, 200, 220, 240, 260, 280, 300, 350, 400])
    print(f_values)

    # height of the change point
    #d_H = np.linspace(0, 50, 10)
    #d_H = np.linspace(0, 5, 10)
    d_H = np.array([0, 1, 2, 3, 4, 5, 10, 12, 15, 18, 20, 25, 30, 40, 50])

    # noise variance 
    var = np.linspace(0, 50, 5)
    #var = np.array([0])

    # FLUSS window size (in time steps)
    window = 300
    print(f"WINDOW: {window}")

    N_repeats = 100

    #outfile = 'CPA_data_N100.pickle'
    #simulate_w_fluss(RS, f_values, d_H, var, window, N_repeats, outfile)

    # ISI / KS changepoint (spike times only)
    n_isi = 5       # ISIs per window (RS neuron fires ~4 Hz at I_0 = 60, so only ~6-12 ISIs before the CP)
    method = 'llr'
    spike_file = None   # None = simulate; or e.g. 'CPA_cusum_N100_spikes.pickle' to re-analyse saved spikes
    outfile = f'CPA_cusum_N{N_repeats}_{method}.pickle'
    simulate_w_cusum(RS, f_values, d_H, var, n_isi, N_repeats, outfile, repeats_per_batch = 5, method = method,
                     spike_file = spike_file)




def simulate_runs(model, f_values, d_H, var, N_repeats, T, changepoint, dt = 0.1,
                  I_0 = 60, noise_rate = 50, e = 0.2, tau = 2.0,
                  repeats_per_batch = 1, save_all = True):
    """
    Simulates every (repeat, var, H, f) combination with a noisy step current and yields
    the results one batch of repeats at a time (so full V traces never all sit in memory).

    - The current steps from I_0 to I_0 + h at `changepoint` [ms].
    - Noise is Gaussian with variance v, resampled at noise_rate [Hz] and held in between.
    - Within each (repeat, var, H) block every autapse strength f gets the same noisy current.

    Params:
        repeats_per_batch:  repeats stacked into one batchAQUA call (limits I_inj memory)
        save_all:           passed to batchAQUA.update_batch (False = spike times only)
    Yields:
        reps:   list of repeat indices in this batch
        out:    (X, t, spikes) if save_all, else the NaN-padded spike time array
                Rows are ordered rep -> var -> H -> f (within a repeat, see find_idx).
    """
    N_iter = int(T/dt)

    # create parameter dictionary
    params = []
    for f in f_values:
        temp_dict = model.copy()
        temp_dict['e'] = e
        temp_dict['tau'] = tau
        temp_dict['f'] = np.round(f, 2)
        params.append(temp_dict)

    N_neurons = len(params)
    N_sims = N_neurons * len(d_H) * len(var)    # rows per repeat, ordered var -> H -> f

    noise_steps = int(1/noise_rate * (1000/dt))
    num_unique = N_iter // noise_steps      # number of random numbers to sample

    # same neuron parameters for every (var, H) block and every repeat
    params_df = pd.DataFrame(params * (len(var) * len(d_H)))

    for r0 in range(0, N_repeats, repeats_per_batch):
        reps = list(range(r0, min(r0 + repeats_per_batch, N_repeats)))
        print(f"Repeats {reps[0]}-{reps[-1]} of {N_repeats}")

        I_inj = np.zeros((len(reps) * N_sims, N_iter))
        for j, _ in enumerate(reps):
            for l, v in enumerate(var):
                for m, h in enumerate(d_H):
                    step = step_current(N_iter, dt, y_0 = I_0, delay = changepoint, I_h = I_0 + h)
                    unique_samples = np.random.normal(loc = 0.0, scale = np.sqrt(v), size = num_unique)
                    WN = np.repeat(unique_samples, noise_steps)
                    idx = j * N_sims + l * N_neurons * len(d_H) + m * N_neurons
                    I_inj[idx:idx+N_neurons, :] = step + WN     # same noise for all f in this block

        batch_df = pd.concat([params_df] * len(reps), ignore_index = True)
        x_start = np.full((len(batch_df), 3), fill_value = np.array([-60, 0, 0]))
        t_start = np.zeros(len(batch_df))

        batch = batchAQUA(batch_df)
        batch.Initialise(x_start, t_start)
        out = batch.update_batch(dt, N_iter, I_inj, SAVE_ALL = save_all)
        del I_inj

        yield reps, out


def simulate_w_fluss(model, f_values, d_H, var, window, N_repeats, outfile):
    """
    Changepoint detection with FLUSS on the membrane potential of every simulated neuron
    (T = 3000 ms, step at 1500 ms). One batch per repeat. Saves a DataFrame with one row
    per neuron per repeat.
    """
    T = 3000 # ms
    dt = 0.1
    changepoint = 1500 # ms

    cols = ['N', 'variance', 'step_height', 'autapse f', 'FLUSS window', 'CAC', 'change point']
    rows = []

    for reps, (X, t, _) in simulate_runs(model, f_values, d_H, var, N_repeats, T, changepoint, dt = dt):
        # calculate the arc crossings using parallel joblib functionality
        results = Parallel(n_jobs = -1)(
            delayed(process_single_row)(row, X, window, dt) for row in range(X.shape[0])
        )

        row_num = 0
        for rep in reps:
            for v in var:
                for h in d_H:
                    for f in f_values:
                        CAC, CP = results[row_num]
                        rows.append([rep, v, h, f, window, CAC, CP])
                        row_num += 1

    data_df = pd.DataFrame(rows, columns = cols)
    print(data_df.head())

    # save the dataframe
    with open(outfile, 'wb') as file:
        pickle.dump(data_df, file)

    return data_df


def simulate_w_cusum(model, f_values, d_H, var, n_isi, N_repeats, outfile,
                     stride = 1, repeats_per_batch = 5, t_discard = 200, method = 'ks', spike_file = None):
    """
    Changepoint detection for autaptic neurons from the ISI distribution of spikes.
    method = 'ks' (windowed KS test, default), 'llr' (log-likelihood-ratio CUSUM on fitted
    ISI distributions) or 'llr_pair' (LLR on consecutive ISI pairs), see cusum_analysis.

    - Only spike times are simulated (SAVE_ALL = False), so much larger runs are possible.
    - For each condition (var, H, f) the ISIs of all repeats are pooled into two reference
      distributions: before and after the (known) changepoint.
    - A window of n_isi consecutive ISIs slides along each spike train. Each window is compared
      to both references with a KS test, d = KS(pre) - KS(post). The changepoint is the best
      single step fit to d (CUSUM argmin), see ks_window_cp in CP_functions.

    Params:
        n_isi:              number of ISIs per window
        stride:             window step (in ISIs)
        repeats_per_batch:  repeats stacked into one batchAQUA call (limits I_inj memory)
        t_discard:          spikes before this time [ms] are ignored (initial transient)
        spike_file:         None = simulate. Otherwise a '_spikes.pickle' from an earlier run, which
                            is re-analysed without simulating; f_values, d_H, var, N_repeats and the
                            changepoint are then taken from the file (model and the passed values are ignored)
    Saves:
        outfile:                    DataFrame with one row per trial
        outfile + '_spikes.pickle': raw spike times, to re-run the analysis without re-simulating
                                    (only written when simulating)
    """
    if spike_file is not None:
        with open(spike_file, 'rb') as file:
            saved = pickle.load(file)
        for name, passed in [('f_values', f_values), ('d_H', d_H), ('var', var), ('N_repeats', N_repeats)]:
            if not np.array_equal(np.asarray(passed), np.asarray(saved[name])):
                print(f"Note: using {name} from {spike_file}: {saved[name]}")
        spikes = saved['spikes']
        f_values, d_H, var = saved['f_values'], saved['d_H'], saved['var']
        N_repeats, changepoint = saved['N_repeats'], saved['changepoint']

    else:
        T = 10000 # ms
        dt = 0.1
        changepoint = T // 2 # ms

        # spikes[(repeat, v, h, f)] = 1D array of spike times
        spikes = {}
        for reps, spike_arr in simulate_runs(model, f_values, d_H, var, N_repeats, T, changepoint, dt = dt,
                                             repeats_per_batch = repeats_per_batch, save_all = False):
            row = 0
            for rep in reps:
                for v in var:
                    for h in d_H:
                        for f in f_values:
                            if spike_arr.shape[1] > 0:
                                s = spike_arr[row]
                                spikes[(rep, v, h, f)] = s[~np.isnan(s)]
                            else:
                                spikes[(rep, v, h, f)] = np.array([])
                            row += 1

        spike_out = outfile.replace('.pickle', '') + '_spikes.pickle'
        with open(spike_out, 'wb') as file:
            pickle.dump({'spikes': spikes, 'f_values': f_values, 'd_H': d_H, 'var': var,
                         'N_repeats': N_repeats, 'changepoint': changepoint, 'dt': dt, 'T': T}, file)

    data_df = cusum_analysis(spikes, f_values, d_H, var, N_repeats, n_isi,
                             changepoint, stride = stride, t_discard = t_discard, method = method)
    print(data_df.head())

    with open(outfile, 'wb') as file:
        pickle.dump(data_df, file)

    return data_df


def _cusum_condition(trials, h, changepoint, n_isi, stride, t_discard, method = 'ks'):
    """
    Builds the references for one condition (step height h) and finds the CP for each of its repeats.
    Returns (list of (CP, score, score per s), pre family, post family, divergences dict, post fit)
    where post fit = (post distribution, post reference) for the Fisher information across step heights.
    Families, post fit and the per-second / divergence values are None / NaN for 'ks'.
    The model-free distances (empirical_distances) between the pooled ISIs are computed for every method.
    """
    isis = [spike_isis(s, t_discard) for s in trials]
    pre_ref, post_ref = reference_isis(isis, changepoint)
    emp = empirical_distances(pre_ref, post_ref)

    if method == 'ks':
        res = [ks_window_cp(isi, t_isi, pre_ref, post_ref, n_isi, stride) + (np.nan,) for isi, t_isi in isis]
        div = dict.fromkeys(DIVERGENCE_KEYS + ('fisher local', 'fisher local per s'), np.nan)
        return res, None, None, {**div, **emp}, (None, post_ref)

    if method == 'llr':
        dist_pre, name_pre, _ = fit_isi_distribution(pre_ref)
        dist_post, name_post, _ = fit_isi_distribution(post_ref)
        div = distribution_divergences(dist_pre, dist_post, pre_ref, post_ref)
        res = []
        for isi, t_isi in isis:
            CP, score, t, l, _ = llr_cp(isi, t_isi, dist_pre, dist_post, return_trace = True)
            res.append((CP, score, evidence_rate(t, l, isi, CP)))
        post_fit = (dist_post, post_ref)

    elif method == 'llr_pair':
        pre_pairs, post_pairs = reference_pairs(isis, changepoint)
        dist_pre, name_pre, _ = fit_isi_pair_distribution(pre_pairs)
        dist_post, name_post, _ = fit_isi_pair_distribution(post_pairs)
        div = distribution_divergences(dist_pre, dist_post, pre_pairs, post_pairs, pairs = True)
        res = []
        for isi, t_isi in isis:
            CP, score, t, l, _ = llr_pair_cp(isi, t_isi, dist_pre, dist_post, return_trace = True)
            res.append((CP, score, evidence_rate(t, l, isi[1:], CP)))
        post_fit = (dist_post, post_pairs)

    else:
        raise ValueError(f"method must be 'ks', 'llr' or 'llr_pair', not {method!r}")

    # local Fisher information about the input current, J ~ I h^2 for small h
    div['fisher local'] = div['divergence'] / h**2 if h != 0 else np.nan
    div['fisher local per s'] = div['divergence per s'] / h**2 if h != 0 else np.nan
    return res, name_pre, name_post, {**div, **emp}, post_fit


def _fisher_across_steps(d_H, post_fits, pairs = False):
    """
    Fisher information about the input current at I_0 + h for every step height, from finite
    differences between the post distributions of neighbouring step heights (one autapse f and variance):
        I(h_i) ~ J(post_{i-1}, post_{i+1}) / (h_{i+1} - h_{i-1})^2     (one-sided at the ends)
    IN
        post_fits:  {h: (post distribution, post reference)} from _cusum_condition
    OUT
        {h: (I per ISI, I per s)}, nats / pA^2
    """
    hs = np.sort(np.asarray(d_H))
    out = {}
    for i, h in enumerate(hs):
        lo, hi = hs[max(i - 1, 0)], hs[min(i + 1, len(hs) - 1)]
        if lo == hi:
            out[h] = (np.nan, np.nan)
            continue
        (dist_lo, ref_lo), (dist_hi, ref_hi) = post_fits[lo], post_fits[hi]
        div = distribution_divergences(dist_lo, dist_hi, ref_lo, ref_hi, pairs = pairs)
        out[h] = (div['divergence'] / (hi - lo)**2, div['divergence per s'] / (hi - lo)**2)
    return out


def cusum_analysis(spikes, f_values, d_H, var, N_repeats, n_isi, changepoint, stride = 1, t_discard = 200,
                   method = 'ks'):
    """
    Runs the ISI changepoint analysis on a dict of spike trains (from simulate_w_cusum,
    or re-loaded from its _spikes.pickle). One parallel job per condition (var, H, f).
        method = 'ks':  windows of n_isi ISIs compared to the pre/post references with a KS test
        method = 'llr': per-ISI log-likelihood ratio under the best-fitting (BIC) gamma /
                        lognormal / inverse Gaussian / log-ISI mixture fit to each reference
        method = 'llr_pair': first-order Markov version; per-ISI log-likelihood ratio of
                        p(ISI_n | ISI_{n-1}) from 2D log-ISI mixtures (K = 1-3, BIC) fitted to
                        consecutive pairs; captures burst structure. n_isi and stride unused for both LLR methods
    """
    conditions = [(v, h, f) for v in var for h in d_H for f in f_values]

    results = Parallel(n_jobs = -1)(
        delayed(_cusum_condition)([spikes[(rep, v, h, f)] for rep in range(N_repeats)],
                                  h, changepoint, n_isi, stride, t_discard, method)
        for v, h, f in tqdm(conditions)
    )

    # Fisher information across step heights, one curve per (var, f)
    fisher = {}
    for v in var:
        for f in f_values:
            if method == 'ks':
                fisher.update({(v, h, f): (np.nan, np.nan) for h in d_H})
                continue
            post_fits = {h: res[4] for (cv, h, cf), res in zip(conditions, results) if cv == v and cf == f}
            for h, I in _fisher_across_steps(d_H, post_fits, pairs = method == 'llr_pair').items():
                fisher[(v, h, f)] = I

    window = n_isi if method == 'ks' else np.nan
    rows = []
    for (v, h, f), (res, name_pre, name_post, div, _) in zip(conditions, results):
        for rep, (CP, score, score_s) in enumerate(res):
            rows.append([rep, v, h, f, window, score, CP, method, name_pre, name_post, score_s,
                         div['divergence'], div['divergence per s'], div['KL post||pre'], div['KL pre||post'],
                         div['JS divergence'], div['fisher local'], div['fisher local per s'], *fisher[(v, h, f)],
                         *[div[k] for k in EMPIRICAL_KEYS]])

    # LLR methods only (NaN for 'ks'), all in nats:
    #   score per s:        evidence rate in this trial, mean(l) x firing rate on each side of the CP
    #   divergence:         Jeffreys divergence of the fitted pre/post distributions, per ISI (unclipped)
    #   divergence per s:   the same weighted by the pre/post firing rates
    #   KL post||pre:       KL(post || pre) per ISI, averaged over the post reference
    #   KL pre||post:       KL(pre || post) per ISI, averaged over the pre reference
    #   JS divergence:      Jensen-Shannon divergence of the pre/post distributions, per ISI (<= ln 2)
    #   fisher local:       Fisher information about the input current, divergence / h^2 [nats/pA^2] (NaN at h = 0)
    #   fisher local per s: the same from divergence per s
    #   fisher:             Fisher information at I_0 + h from the post distributions of neighbouring step heights
    #   fisher per s:       the same from divergence per s
    # All methods (model-free, from the pooled pre/post ISIs):
    #   KS distance:            two-sample KS statistic (0-1)
    #   energy distance:        energy distance of the ISIs [ms]
    #   energy distance (log):  energy distance of log(ISI), scale-free
    cols = ['N', 'variance', 'step_height', 'autapse f', 'ISI window', 'score', 'change point',
            'method', 'pre dist', 'post dist', 'score per s', 'divergence', 'divergence per s',
            'KL post||pre', 'KL pre||post', 'JS divergence', 'fisher local', 'fisher local per s',
            'fisher', 'fisher per s', *EMPIRICAL_KEYS]
    return pd.DataFrame(rows, columns = cols)


if __name__ == "__main__":
    main()
