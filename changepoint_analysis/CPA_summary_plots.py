"""
The key ISI and population-PSTH change point figures in one script (code taken from CPA_divergences_CUSUM_mean,
CPA_method_comparison_ISI, CPA_CP_estimate_CUSUM_mean, CPA_population and CPA_population_libraries).
Run from changepoint_analysis/. Figures go to {OUT}/{tag}_{CHANGE}_<name>.png.
"""
import inspect
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
warnings.filterwarnings('ignore')

from CPA_with_AQUA import *
from aqua.plotting_functions import plot_raster

tag, CHANGE = 'N250', 'mean'           # CHANGE: 'mean' or 'variance'
ISI_METHODS = ['llr_pair', 'edivisive_isi'] #, 'ks', 'npid']
PSTH_METHODS = ['edivisive'] #, 'llr', 'esst']
PSTH_BIN, K, BIN = 25, 10, 50          # ms PSTH bin of the detectors, trials per PSTH, ms bin of the raster PSTH
L2_THRS = [100, 200, 300, 400, 500]    # ms
EXAMPLE_F = [0, 240, 260, 300, 350]    # raster columns
LOGISI_F = [0, 200, 300, 400]          # log-ISI columns
OUT = 'summary_plots'

sim = {k: p.default for k, p in inspect.signature(simulate_runs).parameters.items()
       if p.default is not inspect.Parameter.empty}

with open(f'CPA_cusum_{CHANGE}_{tag}_spikes.pickle', 'rb') as file:
    saved = pickle.load(file)
assert saved.get('change', 'mean') == CHANGE
T, true_cp = saved['T'], saved['changepoint']      # true_cp in the time frame the detectors see
T_DISCARD = max(200, saved.get('t_warmup', 0))     # ms, initial transient and any warm-up
print(f"{tag} {CHANGE}: {saved['N_repeats']} repeats, T = {T} ms, change point at {true_cp} ms, discard {T_DISCARD} ms")


def title_mean(f, v, h):
    return f'f = {f} pA, e = {sim["e"]}, τ = {sim["tau"]} ms\nstep h = {h} pA, var = {v} pA²'


def title_var(f, v, dv):
    return f'f = {f} pA, e = {sim["e"]}, τ = {sim["tau"]} ms\nσ² = {v} → {v + dv} pA² (Δσ² = {dv})'


if CHANGE == 'mean':
    STEP, STEP_LABEL, title_cp = 'step_height', 'Step Height [pA]', title_mean
    RASTER_EXAMPLE, LOGISI_EXAMPLES = (25.0, 10), [(25.0, 10), (12.5, 20)]
else:
    STEP, STEP_LABEL, title_cp = 'variance_step', 'Variance jump Δσ² [pA²]', title_var
    RASTER_EXAMPLE, LOGISI_EXAMPLES = (12.5, 50), [(12.5, 50), (25.0, 100)]

f_order = np.sort(saved['f_values'])[::-1].tolist()     # f = 0 (darkest) drawn last, on top
f_lim = dict(vmin = min(saved['f_values']), vmax = max(saved['f_values']))


def load(path):
    try:
        with open(path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f'missing: {path}')


def save(fig, name):
    path = f'{OUT}/{tag}_{CHANGE}_{name}.png'
    fig.savefig(path, bbox_inches = 'tight')
    plt.close(fig)
    print(f'saved {path}')


# ---------- shared figure builders ----------

def distance_vs_f(cond, measures, title):
    '''model-free pre/post distances averaged over change sizes, against f (one panel per measure)'''
    fig, ax = plt.subplots(1, len(measures), figsize = (6*len(measures), 4.5), squeeze = False)
    ax = ax[0]
    for n, (y, (label, log)) in enumerate(measures.items()):
        auc = curve_auc(cond.dropna(subset = [y]), x = STEP, y = y, group_cols = ('autapse f', 'variance'), normalise = True)
        fig, ax[n] = auc_plot(auc, fig = fig, ax = ax[n], errorbar = None, add_colorbar = n == len(measures) - 1)
        ax[n].set_title(y)
        ax[n].set_ylabel(f'mean {label}')
        ax[n].set_xlabel('Autapse strength f [pA]')
        if log:
            ax[n].set_yscale('log')
    fig.suptitle(title, y = 1.03)
    fig.tight_layout()
    return fig


def l2_figure(norms, title):
    '''L2 CP error against change size: one row per method, one column per noise variance'''
    variances = np.sort(next(iter(norms.values()))['variance'].unique())
    fig, ax = plt.subplots(len(norms), len(variances), figsize = (4*len(variances), 3.8*len(norms)),
                           sharey = True, sharex = True, squeeze = False)
    for r, (m, df) in enumerate(norms.items()):
        for n, v in enumerate(variances):
            fig, ax[r, n] = psychometric_curve(df[df['variance'] == v], x = STEP, y = 'L2 |error|', hue_order = f_order,
                                               fig = fig, ax = ax[r, n], add_colorbar = n == len(variances) - 1)
            ax[r, n].set_ylabel(f'{m}\nL2: RMS |CP error| [ms]' if n == 0 else '')
            ax[r, n].set_title(f'Variance = {v}' if r == 0 else '')
            ax[r, n].set_xlabel(STEP_LABEL if r == len(norms) - 1 else '')
    fig.suptitle(title, y = 1.01)
    fig.tight_layout()
    return fig


def min_step_figure(norms, title):
    '''
    Rows: L2 thresholds. Per method two columns: smallest change size from which L2 stays below the threshold,
    against variance (min_step_below; NaN = never), and its mean ± SD over variance against f.
    '''
    fig, ax = plt.subplots(len(L2_THRS), 2*len(norms), figsize = (4.5*2*len(norms), 3.4*len(L2_THRS)),
                           sharey = 'row', squeeze = False)
    for r, thr in enumerate(L2_THRS):
        for n, (m, df) in enumerate(norms.items()):
            h_min = min_step_below(df, thr, step_col = STEP)
            a, b = ax[r, 2*n], ax[r, 2*n + 1]
            fig, a = psychometric_curve(h_min, x = 'variance', y = STEP, hue_order = f_order, fig = fig, ax = a,
                                        add_colorbar = False)
            a.set_title(f'{m}, L2 < {thr} ms')
            a.set_xlabel('Noise variance [pA²]' if r == len(L2_THRS) - 1 else '')
            a.set_ylabel(f'min {STEP_LABEL}' if n == 0 else '')

            avg = h_min.groupby('autapse f')[STEP].agg(['mean', 'std'])
            b.errorbar(avg.index, avg['mean'], yerr = avg['std'], fmt = 'none', ecolor = 'grey', capsize = 3, zorder = 2)
            sc = b.scatter(avg.index, avg['mean'], c = avg.index, cmap = 'viridis', **f_lim, zorder = 3)
            b.set_title(f'{m}, mean ± SD over variance')
            b.set_xlabel('Autapse strength f [pA]' if r == len(L2_THRS) - 1 else '')
        fig.colorbar(sc, ax = ax[r, -1], label = 'Autapse Strength')
    fig.suptitle(title, y = 1.01)
    fig.tight_layout()
    return fig


# ---------- 1. ISI methods ----------

isi = {m: d for m in ISI_METHODS if (d := load(f'CPA_cusum_{CHANGE}_{tag}_{m}.pickle')) is not None}

# divergences are the same for every repeat (and method): one row per condition; var = 0 fits are degenerate
isi_cond = None
if 'llr_pair' in isi:
    d = isi['llr_pair']
    isi_cond = d[d['variance'] != 0.0].drop_duplicates(['variance', STEP, 'autapse f']).reset_index(drop = True)
    fig = distance_vs_f(isi_cond, {'KS distance':           ('KS distance', False),
                                   'energy distance':       ('energy distance [ms]', True),
                                   'energy distance (log)': ('energy distance, log ISI', False)},
                        'Pooled pre/post ISI distances, averaged over change sizes')
    save(fig, 'ISI_empirical_distance_AUC_vs_f')

# pooled pre/post log-ISI histograms
fig, ax = plt.subplots(len(LOGISI_EXAMPLES), len(LOGISI_F), figsize = (4*len(LOGISI_F), 3.6*len(LOGISI_EXAMPLES)),
                       sharex = True, squeeze = False)
for r, (v, h) in enumerate(LOGISI_EXAMPLES):
    for c, f in enumerate(LOGISI_F):
        trials = [spike_isis(saved['spikes'][(rep, v, h, f)], T_DISCARD) for rep in range(saved['N_repeats'])]
        pre, post = reference_isis(trials, true_cp)
        bins = np.linspace(np.log10(min(pre.min(), post.min())), np.log10(max(pre.max(), post.max())), 60)
        ax[r, c].hist(np.log10(pre), bins = bins, density = True, alpha = 0.6, color = plt.cm.viridis(0.15), label = f'pre (n = {len(pre)})')
        ax[r, c].hist(np.log10(post), bins = bins, density = True, alpha = 0.6, color = plt.cm.viridis(0.85), label = f'post (n = {len(post)})')
        ax[r, c].set_xlabel('log10 ISI [ms]')
        ax[r, c].set_title(title_cp(f, v, h), fontsize = 10)
        if isi_cond is not None:
            row = isi_cond[(isi_cond['variance'] == v) & (isi_cond[STEP] == h) & (isi_cond['autapse f'] == f)]
            if len(row):
                row = row.iloc[0]
                ax[r, c].text(0.97, 0.97, f"JS = {row['JS divergence']:.3f}\nJ = {row['divergence']:.2f}\n"
                                          f"KS = {row['KS distance']:.3f}",
                              transform = ax[r, c].transAxes, ha = 'right', va = 'top', fontsize = 9,
                              bbox = dict(facecolor = 'white', alpha = 0.8, edgecolor = 'none'))
        ax[r, c].legend(loc = 'upper left', fontsize = 8)
    ax[r, 0].set_ylabel('density [per log10 ms]')
fig.suptitle(f'Pooled pre/post ISIs ({saved["N_repeats"]} repeats), I_0 = {sim["I_0"]} pA, '
             f'noise resampled at {sim["noise_rate"]} Hz, change at {true_cp} ms', y = 1.01)
fig.tight_layout()
save(fig, 'ISI_pooled_logISI_distributions')

if isi:
    isi_norms = {m: cp_error_norms(d, true_cp, group_cols = ('autapse f', 'variance', STEP)) for m, d in isi.items()}
    save(l2_figure(isi_norms, 'ISI methods: L2 change point error'), 'ISI_methods_L2')
    save(min_step_figure(isi_norms, 'ISI methods: minimum change size with RMS |CP error| below threshold'),
         'ISI_methods_min_step_L2')


# ---------- 2. Population PSTH ----------

def spike_matrix(saved, v, h, f):
    '''NaN-padded (repeats x spikes) array of spike times [ms] for one condition, for plot_raster'''
    rows = [np.asarray(saved['spikes'][(rep, v, h, f)], dtype = float) for rep in range(saved['N_repeats'])]
    rows = [r[~np.isnan(r)] for r in rows]
    S = np.full((len(rows), max(len(r) for r in rows)), np.nan)
    for i, r in enumerate(rows):
        S[i, :len(r)] = r
    return S


def psth(S, T, bin_ms):
    '''trial-averaged firing rate [Hz]: bin centres [ms] and rate'''
    edges = np.arange(0, T + bin_ms, bin_ms)
    counts, _ = np.histogram(S[~np.isnan(S)], bins = edges)
    return 0.5*(edges[:-1] + edges[1:]), counts / (S.shape[0] * bin_ms/1000)


def mean_rate(S, t0, t1):
    '''mean rate [Hz] over all trials in [t0, t1)'''
    s = S[~np.isnan(S)]
    return np.sum((s >= t0) & (s < t1)) / (S.shape[0] * (t1 - t0)/1000)


# raster + PSTH example
v, h = RASTER_EXAMPLE
fig, ax = plt.subplots(3, len(EXAMPLE_F), figsize = (4*len(EXAMPLE_F), 8.5), sharey = 'row',
                       gridspec_kw = {'height_ratios': [3, 1, 1.4]}, squeeze = False)
for c, f in enumerate(EXAMPLE_F):
    S = spike_matrix(saved, v, h, f)
    fig, ax[0, c] = plot_raster(S, T, ax = ax[0, c], linewidth = 0.4)
    ax[0, c].set_xlabel('')
    ax[0, c].tick_params(labelbottom = False)
    ax[0, c].set_ylabel('Trial' if c == 0 else '')
    ax[0, c].grid(False)
    ax[0, c].set_title(title_cp(f, v, h), fontsize = 10)

    t, rate = psth(S, T, BIN)
    ax[1, c].step(t, rate, where = 'mid', color = 'k', lw = 1)
    pre, post = mean_rate(S, T_DISCARD, true_cp), mean_rate(S, true_cp, T)
    ax[1, c].axhline(pre, color = plt.cm.viridis(0.15), ls = '--', lw = 1.2)
    ax[1, c].axhline(post, color = plt.cm.viridis(0.85), ls = '--', lw = 1.2)
    ax[1, c].text(0.03, 0.95, f'pre {pre:.1f} Hz\npost {post:.1f} Hz', transform = ax[1, c].transAxes,
                  ha = 'left', va = 'top', fontsize = 8, bbox = dict(facecolor = 'white', alpha = 0.8, edgecolor = 'none'))
    ax[1, c].set_xlabel('Time [ms]')
    ax[1, c].set_ylabel(f'Rate [Hz]\n({BIN} ms bins)' if c == 0 else '')
    for a in ax[:2, c]:
        a.axvline(true_cp, color = 'r', ls = '--', lw = 1)
        a.axvspan(0, T_DISCARD, color = 'grey', alpha = 0.3, lw = 0)
        a.set_xlim(0, T)

    # pre/post spike counts per PSTH_BIN bin of k-trial PSTHs, the data the PSTH detectors see
    # (same distribution as psth_cusum_analysis, not the identical resamples)
    trials = [saved['spikes'][(rep, v, h, f)] for rep in range(saved['N_repeats'])]
    X, t_bin = psth_resamples(trials, K, 100, PSTH_BIN, T, T_DISCARD, np.random.default_rng([0, c]))
    pre, post = reference_bins(X, t_bin, PSTH_BIN, true_cp)
    bins = np.arange(min(pre.min(), post.min()) - 0.5, max(pre.max(), post.max()) + 1.5)
    ax[2, c].hist(pre, bins = bins, density = True, alpha = 0.6, color = plt.cm.viridis(0.15), label = f'pre (mean {pre.mean():.1f})')
    ax[2, c].hist(post, bins = bins, density = True, alpha = 0.6, color = plt.cm.viridis(0.85), label = f'post (mean {post.mean():.1f})')
    ax[2, c].set_xlabel(f'spike count per {PSTH_BIN} ms bin (k = {K} trials)')
    ax[2, c].set_ylabel('density' if c == 0 else '')
    ax[2, c].legend(fontsize = 8)
fig.suptitle(f'{CHANGE.capitalize()} change: {saved["N_repeats"]} repeats, I_0 = {sim["I_0"]} pA, '
             f'noise resampled at {sim["noise_rate"]} Hz', y = 1.02)
fig.tight_layout()
save(fig, 'PSTH_raster_example')

psth_res = {m: d for m in PSTH_METHODS if (d := load(f'CPA_psth_{CHANGE}_{tag}_{m}_bin{PSTH_BIN}.pickle')) is not None}
if psth_res:
    cond = next(iter(psth_res.values())).drop_duplicates(['autapse f', 'variance', STEP])
    fig = distance_vs_f(cond, {'KS distance':           ('KS distance', False),
                               'energy distance':       ('energy distance [counts]', False),
                               'energy distance (log)': ('energy distance, log counts', False)},
                        f'Pre/post spike count distances (k = {K}, {PSTH_BIN} ms bins), averaged over change sizes')
    save(fig, 'PSTH_empirical_distance_vs_f')

    psth_norms = {m: cp_error_norms(r, true_cp, group_cols = ('autapse f', 'variance', STEP)) for m, r in psth_res.items()}
    save(l2_figure(psth_norms, f'PSTH: L2 change point error (k = {K}, {PSTH_BIN} ms bins)'), 'PSTH_L2')

    fig, ax = plt.subplots(1, len(psth_norms), figsize = (4.5*len(psth_norms), 3.8), sharey = True, squeeze = False)
    ax = ax[0]
    for n, (m, df) in enumerate(psth_norms.items()):
        avg = curve_auc(df[df[STEP] > 0], x = STEP, y = 'L2 |error|', group_cols = ('autapse f', 'variance'), normalise = True)
        fig, ax[n] = auc_plot(avg, y = 'AUC', fig = fig, ax = ax[n], errorbar = None, add_colorbar = n == len(psth_norms) - 1)
        ax[n].set_title(m)
        ax[n].set_xlabel('Autapse strength f [pA]')
    ax[0].set_ylabel('mean L2 |CP error| [ms]')
    fig.suptitle('PSTH: L2 error averaged over change sizes > 0', y = 1.03)
    save(fig, 'PSTH_L2_vs_f')

    save(min_step_figure(psth_norms, 'PSTH: minimum change size with RMS |CP error| below threshold'), 'PSTH_min_step_L2')

# PSTH rate change and SNR
rows = []
for v in saved['var']:
    for h in saved['d_H']:
        for f in saved['f_values']:
            t, rate = psth(spike_matrix(saved, v, h, f), T, BIN)
            pre, post = rate[(t > T_DISCARD) & (t < true_cp)], rate[t > true_cp]
            d = post.mean() - pre.mean()
            sd = np.sqrt((pre.var() + post.var())/2)
            rows.append([v, h, f, d, d/sd if sd > 0 else np.nan])
rates = pd.DataFrame(rows, columns = ['variance', STEP, 'autapse f', 'Δ rate', 'SNR'])

fig, ax = plt.subplots(1, 2, figsize = (11, 4))
for n, (y, ylabel) in enumerate([('Δ rate', 'mean post − pre rate [Hz]'), ('SNR', 'mean PSTH SNR\n(Δ rate / pooled SD)')]):
    avg = curve_auc(rates[rates[STEP] > 0], x = STEP, y = y, group_cols = ('autapse f', 'variance'), normalise = True)
    fig, ax[n] = auc_plot(avg, y = 'AUC', fig = fig, ax = ax[n], errorbar = None, add_colorbar = n == 1)
    ax[n].set_title(f'{y}, averaged over change sizes > 0')
    ax[n].set_xlabel('Autapse strength f [pA]')
    ax[n].set_ylabel(ylabel)
fig.suptitle(f'{CHANGE.capitalize()} change: PSTH rate change ({saved["N_repeats"]} repeats, {BIN} ms bins)', y = 1.03)
fig.tight_layout()
save(fig, 'PSTH_rate_change_vs_f')
