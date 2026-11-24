import numpy as np
import pandas as pd
import stumpy
from stumpy.floss import _iac
from functools import lru_cache
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns


def find_idx(f, v, h, f_vals, v_vals, h_vals):
    """finds the index of a given set of parameters values
    given the list of those parameter values which
    defined the population simulations.
    Params:
        f, v, h:    target parameter values
        *_vals:     reference arrays in which to find targets
    
    """
    N_f = len(f_vals)
    N_v = len(v_vals)
    N_h = len(h_vals)

    idx_f = np.argwhere(f_vals == f)[0][0]
    idx_v = np.argwhere(v_vals == v)[0][0]
    idx_h = np.argwhere(h_vals == h)[0][0]

    idx = idx_v * (N_f*N_h) + idx_h * N_f + idx_f

    return int(idx)


@lru_cache(maxsize = None)
def _fluss_iac(width):
    '''stumpy's idealised arc curve: depends only on the width (and a fixed seed), and takes ~0.7 s, so cache it'''
    return _iac(width)


def fluss_cp(x, window_size, dt):
    '''
    FLUSS on any 1D series (V trace, PSTH, ...). The score is 1 - corrected arc curve.
    OUT
        CAC:    peak score (confidence)
        CP:     index of the peak x dt (time from the start of x)
    '''
    mat_prof = stumpy.stump(np.asarray(x, dtype = float), m = window_size)
    I = mat_prof[:, 1].astype(np.int64)
    arc_crossings, _ = stumpy.fluss(I, L = window_size, n_regimes = 1, custom_iac = _fluss_iac(len(I)))
    score = 1 - arc_crossings
    return np.max(score), np.argmax(score)*dt


def process_single_row(row_idx, data_matrix, window_size, dt):
    '''
    Computes the matrix profile on a single row and returns the CAC
    '''
    return fluss_cp(data_matrix[row_idx, 0, :], window_size, dt)


def psychometric_curve(data, x, y, hue = 'autapse f', fig = None, ax = None, add_colorbar = True, colorbar_label = 'Autapse Strength', hue_order = None):
    '''
    Plot the psychometric curve for different autapse conditions
    hue_order sets the drawing order (last drawn is on top); colours still follow the hue value
    '''
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (5, 5))

    # 1. Define your colormap
    cmap_name = "viridis"
    norm = mcolors.Normalize(vmin=data[hue].min(), vmax=data[hue].max())

    #palette = {val: color for val, color in zip([s for s in autapses if s != '0.0'], default_colors)}
    #palette['0.0'] = 'black'  # Your pre-defined color

    # 3. Plot
    palette = cmap_name
    if hue_order is not None:
        # seaborn ignores hue_order for numeric hue unless the palette is an explicit dict
        cmap = plt.get_cmap(cmap_name)
        palette = {v: cmap(norm(v)) for v in hue_order}
    sns.lineplot(data=data, x=x, y=y, hue=hue, palette=palette, hue_order=hue_order, ax = ax, legend = False,
                 estimator = 'mean', errorbar = 'sd', err_style = 'band', alpha = 0.8)
    #ax.set_ylim((0, 1))

    if add_colorbar:
        # 3. Create the Colorbar
        sm = cm.ScalarMappable(cmap=cmap_name, norm=norm)
        sm.set_array([])  # Dummy array for the mappable

        # Add the colorbar to the figure
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(colorbar_label)

    return fig, ax


def changepoint_distribution(data, x = 'change point', row = 'autapse f', col = 'variance', fig = None, ax = None,
                             bins = 60, true_cp = None, add_colorbar = True, colorbar_label = 'Autapse Strength'):
    '''
    Grid of histograms of the estimated change point times. One row per value of `row`
    (autapse strength), one column per value of `col` (noise variance). Histograms are
    coloured by the row value. Pools all step heights and repeats: filter `data` first
    to look at a single step height.

    Returns fig, ax (2D array of axes).
    '''
    row_vals = np.sort(data[row].unique())
    col_vals = np.sort(data[col].unique())

    if fig is None or ax is None:
        fig, ax = plt.subplots(len(row_vals), len(col_vals), figsize = (3*len(col_vals), 1.6*len(row_vals)),
                               sharex = True, sharey = True, squeeze = False)
    ax = np.atleast_2d(ax)

    cmap_name = "winter"
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin = data[row].min(), vmax = data[row].max())

    # same bins for every panel
    edges = np.histogram_bin_edges(data[x], bins = bins)

    for i, r in enumerate(row_vals):
        for j, c in enumerate(col_vals):
            sub = data[(data[row] == r) & (data[col] == c)]
            sns.histplot(sub[x], bins = edges, color = cmap(norm(r)), ax = ax[i, j])

            if true_cp is not None:
                ax[i, j].axvline(true_cp, color = 'red', linestyle = '--', linewidth = 1)

            ax[i, j].set_xlabel('')
            ax[i, j].set_ylabel('')
            if i == 0:
                ax[i, j].set_title(f'{col.capitalize()} = {c}')
            if j == 0:
                ax[i, j].set_ylabel(f'f = {r} pA')
            if i == len(row_vals) - 1:
                ax[i, j].set_xlabel('Change point [ms]')

    if add_colorbar:
        sm = cm.ScalarMappable(cmap = cmap_name, norm = norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax = ax)
        cbar.set_label(colorbar_label)

    return fig, ax


def curve_auc(data, y = 'score', x = 'step_height', group_cols = ('autapse f', 'variance', 'N'),
              extra_cols = None, normalise = True, out_col = 'AUC'):
    '''
    Area under the `y` vs `x` curve (trapezoid rule) for every group in `group_cols`,
    by default one value per neuron (autapse f), noise variance and repeat trial (N).

    Params:
        y, x:        columns of the curve, e.g. 'score' vs 'step_height'
        group_cols:  columns defining one curve
        extra_cols:  extra columns to carry into the output. Numeric columns are
                     averaged over the curve (e.g. 'divergence' changes with step
                     height); other columns take their first value (e.g. 'method').
        normalise:   divide by the x range. For a 0-1 score this is the fraction of
                     the maximum area; for an unbounded score (LLR) it is the mean
                     score across x.
        out_col:     name of the area column

    Returns a dataframe with one row per group: group_cols + extra_cols + [out_col].
    '''
    group_cols = list(group_cols)
    extra_cols = list(extra_cols) if extra_cols is not None else []

    rows = []
    for keys, sub in data.groupby(group_cols, sort = False):
        sub = sub.sort_values(x)
        area = np.trapezoid(y = sub[y], x = sub[x])
        if normalise:
            area /= sub[x].max() - sub[x].min()

        row = dict(zip(group_cols, keys))
        for c in extra_cols:
            row[c] = sub[c].mean() if pd.api.types.is_numeric_dtype(sub[c]) else sub[c].iloc[0]
        row[out_col] = area
        rows.append(row)

    return pd.DataFrame(rows)


def weighted_cp_distance(data, true_cp, cp_col = 'change point',
                         group_cols = ('autapse f', 'variance', 'step_height'), eps = 1.):
    '''
    Inverse-distance-weighted accuracy of the change point estimates for each condition
    (by default one per autapse f / variance / step height, pooling the repeat trials N).
    Each estimate gets weight w = 1 / (|CP - true_cp| + eps); eps (ms) avoids dividing by
    zero for an exact hit.

    Returns a dataframe with one row per group: group_cols +
        'weighted CP'       sum(w * CP) / sum(w)                  [ms]
        'weighted CP dist'  |weighted CP - true_cp|               [ms]
        'weighted |error|'  sum(w * |CP - true_cp|) / sum(w)      [ms], lower = closer
        'n'                 number of estimates used (NaN change points dropped)

    Note: 'weighted |error|' is about the harmonic mean of |CP - true_cp|, so a few near-exact
    hits dominate it, and 'weighted CP' is pulled towards true_cp by construction.
    '''
    group_cols = list(group_cols)

    rows = []
    for keys, sub in data.dropna(subset = [cp_col]).groupby(group_cols, sort = False):
        cp = sub[cp_col].to_numpy(dtype = float)
        err = np.abs(cp - true_cp)
        w = 1. / (err + eps)
        w_cp = np.average(cp, weights = w)

        row = dict(zip(group_cols, keys))
        row['weighted CP'] = w_cp
        row['weighted CP dist'] = abs(w_cp - true_cp)
        row['weighted |error|'] = np.average(err, weights = w)
        row['n'] = len(cp)
        rows.append(row)

    return pd.DataFrame(rows)


def cp_error_norms(data, true_cp, cp_col = 'change point',
                   group_cols = ('autapse f', 'variance', 'step_height')):
    '''
    L1 and L2 norms of the change point errors e = |CP - true_cp| for each condition
    (by default one per autapse f / variance / step height, pooling the repeat trials N),
    normalised by the number of estimates n so conditions with dropped NaNs stay comparable.

    Returns a dataframe with one row per group: group_cols +
        'L1 |error|'  sum(e) / n            mean absolute error      [ms]
        'L2 |error|'  sqrt(sum(e**2) / n)   root-mean-square error   [ms]
        'n'           number of estimates used (NaN change points dropped)
    '''
    group_cols = list(group_cols)

    rows = []
    for keys, sub in data.dropna(subset = [cp_col]).groupby(group_cols, sort = False):
        err = np.abs(sub[cp_col].to_numpy(dtype = float) - true_cp)

        row = dict(zip(group_cols, keys))
        row['L1 |error|'] = np.mean(err)
        row['L2 |error|'] = np.sqrt(np.mean(err**2))
        row['n'] = len(err)
        rows.append(row)

    return pd.DataFrame(rows)


def min_step_below(norms, threshold, step_col = 'step_height', y = 'L2 |error|',
                   group_cols = ('autapse f', 'variance')):
    '''
    Smallest step > 0 from which the error `y` (from cp_error_norms) stays below `threshold`
    for every larger step, per group. NaN for a group that never gets there.
    Returns a dataframe: group_cols + step_col.
    '''
    def first_step(g):
        g = g[g[step_col] > 0].sort_values(step_col)
        ok = (g[y] < threshold).astype(int)[::-1].cummin()[::-1].astype(bool)
        return g.loc[ok, step_col].min()
    return norms.groupby(list(group_cols)).apply(first_step).rename(step_col).reset_index()


def mean_isi(spikes, t_discard = 200., step_col = 'step_height'):
    '''
    Mean ISI of each condition, pooling the ISIs of all repeat trials.
        spikes:     dict {(rep, variance, step_height, autapse f): spike times [ms]}, as saved in
                    the '_spikes.pickle' from simulate_w_cusum
        t_discard:  spikes before this time [ms] are ignored (same default as cusum_analysis)
        step_col:   column name for the second key ('variance_step' for a variance change point)

    Returns a dataframe with columns variance, step_col, autapse f, 'mean ISI' [ms].
    '''
    pooled = {}
    for (rep, v, h, f), s in spikes.items():
        pooled.setdefault((v, h, f), []).append(spike_isis(s, t_discard)[0])

    rows = [{'variance': v, step_col: h, 'autapse f': f, 'mean ISI': np.mean(np.concatenate(isis))}
            for (v, h, f), isis in pooled.items()]
    return pd.DataFrame(rows)


def _abs_errors(sub, true_cp, cp_col):
    ''' |CP - true_cp| with NaN change points (no detection) as inf, i.e. never within tolerance '''
    err = np.abs(sub[cp_col].to_numpy(dtype = float) - true_cp)
    return np.where(np.isnan(err), np.inf, err)


def hit_rate_curves(data, true_cp, tols, isi_df = None, cp_col = 'change point',
                    group_cols = ('autapse f', 'variance', 'step_height')):
    '''
    Fraction of trials whose change point estimate is within a tolerance of true_cp,
    P(|CP - true_cp| <= tol), for each condition and each tolerance in `tols`.
        tols:    tolerances in ms, or (with isi_df) in multiples of the condition's mean ISI
        isi_df:  output of mean_isi; scales the tolerance to each condition's firing rate

    Returns a long dataframe: group_cols + 'tol' (as given) + 'tol [ms]' + 'hit rate'.
    The area under it is curve_auc(curves, y = 'hit rate', x = 'tol', group_cols = group_cols).
    '''
    group_cols = list(group_cols)
    tols = np.asarray(tols, dtype = float)
    if isi_df is not None:
        keys = [c for c in isi_df.columns if c != 'mean ISI']
        data = data.merge(isi_df, on = keys, how = 'left')

    frames = []
    for keys, sub in data.groupby(group_cols, sort = False):
        err = _abs_errors(sub, true_cp, cp_col)
        tol_ms = tols * sub['mean ISI'].iloc[0] if isi_df is not None else tols

        df = pd.DataFrame({'tol': tols, 'tol [ms]': tol_ms,
                           'hit rate': (err[None, :] <= tol_ms[:, None]).mean(axis = 1)})
        for c, k in zip(group_cols, keys):
            df[c] = k
        frames.append(df)

    return pd.concat(frames, ignore_index = True)[group_cols + ['tol', 'tol [ms]', 'hit rate']]


def tolerance_roc(data, true_cp, null_step = 0, cp_col = 'change point',
                  group_cols = ('autapse f', 'variance'), step_col = 'step_height'):
    '''
    ROC of change point localisation, sweeping the tolerance tol:
        TPR(tol) = P(|CP - true_cp| <= tol | step h)
        FPR(tol) = P(|CP - true_cp| <= tol | step null_step), the same neuron and noise with no
                   change, i.e. how often an estimate lands that close to true_cp by chance
    AUC = P(err_h < err_null) + P(err_h == err_null) / 2 (Mann-Whitney), exact. The curve and AUC
    do not depend on the tolerance units, so a constant and an ISI-scaled tolerance give the same ROC.
    step_col is the change size column ('variance_step' for a variance change point).

    Returns (roc_df, auc_df):
        roc_df:  group_cols + step_col + 'tol [ms]', 'TPR', 'FPR' (one row per threshold)
        auc_df:  group_cols + step_col + 'AUC'
    '''
    group_cols = list(group_cols)

    roc_frames, auc_rows = [], []
    for keys, sub in data.groupby(group_cols, sort = False):
        null = sub[sub[step_col] == null_step]
        if null.empty:
            continue
        err_0 = _abs_errors(null, true_cp, cp_col)

        for h, sub_h in sub[sub[step_col] != null_step].groupby(step_col):
            err_h = _abs_errors(sub_h, true_cp, cp_col)
            tols = np.unique(np.concatenate(([0., np.inf], err_0, err_h)))

            df = pd.DataFrame({'tol [ms]': tols,
                               'TPR': (err_h[None, :] <= tols[:, None]).mean(axis = 1),
                               'FPR': (err_0[None, :] <= tols[:, None]).mean(axis = 1)})
            row = {**dict(zip(group_cols, keys)), step_col: h}
            for c, k in row.items():
                df[c] = k
            roc_frames.append(df)

            diff = err_h[:, None] - err_0[None, :]
            with np.errstate(invalid = 'ignore'):   # inf - inf: both missed, count as a tie
                row['AUC'] = np.mean(diff < 0) + 0.5 * np.mean((diff == 0) | np.isnan(diff))
            auc_rows.append(row)

    cols = group_cols + [step_col]
    roc_df = pd.concat(roc_frames, ignore_index = True)[cols + ['tol [ms]', 'TPR', 'FPR']]
    return roc_df, pd.DataFrame(auc_rows)


def auc_plot(auc_df, x = 'autapse f', y = 'AUC', hue = 'variance', fig = None, ax = None,
             add_colorbar = True, colorbar_label = 'Noise Variance', errorbar = ('ci', 0.9)):
    '''
    Plot the area under the curve (from curve_auc) against `x`, one line per `hue`
    value, averaged over trials with an error band.
    '''
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (6, 5))

    cmap_name = 'viridis'
    norm = mcolors.Normalize(vmin = auc_df[hue].min(), vmax = auc_df[hue].max())

    sns.lineplot(data = auc_df, x = x, y = y, hue = hue, palette = cmap_name, ax = ax, legend = False,
                 estimator = 'mean', errorbar = errorbar, err_style = 'band', alpha = 0.8)
    ax.set_title('Area under the curve versus peak autapse current')

    if add_colorbar:
        sm = cm.ScalarMappable(cmap = cmap_name, norm = norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax = ax)
        cbar.set_label(colorbar_label)

    return fig, ax


# ---------------------------------------------------------------------------
# ISI / Kolmogorov-Smirnov changepoint detection (used by simulate_w_cusum)
# ---------------------------------------------------------------------------

def spike_isis(spike_row, t_discard = 0.):
    '''
    Converts a (NaN-padded) row of spike times into ISIs.
    Spikes before t_discard (initial transient) are dropped.
    OUT
        isi:    interspike intervals [ms]
        t_isi:  time of the spike that ends each ISI [ms]
    '''
    s = np.asarray(spike_row, dtype = float)
    s = s[~np.isnan(s)]
    s = s[s >= t_discard]
    return np.diff(s), s[1:]


def reference_isis(trials, t_cp):
    '''
    Pools ISIs over many trials of the same condition into pre- and post-changepoint
    reference distributions. ISIs straddling t_cp are dropped.
    IN
        trials: list of (isi, t_isi) tuples from spike_isis
    OUT
        pre_ref, post_ref:  sorted 1D arrays
    '''
    pre, post = [], []
    for isi, t_isi in trials:
        t_start = t_isi - isi
        pre.append(isi[t_isi < t_cp])
        post.append(isi[t_start >= t_cp])
    pre = np.sort(np.concatenate(pre)) if pre else np.array([])
    post = np.sort(np.concatenate(post)) if post else np.array([])
    return pre, post


def ks_stat_windows(windows, ref_sorted):
    '''
    Two-sample KS statistic between each row of `windows` (n_win, n) and one sorted
    reference sample. Exact (handles ties), vectorised over windows.
    '''
    ws = np.sort(windows, axis = 1)
    n = ws.shape[1]
    m = len(ref_sorted)
    # window ECDF just at (right) and just before (left) each of its own points
    Fw_right = (ws[:, None, :] <= ws[:, :, None]).sum(-1) / n
    Fw_left = (ws[:, None, :] < ws[:, :, None]).sum(-1) / n
    Fr_right = np.searchsorted(ref_sorted, ws, side = 'right') / m
    Fr_left = np.searchsorted(ref_sorted, ws, side = 'left') / m
    D = np.maximum(np.abs(Fw_right - Fr_right), np.abs(Fw_left - Fr_left))
    return D.max(axis = 1)


def ks_window_cp(isi, t_isi, pre_ref, post_ref, n_isi, stride = 1, return_trace = False):
    '''
    Slides a window of n_isi ISIs along one spike train. For each window
        d = KS(window, pre_ref) - KS(window, post_ref)     (> 0: more post-like)
    The changepoint is the argmin of cumsum(d): the split where windows before it are as
    pre-like (d < 0) and windows after it as post-like (d > 0) as possible. This is the
    zero crossing of d, but robust to d crossing zero several times. (Using d - mean(d)
    instead biases the CP towards the side with fewer windows, i.e. the lower firing rate.)
    OUT
        CP:     estimated changepoint time [ms]
        score:  mean(d after CP) - mean(d before CP); ~0 when there is no change
        if return_trace, also t_win, d, s (window centres, KS difference, CUSUM curve)
    '''
    n_win = (len(isi) - n_isi) // stride + 1 if len(isi) >= n_isi else 0
    if n_win < 3 or len(pre_ref) == 0 or len(post_ref) == 0:
        if return_trace:
            return np.nan, np.nan, np.array([]), np.array([]), np.array([])
        return np.nan, np.nan

    starts = np.arange(n_win) * stride
    idx = starts[:, None] + np.arange(n_isi)[None, :]
    windows = isi[idx]
    t_win = t_isi[starts + n_isi // 2]      # window centre

    d = ks_stat_windows(windows, pre_ref) - ks_stat_windows(windows, post_ref)
    s = np.cumsum(d)
    k = np.argmin(s[:-1])                   # last window before the change

    CP = 0.5 * (t_win[k] + t_win[k + 1])
    score = d[k + 1:].mean() - d[:k + 1].mean()
    if return_trace:
        return CP, score, t_win, d, s
    return CP, score


# ---------------------------------------------------------------------------
# Log-likelihood-ratio (Page) CUSUM on fitted ISI distributions
# ---------------------------------------------------------------------------
import scipy.stats as stats
from sklearn.mixture import GaussianMixture


class LogGMM:
    '''
    Gaussian mixture fitted to log(ISI), i.e. a mixture of lognormals in ISI space. Handles
    bimodal (bursting) ISI distributions. Exposes logpdf / pdf / cdf like a frozen scipy dist.
    '''
    def __init__(self, n_components = 2, n_init = 3, random_state = 0):
        self.n_components = n_components
        self.gmm = GaussianMixture(n_components, n_init = n_init, random_state = random_state)

    def fit(self, isi):
        self.gmm.fit(np.log(isi).reshape(-1, 1))
        order = np.argsort(self.gmm.means_.ravel())
        self.w = self.gmm.weights_[order]
        self.mu = self.gmm.means_.ravel()[order]
        self.sd = np.sqrt(self.gmm.covariances_.ravel()[order])
        return self

    @property
    def n_params(self):
        return 3 * self.n_components - 1            # means, variances, weights (sum to 1)

    @property
    def params(self):
        return tuple(np.round(np.column_stack([self.w, self.mu, self.sd]).ravel(), 4))

    def logpdf(self, x):
        x = np.asarray(x, dtype = float)
        logx = np.log(x)
        # density in ISI space: p(x) = p_log(log x) / x
        return self.gmm.score_samples(logx.reshape(-1, 1)).reshape(x.shape) - logx

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def cdf(self, x):
        z = (np.log(np.asarray(x, dtype = float))[..., None] - self.mu) / self.sd
        return (self.w * stats.norm.cdf(z)).sum(-1)


def fit_isi_distribution(isi, families = ('gamma', 'lognorm', 'invgauss', 'gmm2', 'gmm3')):
    '''
    Fits each candidate distribution to the ISIs by maximum likelihood and picks the best by BIC.
    Candidates:
        'gamma', 'lognorm', 'invgauss':  scipy.stats fits with location fixed at 0 (2 free params)
        'gmm2', 'gmm3':                  2- / 3-component Gaussian mixture on log(ISI) (5 / 8 free params),
                                         for bimodal / bursting ISI distributions
    BIC = k ln(n) - 2 logL, with logL in ISI space for all candidates, so they are comparable.
    OUT
        best:   fitted distribution with .logpdf/.pdf/.cdf (None if the input is degenerate or all fits fail)
        name:   name of the chosen family
        table:  DataFrame (family, params, logL, BIC, KS, dist) sorted by BIC; KS is for diagnostics only,
                dist holds each fitted object
    '''
    isi = np.asarray(isi, dtype = float)
    isi = isi[np.isfinite(isi) & (isi > 0)]
    cols = ['family', 'params', 'logL', 'BIC', 'KS', 'dist']

    if len(isi) < 10 or len(np.unique(isi)) < 3:      # e.g. var = 0: a few exact values
        table = pd.DataFrame([[f, None, np.nan, np.nan, np.nan, None] for f in families], columns = cols)
        return None, None, table

    n = len(isi)
    rows = []
    for name in families:
        try:
            if name.startswith('gmm'):
                dist = LogGMM(int(name[3:])).fit(isi)
                k, params = dist.n_params, dist.params
            else:
                family = getattr(stats, name)
                p = family.fit(isi, floc = 0)
                dist = family(*p)
                k, params = len(p) - 1, tuple(np.round(p, 4))   # loc is fixed, not a free parameter
            ll = dist.logpdf(isi).sum()
        except Exception:
            ll = np.nan
        if not np.isfinite(ll):
            rows.append([name, None, np.nan, np.nan, np.nan, None])
            continue
        bic = k * np.log(n) - 2 * ll
        ks = stats.kstest(isi, dist.cdf).statistic
        rows.append([name, params, ll, bic, ks, dist])

    table = pd.DataFrame(rows, columns = cols).sort_values('BIC', na_position = 'last', ignore_index = True)
    if table['dist'].iloc[0] is None:
        return None, None, table
    return table['dist'].iloc[0], table['family'].iloc[0], table


def llr_cp(isi, t_isi, dist_pre, dist_post, clip = 10, return_trace = False):
    '''
    Log-likelihood-ratio CUSUM changepoint. For every ISI
        l = log p_post(isi) - log p_pre(isi)        (> 0: more post-like), clipped to +-clip
    The changepoint is argmin of cumsum(l): the maximum-likelihood single change.
    OUT
        CP:     estimated changepoint time [ms] (end of the last pre-change ISI)
        score:  mean(l after CP) - mean(l before CP); ~0 when there is no change
        if return_trace, also t_isi, l, s
    '''
    if dist_pre is None or dist_post is None or len(isi) < 3:
        if return_trace:
            return np.nan, np.nan, np.array([]), np.array([]), np.array([])
        return np.nan, np.nan

    l = np.clip(dist_post.logpdf(isi) - dist_pre.logpdf(isi), -clip, clip)
    s = np.cumsum(l)
    k = np.argmin(s[:-1])                   # last ISI before the change

    CP = t_isi[k]
    score = l[k + 1:].mean() - l[:k + 1].mean()
    if return_trace:
        return CP, score, t_isi, l, s
    return CP, score


# ---------------------------------------------------------------------------
# ISI-pair (first-order Markov) log-likelihood-ratio CUSUM
# ---------------------------------------------------------------------------
from scipy.special import logsumexp


def isi_pairs(isi, t_isi):
    '''
    Consecutive ISI pairs within one trial.
    OUT
        prev, curr: ISI_{n-1} and ISI_n
        t_curr:     end time of ISI_n
    '''
    return isi[:-1], isi[1:], t_isi[1:]


def reference_pairs(trials, t_cp):
    '''
    Pools consecutive ISI pairs over many trials into pre- and post-changepoint references.
    A pair is pre if its later ISI ends before t_cp, post if its earlier ISI starts at/after t_cp;
    pairs straddling t_cp are dropped. Pairs never cross trial boundaries.
    IN
        trials: list of (isi, t_isi) tuples from spike_isis
    OUT
        pre, post:  arrays of shape (n, 2), columns (ISI_{n-1}, ISI_n)
    '''
    pre, post = [], []
    for isi, t_isi in trials:
        if len(isi) < 2:
            continue
        prev, curr, t_curr = isi_pairs(isi, t_isi)
        t_start = t_isi[:-1] - prev                     # start of the earlier ISI
        pair = np.column_stack([prev, curr])
        pre.append(pair[t_curr < t_cp])
        post.append(pair[t_start >= t_cp])
    empty = np.empty((0, 2))
    pre = np.concatenate(pre) if pre else empty
    post = np.concatenate(post) if post else empty
    return pre, post


class LogPairGMM:
    '''
    Gaussian mixture (full covariances) on (log ISI_{n-1}, log ISI_n): a first-order Markov
    model of the ISI sequence. All densities are in log-ISI space; the Jacobian terms cancel
    in the log-likelihood ratio.
    '''
    def __init__(self, n_components = 1, n_init = 3, random_state = 0):
        self.n_components = n_components
        self.gmm = GaussianMixture(n_components, covariance_type = 'full',
                                   n_init = n_init, random_state = random_state)

    def fit(self, pairs):
        self.gmm.fit(np.log(pairs))
        self.w = self.gmm.weights_
        self.mu0 = self.gmm.means_[:, 0]
        self.sd0 = np.sqrt(self.gmm.covariances_[:, 0, 0])
        return self

    @property
    def n_params(self):
        return 6 * self.n_components - 1            # 2 means + 3 covariance terms each, weights sum to 1

    def logpdf_joint(self, prev, curr):
        return self.gmm.score_samples(np.log(np.column_stack([prev, curr])))

    def logpdf_marg(self, prev):
        '''analytic marginal of log ISI_{n-1}: 1D mixture with the same weights'''
        a = np.log(np.asarray(prev, dtype = float))[:, None]
        return logsumexp(np.log(self.w) + stats.norm.logpdf(a, self.mu0, self.sd0), axis = 1)

    def logpdf_cond(self, prev, curr):
        '''log p(log ISI_n | log ISI_{n-1})'''
        return self.logpdf_joint(prev, curr) - self.logpdf_marg(prev)


def fit_isi_pair_distribution(pairs, ks = (1, 2, 3)):
    '''
    Fits 2D GMMs with K components to log ISI pairs and picks the best by BIC
    (BIC = k ln(n) - 2 logL, joint log-likelihood in log space).
    OUT
        best:   fitted LogPairGMM (None if the input is degenerate or all fits fail)
        name:   'pair_gmmK' of the chosen model
        table:  DataFrame (family, logL, BIC, dist) sorted by BIC
    '''
    pairs = np.asarray(pairs, dtype = float).reshape(-1, 2)
    pairs = pairs[np.all(np.isfinite(pairs) & (pairs > 0), axis = 1)]
    cols = ['family', 'logL', 'BIC', 'dist']
    names = [f'pair_gmm{k}' for k in ks]

    if len(pairs) < 20 or min(len(np.unique(pairs[:, 0])), len(np.unique(pairs[:, 1]))) < 3:
        return None, None, pd.DataFrame([[n, np.nan, np.nan, None] for n in names], columns = cols)

    n = len(pairs)
    rows = []
    for k, name in zip(ks, names):
        try:
            dist = LogPairGMM(k).fit(pairs)
            ll = dist.logpdf_joint(pairs[:, 0], pairs[:, 1]).sum()
        except Exception:
            ll = np.nan
        if not np.isfinite(ll):
            rows.append([name, np.nan, np.nan, None])
            continue
        rows.append([name, ll, dist.n_params * np.log(n) - 2 * ll, dist])

    table = pd.DataFrame(rows, columns = cols).sort_values('BIC', na_position = 'last', ignore_index = True)
    if table['dist'].iloc[0] is None:
        return None, None, table
    return table['dist'].iloc[0], table['family'].iloc[0], table


def llr_pair_cp(isi, t_isi, dist_pre, dist_post, clip = 10, return_trace = False):
    '''
    Markov (ISI-pair) log-likelihood-ratio CUSUM changepoint. For ISIs 2..n
        l = log p_post(ISI_n | ISI_{n-1}) - log p_pre(ISI_n | ISI_{n-1}),   clipped to +-clip
    The changepoint is argmin of cumsum(l). Same outputs as llr_cp; the first ISI is not scored.
    '''
    if dist_pre is None or dist_post is None or len(isi) < 4:
        if return_trace:
            return np.nan, np.nan, np.array([]), np.array([]), np.array([])
        return np.nan, np.nan

    prev, curr, t_curr = isi_pairs(isi, t_isi)
    l = np.clip(dist_post.logpdf_cond(prev, curr) - dist_pre.logpdf_cond(prev, curr), -clip, clip)
    s = np.cumsum(l)
    k = np.argmin(s[:-1])                   # last ISI before the change

    CP = t_curr[k]
    score = l[k + 1:].mean() - l[:k + 1].mean()
    if return_trace:
        return CP, score, t_curr, l, s
    return CP, score


# ---------------------------------------------------------------------------
# Evidence rates and divergences for the LLR methods (nats)
# ---------------------------------------------------------------------------

def evidence_rate(t, l, isi_scored, CP):
    '''
    Score per second for one trial: evidence accumulated per unit time on each side of the CP,
        sum(l after) / time after  -  sum(l before) / time before        [nats / s]
    = mean(l) x firing rate on each side.
    IN
        t, l:        trace from llr_cp / llr_pair_cp (end time and LLR of each scored ISI)
        isi_scored:  the ISIs those l values belong to [ms]
        CP:          estimated changepoint (= t of the last pre-change ISI)
    '''
    if len(l) < 2 or not np.isfinite(CP):
        return np.nan
    k = np.searchsorted(t, CP)
    before = l[:k + 1].sum() / isi_scored[:k + 1].sum()
    after = l[k + 1:].sum() / isi_scored[k + 1:].sum()
    return (after - before) * 1000.          # per ms -> per s


EMPIRICAL_KEYS = ('KS distance', 'energy distance', 'energy distance (log)')


def empirical_distances(pre_ref, post_ref):
    '''
    Model-free distances between the pooled pre and post ISIs (no fitted distributions):
        'KS distance'            two-sample KS statistic, 0-1
        'energy distance'        energy distance [ms]; scales with the ISIs, so it shrinks for faster firing
        'energy distance (log)'  energy distance of log(ISI), scale-free
    All values are NaN if either reference is empty.
    '''
    if len(pre_ref) == 0 or len(post_ref) == 0:
        return dict.fromkeys(EMPIRICAL_KEYS, np.nan)
    return {'KS distance': stats.ks_2samp(pre_ref, post_ref).statistic,
            'energy distance': stats.energy_distance(pre_ref, post_ref),
            'energy distance (log)': stats.energy_distance(np.log(pre_ref), np.log(post_ref))}


DIVERGENCE_KEYS = ('KL post||pre', 'KL pre||post', 'divergence', 'divergence per s', 'JS divergence')


def distribution_divergences(dist_pre, dist_post, pre_ref, post_ref, pairs = False, sample_ms = None):
    '''
    Divergences between the fitted pre and post distributions, estimated on the reference
    data (unclipped). With KL(post || pre) = mean over post_ref of [log p_post - log p_pre]
    and vice versa:
        'KL post||pre', 'KL pre||post'      nats per ISI
        'divergence'        Jeffreys J = KL(post||pre) + KL(pre||post), nats per ISI
        'divergence per s'  KL(post||pre) x post rate + KL(pre||post) x pre rate, nats per second
        'JS divergence'     Jensen-Shannon, 1/2 E_post[log p_post - log m] + 1/2 E_pre[log p_pre - log m]
                            with m = (p_pre + p_post) / 2; nats per ISI, bounded by ln 2
    For pairs = True the conditional densities p(ISI_n | ISI_{n-1}) are used (references of shape (n, 2)).
    Works for any two fitted distributions, e.g. the post distributions of two step heights.
    sample_ms: time per sample [ms] for 'divergence per s' (e.g. the PSTH bin width). None = the
               mean ISI of each reference, i.e. the samples are ISIs.
    All values are NaN if either distribution is missing.
    '''
    if dist_pre is None or dist_post is None or len(pre_ref) == 0 or len(post_ref) == 0:
        return dict.fromkeys(DIVERGENCE_KEYS, np.nan)
    if pairs:
        lp = lambda dist, ref: dist.logpdf_cond(ref[:, 0], ref[:, 1])
        isi_pre, isi_post = pre_ref[:, 1], post_ref[:, 1]
    else:
        lp = lambda dist, ref: dist.logpdf(ref)
        isi_pre, isi_post = pre_ref, post_ref

    lpre_on_pre, lpost_on_pre = lp(dist_pre, pre_ref), lp(dist_post, pre_ref)
    lpre_on_post, lpost_on_post = lp(dist_pre, post_ref), lp(dist_post, post_ref)

    kl_post_pre = np.mean(lpost_on_post - lpre_on_post)
    kl_pre_post = np.mean(lpre_on_pre - lpost_on_pre)

    # log of the mixture m = (p_pre + p_post) / 2 on each reference
    lm_pre = np.logaddexp(lpre_on_pre, lpost_on_pre) - np.log(2)
    lm_post = np.logaddexp(lpre_on_post, lpost_on_post) - np.log(2)
    JS = 0.5 * np.mean(lpost_on_post - lm_post) + 0.5 * np.mean(lpre_on_pre - lm_pre)

    ms_pre, ms_post = (isi_pre.mean(), isi_post.mean()) if sample_ms is None else (sample_ms, sample_ms)
    return {'KL post||pre': kl_post_pre,
            'KL pre||post': kl_pre_post,
            'divergence': kl_post_pre + kl_pre_post,
            'divergence per s': (kl_post_pre / ms_post + kl_pre_post / ms_pre) * 1000.,
            'JS divergence': JS}


# ---------------------------------------------------------------------------
# PSTH lead-in: population spike counts in place of ISIs (used by psth_cusum_analysis)
# ---------------------------------------------------------------------------

def psth_resamples(trials, k, n_resamples, bin_ms, T, t_discard, rng):
    '''
    Population PSTHs of k randomly chosen trials (without replacement), n_resamples times.
    Bins run from t_discard to T, so a change point on a bin edge falls between bins.
    IN
        trials: list of 1D spike time arrays [ms], one per repeat
        rng:    np.random.Generator
    OUT
        X:      (n_resamples, n_bins) summed spike counts per bin
        t_bin:  end time of each bin [ms]
    '''
    edges = np.arange(t_discard, T + bin_ms/2, bin_ms)
    counts = np.array([np.histogram(np.asarray(s, dtype = float), bins = edges)[0] for s in trials])
    X = np.array([counts[rng.choice(len(trials), k, replace = False)].sum(0) for _ in range(n_resamples)])
    return X, edges[1:]


def reference_bins(X, t_bin, bin_ms, t_cp):
    '''
    Pools the bin values of all resamples into pre- (bin ends by t_cp) and post-changepoint
    (bin starts at/after t_cp) references, as reference_isis does for ISIs.
    OUT
        pre_ref, post_ref:  sorted 1D arrays
    '''
    return np.sort(X[:, t_bin <= t_cp].ravel()), np.sort(X[:, t_bin - bin_ms >= t_cp].ravel())


# ---------------------------------------------------------------------------
# Library detectors on a PSTH: changepoynt (SST, ESST, uLSIF, RuLSIF) and e-divisive (ruptures)
# ---------------------------------------------------------------------------

import ruptures as rpt
from changepoynt.algorithms.sst import SST
from changepoynt.algorithms.esst import ESST
from changepoynt.algorithms.ulsif import ULSIF
from changepoynt.algorithms.rulsif import RuLSIF


class CostEnergy(rpt.base.BaseCost):
    '''
    Kernel change point cost with the distance kernel k(x, y) = |x| + |y| - |x - y|, which makes
    kernel CPD the e-divisive (energy statistic, alpha = 1) method. For a segment it reduces to
    sum_{i,j in seg} |x_i - x_j| / n_seg.
    '''
    model = 'energy'
    min_size = 2

    def fit(self, signal):
        self.signal = x = np.asarray(signal, dtype = float).ravel()
        self.D = np.abs(x[:, None] - x[None, :])
        return self

    def error(self, start, end):
        return self.D[start:end, start:end].sum() / (end - start)


# name -> detector factory; None = e-divisive via ruptures. The CP is the raw argmax of the score, uncorrected
# (on a clean step SST peaks ~8 bins early, ESST ~3 early; uLSIF one window either side; RuLSIF centred).
LIBRARY_METHODS = {
    'sst':    lambda: SST(20),
    'esst':   lambda: ESST(20, threads = 1),     # threads = 1: numba is limited to 1 thread inside joblib workers
    'ulsif':  lambda: ULSIF(window_length = 10, n_windows = 20),
    'rulsif': lambda: RuLSIF(window_length = 10, n_windows = 20),
    'edivisive': None,
}


def library_cp(x, t_bin, bin_ms, method, return_trace = False):
    '''
    Single change point of a PSTH x (bins ending at t_bin [ms]) with a library detector.
        changepoynt methods: argmax of the score trace (no correction);
                             score = peak value
        'edivisive':         exact best single split (rpt.Dynp) under CostEnergy;
                             score = drop in cost from the split
    CP is the start time of the first post-change bin [ms].
    return_trace: also return the score per bin (changepoynt) or the split cost drop per bin (edivisive)
    '''
    make = LIBRARY_METHODS[method]
    x = np.asarray(x, dtype = float)
    if make is None:
        cost = CostEnergy().fit(x)
        n = len(x)
        bkp = rpt.Dynp(custom_cost = cost, min_size = 2, jump = 1).fit(x).predict(n_bkps = 1)[0]
        total = cost.error(0, n)
        CP, score = t_bin[bkp - 1], total - cost.error(0, bkp) - cost.error(bkp, n)
        if return_trace:
            s = np.full(n, np.nan)
            s[2:n - 1] = [total - cost.error(0, b) - cost.error(b, n) for b in range(2, n - 1)]
            return CP, score, s
        return CP, score
    try:
        s = make().transform(x)
    except ValueError:     # SST's Krylov step breaks down on noiseless, rank-deficient PSTHs (var = 0)
        nan = np.full(len(x), np.nan)
        return (np.nan, np.nan, nan) if return_trace else (np.nan, np.nan)
    i = np.argmax(s)
    CP, score = t_bin[i] - bin_ms, s[i]
    return (CP, score, s) if return_trace else (CP, score)


# ---------------------------------------------------------------------------
# Reference-free, non-parametric single-train detectors on the ISI series:
# e-divisive (energy cost) and Non-Parametric Isolate-Detect (Anastasiou & Fryzlewicz 2025)
# ---------------------------------------------------------------------------

def edivisive_split(x):
    '''
    Best single split of x under the e-divisive energy cost (same cost as CostEnergy:
    cost(seg) = sum_{i,j in seg} |x_i - x_j| / n_seg), for all splits at once via a 2D prefix
    sum of |x_i - x_j|: O(n^2) instead of O(n^3) for rpt.Dynp.
    OUT
        bkp:    first index of the second segment (segments x[:bkp], x[bkp:]); None if n < 4
        drop:   cost(0, n) - cost(0, b) - cost(b, n) for every split b (NaN where a segment has < 2 points)
    '''
    x = np.asarray(x, dtype = float)
    n = len(x)
    drop = np.full(n, np.nan)
    if n < 4:
        return None, drop
    P = np.zeros((n + 1, n + 1))
    P[1:, 1:] = np.abs(x[:, None] - x[None, :]).cumsum(0).cumsum(1)
    b = np.arange(2, n - 1)
    drop[b] = P[n, n] / n - P[b, b] / b - (P[n, n] - 2 * P[b, n] + P[b, b]) / (n - b)   # P symmetric
    return int(b[np.argmax(drop[b])]), drop


def isi_time_series(isi, t_isi, dt_grid):
    '''
    ISI step function on a regular grid from the first to the last spike: at each grid time the
    value is the length of the ISI containing it (so long ISIs are repeated, weighted by duration).
    OUT
        x, t_grid
    '''
    t_grid = np.arange(t_isi[0] - isi[0] + dt_grid, t_isi[-1], dt_grid)
    return isi[np.searchsorted(t_isi, t_grid)], t_grid


def edivisive_isi_cp(isi, t_isi, mode = 'isi', dt_grid = 10., return_trace = False):
    '''
    e-divisive single change point of one spike train's ISI series.
        mode = 'isi':   one point per ISI (x_i = ISI_i)
        mode = 'time':  ISI step function sampled every dt_grid ms (isi_time_series)
    OUT
        CP:     time of the last pre-change point [ms] (end of that ISI / that grid time), as in llr_cp
        score:  largest cost drop / n (comparable across trials of different length) [ms]
        if return_trace, also t (time of each point) and drop / n
    '''
    x, t = (isi, t_isi) if mode == 'isi' else isi_time_series(isi, t_isi, dt_grid)
    bkp, drop = edivisive_split(x)
    if bkp is None:
        return (np.nan, np.nan, t, drop) if return_trace else (np.nan, np.nan)
    drop = drop / len(x)
    CP, score = t[bkp - 1], drop[bkp]
    return (CP, score, t, drop) if return_trace else (CP, score)


def _npid_cusum(C, sp, ep):
    '''
    Non-parametric CUSUM on [sp, ep] (1-based, inclusive): max over thresholds u of
    |sqrt(beta/(l alpha)) sum_pre B(u) - sqrt(alpha/(l beta)) sum_post B(u)| for each b in [sp, ep - 1].
    C: cumulative sums of the binary sequences with a leading zero row. Returns (argmax b, max value).
    '''
    b = np.arange(sp, ep)
    a, be, l = b - sp + 1, ep - b, ep - sp + 1
    v = np.abs(np.sqrt(be / (l * a))[:, None] * (C[b] - C[sp - 1])
               - np.sqrt(a / (l * be))[:, None] * (C[ep] - C[b])).max(1)
    i = np.argmax(v)
    return b[i], v[i]


def _npid_rec(C, s, e, thr, r_all, l_all, k_l, k_r, out):
    '''Literal port of L_inf_norm in the authors' main.R (1-based indices, same k_l / k_r carry-over).'''
    if e - s <= 1:
        return
    right = [r for r in r_all if s < r < e] + [e]           # IDetect::s_e_points
    left = [q for q in l_all if s < q < e] + [s]
    rur, lur = len(right), len(left)
    chp = 0
    if k_r < k_l:
        while chp == 0 and k_r < min(k_l, rur):
            pos, c = _npid_cusum(C, s, right[k_r - 1])
            if c > thr: chp, sp, ep = pos, s, right[k_r - 1]
            else: k_r += 1
    if k_l < k_r:
        while chp == 0 and k_l < min(k_r, lur):
            pos, c = _npid_cusum(C, left[k_l - 1], e)
            if c > thr: chp, sp, ep = pos, left[k_l - 1], e
            else: k_l += 1
    while chp == 0 and k_l <= lur and k_r <= rur:
        pos, c = _npid_cusum(C, s, right[k_r - 1])
        if c > thr:
            chp, sp, ep = pos, s, right[k_r - 1]
            break
        pos, c = _npid_cusum(C, left[k_l - 1], e)
        if c > thr:
            chp, sp, ep = pos, left[k_l - 1], e
            break
        k_r += 1
        k_l += 1
    if chp == 0:
        return
    out.append((chp, c, sp, ep))
    if chp > (e + s) / 2:
        _npid_rec(C, s, chp, thr, r_all, l_all, 1, k_r, out)
    else:
        _npid_rec(C, chp + 1, e, thr, r_all, l_all, max(1, k_l - 1), 1, out)


def npid(x, thr_const = 0.9, points = 15):
    '''
    Non-Parametric Isolate-Detect (Anastasiou & Fryzlewicz 2025, arXiv 2504.21379), thresholded L_inf
    version (L_inf_norm in github.com/Anastasiou-Andreas/NPID, default arguments). Change points are
    detected by the ECDF-based CUSUM (all thresholds u = unique values except the max) on intervals
    expanding by `points` from the right and left ends, until it exceeds thr_const * sqrt(log n).
    OUT (sorted by position, 0-based)
        cps:        index of the last point of each pre-change segment
        stats:      CUSUM value at detection
        intervals:  (start, end) inclusive of the interval that isolated each change point
    '''
    x = np.asarray(x, dtype = float)
    n = len(x)
    u = np.unique(x)[:-1]
    if n < 3 or len(u) == 0:
        return np.array([], int), np.array([]), np.zeros((0, 2), int)
    C = np.zeros((n + 1, len(u)))
    C[1:] = np.cumsum(x[:, None] > u[None, :], axis = 0)
    out = []
    _npid_rec(C, 1, n, thr_const * np.sqrt(np.log(n)), list(range(points, n + 1, points)),
              list(range(n - points + 1, 0, -points)), 1, 1, out)
    out.sort()
    cps = np.array([o[0] - 1 for o in out], int)
    return cps, np.array([o[1] for o in out]), np.array([(o[2] - 1, o[3] - 1) for o in out], int).reshape(-1, 2)


def npid_cp(isi, t_isi, thr_const = 0.9, points = 3, return_trace = False):
    '''
    Single change point of one spike train's ISI series (one point per ISI) with npid.
    points = 3 (not the R default 15) because a trial has only a few tens of ISIs.
    OUT
        CP:     end time of the last pre-change ISI of the strongest detection [ms]; NaN if none
        score:  its CUSUM / sqrt(log n), i.e. the threshold constant at which it would just be missed; 0 if none
        if return_trace, also the npid output (cps, stats, intervals) for all detections
    '''
    if len(isi) < 3:
        return (np.nan, np.nan, npid([])) if return_trace else (np.nan, np.nan)
    det = npid(isi, thr_const, points)
    cps, stats_ = det[0], det[1]
    if len(cps) == 0:
        CP, score = np.nan, 0.
    else:
        i = np.argmax(stats_)
        CP, score = t_isi[cps[i]], stats_[i] / np.sqrt(np.log(len(isi)))
    return (CP, score, det) if return_trace else (CP, score)


# ---------------------------------------------------------------------------
# Online CUSUM detection: latency vs false alarms, and the divergence-based bounds
# ---------------------------------------------------------------------------

def llr_trace(isi, t_isi, dist_pre, dist_post, pairs = False, clip = None):
    '''
    Per-ISI log-likelihood ratio, as in llr_cp (pairs = False) or llr_pair_cp (pairs = True),
    without the changepoint search.
    OUT
        t:  end time of each scored ISI [ms] (the first ISI is not scored for pairs)
        l:  log p_post - log p_pre, clipped to +-clip (None = unclipped)
    '''
    if pairs:
        prev, curr, t = isi_pairs(isi, t_isi)
        l = dist_post.logpdf_cond(prev, curr) - dist_pre.logpdf_cond(prev, curr)
    else:
        t, l = t_isi, dist_post.logpdf(isi) - dist_pre.logpdf(isi)
    if clip is not None:
        l = np.clip(l, -clip, clip)
    return t, l


def online_cusum(l, b):
    '''
    Online CUSUM W_n = max(0, W_{n-1} + l_n), alarm when W_n > b, run for every threshold in b at once.
    W restarts at 0 after each alarm (so the alarm count measures the false-alarm rate on pre-change data);
    the first alarm is the same with or without restarting.
    IN
        l:  LLR of each ISI, in arrival order
        b:  thresholds [nats]
    OUT
        first:      index into l of the first alarm for each b (-1 = no alarm)
        n_alarms:   number of alarms for each b
    '''
    b = np.atleast_1d(np.asarray(b, dtype = float))
    W = np.zeros_like(b)
    first = np.full(len(b), -1)
    n_alarms = np.zeros(len(b), dtype = int)
    for n, ln in enumerate(l):
        W = np.maximum(0., W + ln)
        hit = W > b
        first[hit & (first < 0)] = n
        n_alarms += hit
        W[hit] = 0.
    return first, n_alarms


def cusum_theory(kl_post_pre, kl_pre_post, b):
    '''
    Siegmund's approximations for the CUSUM with threshold b (iid observations, overshoot ignored):
        ADD  ~ (b + e^-b - 1) / KL(post || pre)     mean delay from W = 0, in ISIs
        ARL0 ~ (e^b - b - 1) / KL(pre || post)      mean run length to a false alarm, in ISIs
    '''
    b = np.asarray(b, dtype = float)
    return (b + np.exp(-b) - 1) / kl_post_pre, (np.exp(b) - b - 1) / kl_pre_post


def lai_bound(kl_post_pre, arl0_isi):
    '''
    Lai (1998) asymptotic lower bound on the delay of any online detector with mean run length
    arl0_isi to a false alarm: ln(ARL0) / KL(post || pre), in ISIs.
    '''
    return np.log(np.maximum(arl0_isi, 1.)) / kl_post_pre


def residual_isi(isi):
    '''
    Mean wait from a random time (the step) to the next spike, E[X^2] / (2 E[X]) for a renewal
    process (inspection paradox); same units as isi.
    '''
    isi = np.asarray(isi, dtype = float)
    return np.mean(isi**2) / (2 * np.mean(isi))


def optimal_delay(arl0_isi, kl_post_pre, kl_pre_post, b_max = 50.):
    '''
    Mean CUSUM delay [ISIs] at a given mean run length to a false alarm [ISIs], from cusum_theory with
    b eliminated: the delay / false-alarm trade-off of the optimal (Lorden) detector for these divergences.
    Matching on the false-alarm rate, rather than on b, cancels most of the overshoot error of the
    Siegmund approximations. Broadcasts over arl0_isi and the KLs.
    '''
    # ARL0 x KL(pre||post) = e^b - b - 1 does not depend on the KLs: invert it for b, then ADD = (b + e^-b - 1) / KL
    b = np.linspace(0., b_max, 5001)[1:]
    b_at = np.interp(np.log(np.asarray(arl0_isi) * kl_pre_post), np.log(np.exp(b) - b - 1), b,
                     left = np.nan, right = np.nan)
    return (b_at + np.exp(-b_at) - 1) / kl_post_pre


def delay_to_ms(add_isi, mean_isi_post, residual):
    '''
    Converts a CUSUM delay in observations to time after the step [ms]. The first observation is the
    ISI straddling the step, which ends on average `residual` after it (residual_isi); each further
    observation adds a mean post-step ISI. Never below the residual.
    '''
    return residual + np.maximum(np.asarray(add_isi) - 1, 0) * mean_isi_post
