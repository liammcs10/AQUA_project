import numpy as np
import stumpy
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


def process_single_row(row_idx, data_matrix, window_size, dt):
    '''
    Computes the matrix profile on a single row and returns the CAC
    '''
    row_data = data_matrix[row_idx, 0, :]

    mat_prof = stumpy.stump(row_data, m = window_size)

    arc_crossings, _ = stumpy.fluss(mat_prof[:, 1], L = window_size, n_regimes = 1)

    score = 1 - arc_crossings

    CAC = np.max(score)
    CP = np.argmax(score)*dt

    return (CAC, CP)


def psychometric_curve(data, x, y, hue = 'autapse f', fig = None, ax = None, add_colorbar = True, colorbar_label = 'Autapse Strength'):
    '''
    Plot the psychometric curve for different autapse conditions
    
    '''
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (5, 5))

    # 1. Define your colormap
    cmap_name = "viridis"
    norm = mcolors.Normalize(vmin=data[hue].min(), vmax=data[hue].max())

    #palette = {val: color for val, color in zip([s for s in autapses if s != '0.0'], default_colors)}
    #palette['0.0'] = 'black'  # Your pre-defined color

    # 3. Plot
    sns.lineplot(data=data, x=x, y=y, hue=hue, palette=cmap_name, ax = ax, legend = False, 
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

    cmap_name = "viridis"
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
import pandas as pd
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


def jeffreys_divergence(dist_pre, dist_post, pre_ref, post_ref, pairs = False):
    '''
    Symmetric KL (Jeffreys) divergence between the fitted pre and post distributions, estimated
    on the reference data (unclipped):
        J = KL(post || pre) + KL(pre || post)
    with KL(post || pre) = mean over post_ref of [log p_post - log p_pre], and vice versa.
    For pairs = True the conditional densities p(ISI_n | ISI_{n-1}) are used (references of shape (n, 2)).
    OUT
        J:      nats per ISI
        J_s:    nats per second, KL(post||pre) x post rate + KL(pre||post) x pre rate
    '''
    if dist_pre is None or dist_post is None or len(pre_ref) == 0 or len(post_ref) == 0:
        return np.nan, np.nan
    if pairs:
        lp = lambda dist, ref: dist.logpdf_cond(ref[:, 0], ref[:, 1])
        isi_pre, isi_post = pre_ref[:, 1], post_ref[:, 1]
    else:
        lp = lambda dist, ref: dist.logpdf(ref)
        isi_pre, isi_post = pre_ref, post_ref
    kl_post_pre = np.mean(lp(dist_post, post_ref) - lp(dist_pre, post_ref))
    kl_pre_post = np.mean(lp(dist_pre, pre_ref) - lp(dist_post, pre_ref))
    J = kl_post_pre + kl_pre_post
    J_s = (kl_post_pre / isi_post.mean() + kl_pre_post / isi_pre.mean()) * 1000.
    return J, J_s
