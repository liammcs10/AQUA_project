# Change point analysis with AQUA

Does an autapse help a neuron signal a sudden change in its input?

Single regular-spiking (RS, Izhikevich-type) AQUA neurons, with and without an autapse, receive a noisy step current. A change point detector is then run on each neuron's output to estimate when the step happened, and how confidently. Results are compared across autapse strength, step height and noise level.

There are two pipelines:

1. **FLUSS** (offline, model-free) on the membrane potential V(t).
2. **CUSUM** on the spike times only, using the inter-spike interval (ISI) distribution. Three variants: a windowed KS test, a log-likelihood-ratio (LLR) CUSUM on fitted ISI distributions, and an ISI-pair (Markov) LLR CUSUM that captures bursting.

- Aims and protocol: [PROTOCOL.md](PROTOCOL.md)
- Results so far: [FINDINGS.txt](FINDINGS.txt)

## Main script: `CPA_with_AQUA.py`

Runs a full parameter sweep with one of the two pipelines and saves one results table. `main()` currently runs the CUSUM pipeline; the FLUSS call is commented out above it.

### 1. Parameters (`main()`)

Shared by both pipelines:

| Parameter | Value |
|---|---|
| Neuron model | `RS` dict (C = 100, k = 0.7, v_r = -60, v_t = -40, ...) |
| Autapse strength `f` | `[0, 100, 150, 200, 220, 240, 260, 280, 300, 350, 400]` |
| Autapse `e`, `tau` | 0.2 /ms, 2 ms (fixed) |
| Step height `d_H` (pA) | `[0, 1, 2, 3, 4, 5, 10, 12, 15, 18, 20, 25, 30, 40, 50]` |
| Noise variance `var` | `np.linspace(0, 50, 5)` |
| Repeats `N_repeats` | 100 |

Pipeline-specific:

| Parameter | FLUSS | CUSUM |
|---|---|---|
| Window | `window = 300` time steps (30 ms) | `n_isi = 5` ISIs (KS only) |
| Method | - | `method = 'ks'`, `'llr'` or `'llr_pair'` |
| Output file | `CPA_data_N100.pickle` | `CPA_cusum_N{N_repeats}_{method}.pickle` |

Change these in `main()` to run a different sweep.

### 2. Stimulus

| | `simulate_w_fluss()` | `simulate_w_cusum()` |
|---|---|---|
| Duration T | 3000 ms | 10000 ms |
| True change point | 1500 ms | 5000 ms (T / 2) |
| Time step dt | 0.1 ms | 0.1 ms |

- The current starts at I_0 = 60 pA and steps to I_0 + h at the change point (built with `step_current`).
- Noise is piecewise-constant Gaussian: one sample is drawn every 20 ms (50 Hz) and held, with the chosen variance.
- Within each (variance, step height) block, every autapse strength `f` gets the **same** noisy current, so neurons are compared on identical input.

### 3. Simulation

Both pipelines get their data from `simulate_runs()`, which builds the stimulus above and runs `batchAQUA` from state `[-60, 0, 0]`. It is a generator: it yields `(reps, out)` one batch of repeats at a time, so the full V traces are never all held in memory at once. Within a repeat, neurons are ordered with variance outermost, then step height, then `f` (use `find_idx()` to look one up).

- FLUSS: one repeat per batch, full state saved (`out = (X, t, spikes)`, V is needed).
- CUSUM: `repeats_per_batch` (default 5) repeats per batch, with `save_all = False` so only spike times are kept (`out` = spike time array). This is what makes the longer, larger runs possible.

`simulate_w_cusum(..., spike_file = ...)` skips the simulation and re-analyses a saved `_spikes.pickle` instead. `f_values`, `d_H`, `var`, `N_repeats` and the change point are then read from the file. Set `spike_file` in `main()` to try another `method` on an existing sweep.

### 4a. FLUSS detection (`process_single_row()`)

Each membrane potential trace is analysed in parallel with `joblib`:

1. `stumpy.stump` computes the matrix profile of V(t) with the FLUSS window.
2. `stumpy.fluss` (one regime) returns the corrected arc curve.
3. The score is `1 - arc curve`. Its peak value is the **CAC** (a measure of confidence) and the time of the peak (`argmax × dt`, in ms) is the **estimated change point**.

### 4b. CUSUM detection (`cusum_analysis()`)

One parallel job per condition (variance, step height, `f`), in `_cusum_condition()`:

1. Spike trains become ISIs (`spike_isis`); spikes before `t_discard` = 200 ms (initial transient) are dropped.
2. ISIs from **all repeats** of the condition are pooled into a pre- and a post-change reference, using the known change point (`reference_isis`). ISIs straddling it are dropped.
3. Each trial is then scored against those references, depending on `method`:
   - **`'ks'`** (`ks_window_cp`): a window of `n_isi` ISIs slides along the train; for each window d = KS(window, pre) − KS(window, post).
   - **`'llr'`** (`fit_isi_distribution`, `llr_cp`): gamma, lognormal, inverse Gaussian and 2- / 3-component Gaussian mixtures on log(ISI) (`LogGMM`) are fitted to each reference and the best is chosen by BIC. Every ISI is scored with l = log p_post − log p_pre, clipped to ±10.
   - **`'llr_pair'`** (`reference_pairs`, `fit_isi_pair_distribution`, `llr_pair_cp`): as `'llr'`, but on consecutive ISI pairs. 2D Gaussian mixtures (K = 1–3, BIC) on (log ISI_{n−1}, log ISI_n) (`LogPairGMM`) give p(ISI_n | ISI_{n−1}), so burst structure counts as evidence.
4. The **estimated change point** is the argmin of the cumulative sum of d or l (the best single step fit). The **score** is mean(after) − mean(before), about 0 when there is no change.
5. For the LLR methods only:
   - the **score per s** (`evidence_rate`)
   - divergences between the fitted pre and post distributions, estimated on the reference data (`distribution_divergences`): KL in both directions, Jeffreys (their sum, also per second) and Jensen–Shannon
   - the **Fisher information** about the input current, in two ways. *Local*: J / h², from each condition's pre/post pair; only accurate for small h, and NaN at h = 0. *Across step heights*: `_fisher_across_steps` uses central differences between the post distributions of neighbouring step heights, I(I_0 + h_i) ≈ J(post_{i−1}, post_{i+1}) / (h_{i+1} − h_{i−1})², one-sided at the ends.

### 5. Output

A pandas DataFrame, pickled to the output file, with one row per simulated neuron per repeat.

FLUSS (`simulate_w_fluss`):

| Column | Meaning |
|---|---|
| `N` | repeat index |
| `variance` | noise variance |
| `step_height` | step size h (pA) |
| `autapse f` | autapse strength |
| `FLUSS window` | window length (time steps) |
| `CAC` | peak FLUSS score |
| `change point` | estimated change time (ms); true value is 1500 |

CUSUM (`simulate_w_cusum`) has the same first four columns, then:

| Column | Meaning |
|---|---|
| `ISI window` | `n_isi` for `'ks'`, NaN for the LLR methods |
| `score` | mean(d or l after CP) − mean(before CP) |
| `change point` | estimated change time (ms); true value is 5000 |
| `method` | `'ks'`, `'llr'` or `'llr_pair'` |
| `pre dist`, `post dist` | chosen family for each reference (LLR only) |
| `score per s` | evidence rate in this trial, nats/s (LLR only) |
| `divergence` | Jeffreys divergence of the fitted pre/post distributions, nats per ISI (LLR only) |
| `divergence per s` | the same weighted by the pre/post firing rates, nats/s (LLR only) |
| `KL post\|\|pre`, `KL pre\|\|post` | the two KL divergences that make up `divergence`, nats per ISI (LLR only) |
| `JS divergence` | Jensen–Shannon divergence of pre/post, nats per ISI, at most ln 2 (LLR only) |
| `fisher local`, `fisher local per s` | Fisher information about the input current, `divergence` / h², nats/pA² per ISI and per s; NaN at h = 0 (LLR only) |
| `fisher`, `fisher per s` | Fisher information at I_0 + h from neighbouring step heights' post distributions, nats/pA² per ISI and per s (LLR only) |
| `KS distance` | two-sample KS statistic between the pooled pre and post ISIs, 0–1 (all methods, model-free) |
| `energy distance`, `energy distance (log)` | energy distance between the pooled pre and post ISIs, in ms and on log(ISI) (all methods, model-free) |

The CUSUM run also writes `<outfile>_spikes.pickle`: a dict with `spikes[(repeat, v, h, f)]` (spike times) plus `f_values`, `d_H`, `var`, `N_repeats`, `changepoint`, `dt` and `T`.

### Running

```bash
python CPA_with_AQUA.py
```

Requires `aqua`, `stumpy`, `changepoynt`, `joblib`, `pandas`, `scipy`, `scikit-learn`, `seaborn` and `tqdm` (all installed in the `aqua` conda env; `aqua` also imports `brian2`). The full sweep takes a long time; `nohup.out` is the log of the last background run.

## Helper functions: `CP_functions.py`

Imported into `CPA_with_AQUA.py` with `from CP_functions import *`. The notebooks get them through `from CPA_with_AQUA import *`.

**Indexing and FLUSS**

- `find_idx(f, v, h, f_vals, v_vals, h_vals)` returns the row index of the neuron with autapse strength `f`, variance `v` and step height `h` in a batch. Used by `CPA_TRACES_with_AQUA.ipynb` to pull out single traces.
- `process_single_row(row_idx, data_matrix, window_size, dt)` runs FLUSS on one voltage trace and returns `(CAC, change point)` (step 4a).

**ISI CUSUM** (step 4b)

- `spike_isis`, `reference_isis`: spike times → ISIs, and pooled pre/post references.
- `ks_stat_windows`, `ks_window_cp`: vectorised two-sample KS statistic and the windowed KS detector.
- `LogGMM`, `fit_isi_distribution`, `llr_cp`: 1D distribution fits (BIC choice) and the LLR detector.
- `isi_pairs`, `reference_pairs`, `LogPairGMM`, `fit_isi_pair_distribution`, `llr_pair_cp`: the ISI-pair (Markov) versions.
- `evidence_rate`, `distribution_divergences`: score per second, and KL / Jeffreys / Jensen–Shannon divergences between any two fitted distributions (nats).

All detectors take `return_trace = True` to also return the per-window/per-ISI values and the CUSUM curve, for plotting.

**Metrics and plots**

- `psychometric_curve(data, x, y, hue='autapse f', ...)` plots mean ± SD of a metric (e.g. CAC or score) against step height, one line per value of `hue`, with a colourbar.
- `changepoint_distribution(data, x='change point', row='autapse f', col='variance', ...)` draws a grid of histograms of the estimated change point times (rows = autapse strength, columns = noise variance), with an optional dashed line at `true_cp`. Returns `(fig, ax)`.
- `curve_auc(data, y='score', x='step_height', ...)` returns the (optionally normalised) area under the `y` vs step height curve, one row per neuron / variance / trial.
- `auc_plot(auc_df, x='autapse f', y='AUC', hue='variance', ...)` plots that area against `f`, one line per noise level.
- `weighted_cp_distance(data, true_cp, ...)` gives inverse-distance-weighted change point accuracy per condition (`weighted CP`, `weighted CP dist`, `weighted |error|`). Alternative accuracy measures to try later are listed in `alternative_distances`.

## Other files

| File | Description |
|---|---|
| `CUSUM_method_demo.ipynb` | Worked example of the KS, LLR and ISI-pair LLR methods on one condition: references, fitted distributions, return maps and detection traces. Start here for the CUSUM pipeline. |
| `AQUA_CPA_plots_CUSUM.ipynb` | Loads a CUSUM results pickle and plots psychometric curves, AUC of score / score per s / divergence vs `f` and variance, change point errors, weighted accuracy and change point histograms. |
| `CPA_divergences_CUSUM.ipynb` | Plots the divergence measures (KL, Jeffreys, JS, local and across-step Fisher information) against step height, per noise level and per neuron. Also: their mean vs `f`, JS / Fisher heatmaps, and checks (local vs across-step Fisher, evidence collected vs Jeffreys per s). Ends with the model-free KS / energy distances and how they compare with the fitted measures. Uses one row per condition; drops var = 0 by default. |
| `AQUA_CPA_plots_FLUSS.ipynb` | The same for FLUSS results (CAC psychometric curves, AUC vs `f`, change point error distributions). Saves figures to `plots/`. |
| `CPA_TRACES_with_AQUA.ipynb` | Re-simulates selected (f, variance, step height) conditions and plots their traces with FLUSS scores, to explain unusual results. |
| `changepoint_neurons.ipynb` | Single-neuron exploration: autaptic vs non-autaptic responses to a step, trying ESST, FLUSS and RuLSIF on the voltage trace. |
| `changepoint_testing.ipynb` | Tests CUSUM, SST and ESST on a synthetic noisy step signal. |
| `changepoint.py` | `CUSUM_mean()`: a simple offline CUSUM detector for a single change in mean, with a variance-normalised threshold. |
| `decodability.ipynb` | Stub: planned decodability metrics (Fisher information, KL divergence, ROC). |
| `generate_llr_results.ipynb` | Empty placeholder. |
| `test.ipynb` | Scratch checks (pickle reshaping, biexponential autapse conversion). |
| `alternative_distances` | Notes on change point accuracy measures not yet implemented. |
| `PROTOCOL.md` | Aims and approach (offline vs online detection, follow-up ideas). |
| `FINDINGS.txt` | Notes on the results from each data set. |

## Data and figures

| File | Contents |
|---|---|
| `CPA_data_N100.pickle` | FLUSS results: 100 repeats, noise resampled at 50 Hz. |
| `CPA_data_N10.pickle` | Earlier FLUSS run: 10 repeats, noise resampled at 10 kHz. |
| `CPA_cusum_N100_spikes.pickle` | Raw spike times for the 100-repeat CUSUM sweep (T = 10 s, CP at 5 s). |
| `CPA_cusum_N100_ks.pickle` | Windowed KS CUSUM (`n_isi` = 5). |
| `CPA_cusum_N100_llr.pickle` | LLR CUSUM, families gamma / lognormal / inverse Gaussian only. |
| `CPA_cusum_N100_llr_gmm.pickle` | LLR CUSUM, plus 2-component log-ISI mixture. |
| `CPA_cusum_N100_llr_gmm3.pickle` | LLR CUSUM, plus 2- and 3-component mixtures (current default), with score per s and divergence columns. |
| `CPA_cusum_N100_llr_pair.pickle` | ISI-pair LLR CUSUM, with score per s and divergence columns. |
| `CPA_cusum_N100_llr_pair_div.pickle` | The same re-analysed from the spike file, plus the KL, JS and Fisher information columns. |
| `CPA_test_data.pickle`, `CPA_test_.pickle`, `results_test.pickle` | Small test runs. |
| `plots/` | FLUSS figures for the N10 and N100 data sets (CAC, change point and AUC plots). |
| `nohup.out` | Log of the last background run. |
