# Change point analysis with AQUA

Does an autapse help a neuron signal a sudden change in its input?

Single regular-spiking (RS, Izhikevich-type) AQUA neurons, with and without an autapse, receive a noisy step current. The FLUSS algorithm is then run on each neuron's membrane potential to estimate when the step happened, and how confidently. Results are compared across autapse strength, step height and noise level.

- Aims and protocol: [PROTOCOL.md](PROTOCOL.md)
- Results so far: [FINDINGS.txt](FINDINGS.txt)

## Main script: `CPA_with_AQUA.py`

Runs the full parameter sweep and saves one results table.

### 1. Parameters (`main()`)

| Parameter | Value |
|---|---|
| Neuron model | `RS` dict (C = 100, k = 0.7, v_r = -60, v_t = -40, ...) |
| Autapse strength `f` | `[0, 100, 150, 200, 220, 240, 260, 280, 300, 350, 400]` |
| Autapse `e`, `tau` | 0.2 /ms, 2 ms (fixed) |
| Step height `d_H` (pA) | `[0, 1, 2, 3, 4, 5, 10, 12, 15, 18, 20, 25, 30, 40, 50]` |
| Noise variance `var` | `np.linspace(0, 50, 5)` |
| FLUSS window | 300 time steps (30 ms) |
| Repeats `N_repeats` | 100 |
| Output file | `CPA_data_N100.pickle` |

Change these arrays in `main()` to run a different sweep.

### 2. Stimulus (`simulate_w_fluss()`)

- Duration T = 3000 ms, time step dt = 0.1 ms.
- The current starts at I_0 = 60 pA and steps to I_0 + h at **1500 ms** (built with `step_current`).
- Noise is piecewise-constant Gaussian: one sample is drawn every 20 ms (50 Hz) and held, with the chosen variance.
- Within each (variance, step height) block, every autapse strength `f` gets the **same** noisy current, so neurons are compared on identical input.

### 3. Simulation

For each repeat, every combination (variance × step height × `f`) is simulated in one `batchAQUA` batch, starting from state `[-60, 0, 0]`. Neurons are ordered with variance outermost, then step height, then `f` (use `find_idx()` to look one up).

### 4. Change point detection (`process_single_row()` in `CP_functions.py`)

Each membrane potential trace is analysed in parallel with `joblib`:

1. `stumpy.stump` computes the matrix profile of V(t) with the FLUSS window.
2. `stumpy.fluss` (one regime) returns the corrected arc curve.
3. The score is `1 - arc curve`. Its peak value is the **CAC** (a measure of confidence) and the time of the peak (`argmax × dt`, in ms) is the **estimated change point**.

### 5. Output

A pandas DataFrame, pickled to the output file, with one row per simulated neuron per repeat:

| Column | Meaning |
|---|---|
| `N` | repeat index |
| `variance` | noise variance |
| `step_height` | step size h (pA) |
| `autapse f` | autapse strength |
| `FLUSS window` | window length (time steps) |
| `CAC` | peak FLUSS score |
| `change point` | estimated change time (ms); true value is 1500 |

### Running

```bash
python CPA_with_AQUA.py
```

Requires `aqua`, `stumpy`, `changepoynt`, `joblib`, `pandas`, `seaborn` and `tqdm` (all installed in the `aqua` conda env). The full sweep takes a long time; `nohup.out` is the log of an earlier run.

## Helper functions: `CP_functions.py`

Imported into `CPA_with_AQUA.py` with `from CP_functions import *`. The notebooks get them through `from CPA_with_AQUA import *`.

- `find_idx(f, v, h, f_vals, v_vals, h_vals)` returns the row index of the neuron with autapse strength `f`, variance `v` and step height `h` in a batch built by `simulate_w_fluss()`. Used by `CPA_TRACES_with_AQUA.ipynb` to pull out single traces.
- `process_single_row(row_idx, data_matrix, window_size, dt)` runs FLUSS on one voltage trace and returns `(CAC, change point)` (step 4 above).
- `psychometric_curve(data, x, y, hue='autapse f', ...)` plots mean ± SD of a metric (e.g. CAC) against step height, one line per value of `hue`, with a colourbar. Used by `AQUA_CPA_plots.ipynb`.
- `changepoint_distribution(data, x='change point', row='autapse f', col='variance', ...)` draws a grid of histograms of the estimated change point times (rows = autapse strength, columns = noise variance), coloured by autapse strength, with a dashed line at the true change (1500 ms). Accepts optional `fig`/`ax` and returns `(fig, ax)`. Used by `AQUA_CPA_plots.ipynb`.

## Other files

| File | Description |
|---|---|
| `changepoint.py` | `CUSUM_mean()`: a simple offline CUSUM detector for a single change in mean, with a variance-normalised threshold. |
| `changepoint_testing.ipynb` | Tests CUSUM, SST and ESST on a synthetic noisy step signal. |
| `changepoint_neurons.ipynb` | Single-neuron exploration: autaptic vs non-autaptic responses to a step, trying ESST, FLUSS and RuLSIF on the voltage trace. |
| `CPA_TRACES_with_AQUA.ipynb` | Re-simulates selected (f, variance, step height) conditions and plots their traces with FLUSS scores, to explain unusual results. |
| `AQUA_CPA_plots.ipynb` | Loads a results pickle and plots psychometric curves, area under the curve vs `f`, and change point error distributions. Saves figures to `plots/`. |
| `test.ipynb` | Scratch checks (pickle reshaping, biexponential autapse conversion). |
| `PROTOCOL.md` | Aims and approach (offline vs online detection, follow-up ideas). |
| `FINDINGS.txt` | Notes on the results from each data set. |

## Data and figures

| File | Contents |
|---|---|
| `CPA_data_N100.pickle` | Main results: 100 repeats, noise resampled at 50 Hz. |
| `CPA_data_N10.pickle` | Earlier run: 10 repeats, noise resampled at 10 kHz. |
| `CPA_test_data.pickle`, `CPA_test_.pickle`, `results_test.pickle` | Small test runs. |
| `plots/` | Figures for the N10 and N100 data sets (CAC, change point and AUC plots). |
| `nohup.out` | Log of a background run. |
