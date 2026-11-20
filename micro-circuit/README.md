# micro-circuit

Simulations of small excitatory/inhibitory circuits in which one neuron has an autapse.
The question is whether an autapse, by changing a single neuron's behaviour, also changes what the circuit does:
synchrony, entrainment, regularity, and which neuron controls the network's output.

The motivation is that an excitatory autapse can make a neuron more sensitive to input patterns. That could act as a
neuromodulated mechanism for attentional control, sensory gating or winner-takes-all dynamics. These mechanisms
rely on a few building blocks, which these simulations test:

- **Entrainment / coherence**: can one neuron force another to follow its firing rhythm?
- **Competing (shunting) inhibition**: can one neuron get an interneuron to suppress another neuron?
- **Frequency gating**: can an autapse set the frequency the circuit fires at (e.g. gamma nested in theta)?

`project.md` and `PROJECT_PLAN.txt` hold the full plan and running findings.

## Model networks

All neurons are Izhikevich-type AQUA neurons simulated in Brian2. In every network **E1 is the autaptic neuron**
(RS resonator, autapse `e`, `f`, `tau`), **E2 is the same neuron with no autapse**, and **I is an FS interneuron**.
Each network is compared against a control where E1 is replaced by a copy of E2 (no autapse network).

| Network | Question |
|---------|----------|
| E1 ↔ E2 | Can the autaptic neuron entrain the non-autaptic one? |
| E1 → I → E2 | Can E1 drive shunting inhibition of E2 through the interneuron? |
| E1, E2 ↔ I (STDP on I's synapses) | Can the autapse set synchrony, and its frequency, across the whole circuit? STDP helps the network settle into a steady state. |

## Main finding so far

Sweeping the autapse peak `f` in the 3-neuron E/I network (`bursting_analysis.py`):

- **Identical drive** (`I_inj = 200`, no noise): the network starts highly synchronised, then **desynchronises** as E1
  crosses into bursting (`f` > ~200 pA), slightly before the bursting transition itself.
- **Unequal drive** ("test200": E1 = 200, E2 = 150): synchrony **improves sharply**, exactly at the bursting
  transition, and CV_ISI drops for all neurons.
- **OU noise** (`theta = 0.2`, `sigma = 5` pA): results match the identical, noise-free case.

So the network changes state at the autapse's bifurcation point. Other transitions in synchrony and directionality
that are not tied to bursting are investigated in the `test_other_predictors_*` notebooks.

## Simulation scripts

Run from this folder with the `aqua` package installed (`AQUA_package/`). Each saves pickled DataFrames here.

| Script | What it tests | Sweep | Output |
|--------|---------------|-------|--------|
| `synchrony_analysis.py` | Entrainment in the 3-neuron E/I network, autaptic vs non-autaptic E1 | E1 drive (120–280 pA), E1→E2 and E2→E1 weights (20–100) | `synch_analysis_{aut,naut}_test{DIST,ISI_E1,ISI_E2}.pickle`: FT, ISI and SPIKE distances, SPIKE synchrony, directionality, and ISI stats per neuron |
| `shunting_analysis.py` | Shunting inhibition through the interneuron (E1 → I → E2) | E1 drive (120–280 pA), E1→I and I→E2 weights (20–100) | `shunt_analysis_{aut,naut}.pickle`: synchrony and distance metrics |
| `bursting_analysis.py` | Fixes drive and weights and varies **only the autapse** (`f` from 10 to 450 pA), to show the autapse's non-linearity alone drives the change in network behaviour | `f`, under identical, unequal (test200) or OU-noise conditions | `burst_analysis_<tag>.pickle` (synchrony, regularity: CV_ISI, LV, ISI entropy, number of ISI peaks) and `burst_analysis_<tag>TRACES.pickle` (spike trains and voltage traces) |

Helper modules:

| File | Contents |
|------|----------|
| `FT_metrics.py` | Synchrony via spectra: FFT of (smoothed) binary spike trains, spectral distance between two neurons, rolling FT distance, top FFT peaks, ISI-histogram distance |
| `functions.py` | Synapse connectivity plot, heat maps of a metric over two parameters (autaptic, non-autaptic and difference), ISI peak plots |

## Notebooks

**Building the circuits** (exploratory; the scripts above are the sweep versions):

| Notebook | Contents |
|----------|----------|
| `2_excitatory.ipynb` | E1 ↔ E2: can E1 entrain E2, and does the autapse help? |
| `excitatory-inhibitory.ipynb` | 2E + 1I: winner-takes-all vs coherence; can the autaptic neuron set the other neurons' firing rate? |
| `EI_STDP.ipynb` | The same network with STDP on the synapses to and from I |
| `EI_STDP_comparison.ipynb` | Autaptic vs non-autaptic networks reaching synchrony (STDP network) |
| `EI_shunting.ipynb` | The same comparison for the shunting-inhibition network |
| `EI_regulate_autapse.ipynb` | *Why* the autapse changes synchrony: is it the extra current, the waveform or the timing? |
| `EI_regulate_autapse_noise_test.ipynb` | The same with input noise |

**Plotting script outputs:**

| Notebook | Plots |
|----------|-------|
| `plot_burstdata.ipynb` | Metrics against `f` for `bursting_analysis.py` outputs, with background shading by number of ISI peaks; figures in `plots/` |
| `plot_burstdata_test.ipynb` | Quick look at a burst-analysis pickle |
| `plot_synchdataDIST.ipynb` | Heat maps of distance and synchrony metrics from `synchrony_analysis.py` |
| `plot_synchdataISI.ipynb` | Heat maps of ISI histogram summaries from `synchrony_analysis.py` |

**Metrics and follow-ups:**

| Notebook | Contents |
|----------|----------|
| `test_other_predictors_identical.ipynb`, `test_other_predictors_test200.ipynb` | Transitions in synchrony and directionality that are not caused by bursting (e.g. directionality peaks at `f` ~ 40 and ~150 pA) |
| `test_coding_schemes.ipynb` | Compares synchrony measures (e.g. van Rossum distance over filter time constants) on the burst-analysis traces |
| `test_peakFinding.ipynb` | ISI peak finding: `scipy.find_peaks` vs Gaussian mixture vs Bayesian Gaussian mixture |
| `test.ipynb` | Scratch test of spike binarising and filtering |

## Subfolders

Can the job be done by something other than an autapse? These test which features of the autaptic current matter.

| Folder | Contents |
|--------|----------|
| `autapse_waveforms/` | Autaptic currents of different shapes with the same net charge (`autapse_test_all_waveforms.ipynb`), and checks that the Brian2 implementation matches `batchAQUA` (`autapse_meetBrian.ipynb`) |
| `autapse_random_delay/` | The autapse fires after a random delay drawn from a uniform, normal, gamma or Poisson distribution; `test_aqua_different_distribution.ipynb` tests the matching `autapse_mode` options in `batchAQUA` |
| `plots/` | Figures from `plot_burstdata.ipynb` |
| `output/` | Generated Brian2 CUDA standalone code (from the commented-out `brian2cuda` option in the scripts) |
