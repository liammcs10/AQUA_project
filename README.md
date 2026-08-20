# AQUA Project

This repository contains implementations, experiments, and analysis notebooks for
the **AQUA** model: an **Adaptive QUadratic integrate-and-fire neuron with an
Autapse**.

The project studies how self-feedback through an autapse changes neuronal
dynamics, including firing patterns, bursting, gain modulation, resonance,
change-point detection, synchrony, and small circuit behaviour.

## Project Structure

The repository is organized into three broad layers.

### 1. Reusable AQUA package

The most reusable implementation lives in `AQUA_package/aqua/`.

- `AQUA_general.py` implements a single-neuron AQUA model using RK2 integration.
- `batchAQUA_general.py` implements vectorized batch simulations and Brian2
  integration.
- `stimulus.py` contains input current generators, including step currents,
  ramps, Ornstein-Uhlenbeck noise, spike trains, sinusoids, and filtered white
  noise.
- `utils.py` contains analysis helpers for spike trains, spike-triggered
  averages, synchrony measures, and distance metrics.
- `plotting_functions.py` contains plotting utilities.
- `test_AQUA.py` contains unit tests comparing single-neuron and batch
  simulations.

The package metadata is defined in `AQUA_package/pyproject.toml`.
Package-specific usage notes are in `AQUA_package/README.md`.

### 2. Research experiments and notebooks

Most folders contain exploratory simulations, notebooks, generated figures, and
intermediate data from different research questions.

- `bursting/` explores how autapses affect bursting, firing-rate curves, gain,
  chaos, and transitions between regular-spiking and intrinsically bursting
  behaviour.
- `changepoint_analysis/` tests whether autaptic dynamics make input changes
  easier to detect from membrane potential traces.
- `micro-circuit/` studies small excitatory/inhibitory circuits, entrainment,
  synchrony, inhibition, and winner-takes-all style mechanisms.
- `Resonance/` explores resonance and integrator/resonator neuron behaviour.
- `classifying_neurons/` investigates neuron excitability classes and response
  profiles.
- `phase_response/` contains phase-response analyses with and without autapses.
- `frequency_filters/` explores how neurons respond to filtered inputs.

There are also folders based on specific models or papers, including
`Wang et al 2014/`, `yin et al 2018/`, `Guo et al 2016/`, and `bacci et al/`.

### 3. Earlier prototypes and ports

Several files and folders appear to contain earlier versions or parallel
implementations of the model.

- `AQUA_python/` contains earlier Python implementations.
- `AQUA_matlab/` contains MATLAB versions of the AQUA model.
- Top-level files such as `AQUA_class.py`, `batchAQUA.py`, `batchAQUA_GPU.py`,
  and `bifurcation_sim.py` are standalone scripts or earlier research utilities.

## Scientific Motivation

The central question is how an autapse, a neuron's self-connection, changes the
computational role of a neuron or small circuit.

Current project notes suggest several recurring themes:

- Autapses can push regular-spiking neurons toward intrinsically bursting-like
  behaviour.
- Autaptic feedback can create or expand bursting regimes.
- Autapse strength and delay can reshape firing-rate gain and input-output
  sensitivity.
- Certain autapse parameters may improve change-point detection from membrane
  potential data.
- In small circuits, autapses may support entrainment, sensory gating,
  attentional control, or winner-takes-all dynamics.

## Environment

The repository includes a Conda environment file:

```bash
conda env create -f aqua_environment.yml
conda activate aqua
```

Key dependencies include NumPy, SciPy, pandas, matplotlib, seaborn, Brian2,
CuPy, tqdm, pytest, and notebook/profiling tools.

## Tests

The package includes unit tests in `AQUA_package/aqua/test_AQUA.py`.

The test file notes that it should be run with `unittest` rather than `pytest`:

```bash
cd AQUA_package/aqua
python -m unittest -v test_AQUA.py
```

These tests compare the single-neuron and batch implementations across several
cases, including no autapse, zero-delay autapse, delayed autapse, mid-run
initialisation, and fast-spiking neuron dynamics.

## Current Status

This is an active research codebase rather than a polished application. It
contains a reusable package core, many exploratory notebooks, generated figures,
pickled simulation outputs, and paper-specific experiments.

For new development, `AQUA_package/aqua/` is the best starting point. For
scientific context, the `bursting/`, `changepoint_analysis/`, and
`micro-circuit/` folders contain the clearest research direction notes.
