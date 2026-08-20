# AQUA Package

This directory contains the reusable Python package for the AQUA model:
an **Adaptive QUadratic integrate-and-fire neuron with an Autapse**.

The wider repository contains exploratory notebooks, figures, and research
scripts. This package is the best starting point for reusable model code.

## Package Contents

- `aqua/AQUA_general.py`
  - Single-neuron AQUA simulation.
  - Uses RK2 integration.
  - Tracks membrane potential `v`, recovery variable `u`, and autaptic current
    `w`.

- `aqua/batchAQUA_general.py`
  - Vectorized batch simulation for many parameter sets.
  - Includes a Brian2 bridge through `batchAQUA.meetBrian`.
  - Supports standard, biexponential, and uniform autapse waveforms.
  - Supports fixed and sampled autapse delays.

- `aqua/stimulus.py`
  - Injected-current generators such as steps, ramps, Ornstein-Uhlenbeck noise,
    spike trains, sinusoids, and filtered white noise.

- `aqua/utils.py`
  - Spike train conversion and analysis helpers.
  - Includes spike-triggered averages, ISI traces, synchrony/distance metrics,
    and ISI peak analysis.

- `aqua/plotting_functions.py`
  - Convenience plotting functions for membrane traces, ISI distributions,
    rasters, and return maps.

- `aqua/test_AQUA.py`
  - Unit tests comparing the single-neuron and batch implementations.

## Model State

The model state is stored as:

```text
x = [v, u, w]
```

where:

- `v` is membrane potential.
- `u` is the recovery variable.
- `w` is the autaptic current.

The standard autapse is parameterized by:

- `e`: decay rate of the autaptic current.
- `f`: current added to `w` after a spike.
- `tau`: delay before the autaptic current affects the neuron.

## Basic Usage

```python
import numpy as np
from aqua.AQUA_general import AQUA
from aqua.stimulus import step_current

RS = {
    "name": "RS",
    "C": 100,
    "k": 0.7,
    "v_r": -60,
    "v_t": -40,
    "v_peak": 35,
    "a": 0.03,
    "b": -2,
    "c": -50,
    "d": 100,
    "e": 0.03,
    "f": 8.0,
    "tau": 0.5,
}

dt = 0.01
T = 500
N_iter = int(T / dt)

neuron = AQUA(RS)
neuron.Initialise(x_start=[-65, 0, 0], t_start=0)

I_inj = step_current(N_iter, dt, y_0=0, delay=50, I_h=100)
X, times, spikes = neuron.update_RK2(dt, N_iter, I_inj)
```

## Environment

From the repository root:

```bash
conda env create -f aqua_environment.yml
conda activate aqua
```

For editable development from this directory:

```bash
cd AQUA_package
pip install -e .
```

## Tests

The current tests are written for `unittest`:

```bash
cd AQUA_package/aqua
python -m unittest -v test_AQUA.py
```

The tests check consistency between the single-neuron and batch simulators for
no autapse, zero-delay autapse, delayed autapse, mid-run initialization, and
fast-spiking neuron dynamics.
