# classifying_neurons

Tests that classify the excitability of Izhikevich-type AQUA neurons with and without an autapse
(no autapse, excitatory, inhibitory). Results are read in terms of how the autaptic current moves the nullclines.

The autapse has three parameters, each sampled around realistic values:

| Param | Meaning |
|-------|---------|
| `tau` | time delay: synaptic transmission + PSC rise time (0.3–0.5 ms) |
| `f`   | conductance: the maximum PSC |
| `e`   | decay rate: `1/e` is the time for the PSC to fall to ~37% of its peak |

Derived quantities: autapse strength (integral of the PSC, a function of `f` and `e`) and effective delay (`tau + 1/e`).

## Tests

- **Step currents** of different heights → steady-state and instantaneous F-I curves (Hodgkin class I / II / III).
- **Ramp currents**, up and down → F-I curves and hysteresis (spiking class).
- **Bifurcation diagrams**: ISIs against injected current.
- **Adaptation**: how the intrinsic adaptation rate (`a`) and the autapse decay (`e`) change the autapse's effect.
- **Phase response curves**: planned, not yet implemented.

`project_plan` lists further ideas: STA/STC, first return maps, periodic and noisy inputs, statistical complexity, chaotic resonance, criticality.

## Files

| File | Contents |
|------|----------|
| `functions.py` | `FI_from_steps`, `instant_FI_from_steps` (batch of neurons, one step height each) and `FI_from_ramps` (single neuron) |
| `RS_excitability_test.ipynb` | F-I curves (steps and ramps) for the RS neuron; plots saved to `RS_excitability/` |
| `FS_excitability_test.ipynb` | The same for the FS neuron (saves to `FS_excitability/`, which does not exist yet) |
| `RS_bifurcation_test.ipynb` | Step-current bifurcation diagrams; plots in `bifurcations/` |
| `adaptation_test.ipynb` | Effect of adaptation rate and autapse decay on ramp responses |
| `phase_response.ipynb` | Stub for phase response curves |
| `rand_test.ipynb` | numpy scratch tests |
| `main.py`, `simulation.py` | CLI pipeline: sweep `e`, `f`, `tau` at a fixed drive, store ISI distributions in a pickled DataFrame, plot heat maps |
| `CLI.py` | argparse options: `--sim`, `--config`, `--pickle`, `--save` |
| `CF.py` | `read_conf`: reads an `.ini` file into a dict of sections |
| `RS_config.ini` | Placeholder config, no values filled in |
| `#README` | Original design notes |
| `project_plan`, `protocol.txt` | Research plan and test protocol notes |

## Output folders

- `RS_excitability/`: F-I curves (no autapse, excitatory, inhibitory; steady-state, instantaneous, up and down ramps)
- `bifurcations/`: bifurcation diagrams for RS and FS, each with no, excitatory and inhibitory autapse
- `first_pass_img/`: first-pass class 1 and class 2 responses with and without autapses

## Running

The code imports modules from the parent directory via `sys.path.append("..\\")`, a Windows-style path.
Current state of those imports:

- The notebooks and `functions.py` import `AQUA_general`, `batchAQUA_general`, `stimulus` and `plotting_functions`.
  None of these are in the parent directory at the moment (it has `AQUA_class.py` and `batchAQUA.py`).
- `main.py` won't run as written: it uses `simulation`, `pickle` and `functions.plot_heatmap` without importing or
  defining them, and `RS_config.ini` has none of the `[Simulation]`, `[Autapse]` or `[Output]` keys that `simulation.py` reads.

To run with a complete config:

```bash
python main.py --sim --config RS_config.ini   # run the sweep and plot
python main.py --pickle results.pkl           # plot saved results
```
