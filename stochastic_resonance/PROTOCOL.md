### Stochastic Resonance

There is some evidence that the autaptic neuron undergoes stochastic resonance. 

Idea is to simulate RS neurons with different autapse strengths at an injected current that is just below threshold with 
different noise levels. Quantify firing rate or spike count over a long trial.

Plot the FR/spike count as a function of the input noise level. 

Neuron model:

# neuron model (RS integrator)
RS = {'name': 'RS_ref', 'C': 100, 'k': 0.7, 'v_r': -60, 'v_t': -40, 'v_peak': 35,
     'a': 0.03, 'b': -2, 'c': -50, 'd': 100, 'e': 0., 'f': 0., 'tau': 0.}