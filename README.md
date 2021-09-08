# Dendrites

Code associated with the paper "Bicknell B.A. & Häusser M., (2021). A synaptic learning rule for exploiting nonlinear dendritic computation".

Requirements
-----------------
numpy\
matplotlib\
numba\
NEURON==7.7 (tested version)\
neurom==1.8.0 (currently must use version<2.0)	

Directories
-----------
dendrites: main package for building and simulating models\
input: fitted plasticity kernels and morphology files\
mod_files: channel and synapse models for NEURON simulations (need to compile with "nrnivmodl ./mod_files" from command line)\
outputs: for saving results

Example scripts
---------------
sim_example.py: Simulate the response of the model to random input, using either the custom simulator or the equivalent NEURON implementation. Computes and displays the voltage gradients (dv_soma/dw) for each synaptic activation for the first few output spikes. The first run with the custom simulator will be slower than normal, since Numba needs to cache the jit-compiled functions.

train_example.py: Train a biophysical model to solve a nonlinear feature-binding task. Options that can be set include the type of model (active, passive, point neuron), flavour of input, number of input patterns to be classified, and other stimulus, noise and learning parameters. With default settings (active model, precisely timed burst input, 4 patterns), this may take around 10 minutes.

Note that most of the simulations in the paper were run remotely on a cluster. For instance, using sta_sim.py with 500 different seeds to get a large amount of data for fitting plasticity kernels, or run_fbt_N.py for model training, which can have run times of up to 2 days. So caution is advised if experimenting with these locally.

Contact
-------
For any questions please contact B.A.B. via email.