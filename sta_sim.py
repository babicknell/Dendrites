#!/usr/bin/env python3
"""
Simulate model with random input and compute synaptic influence before each spike.
"""

import numpy as np
import pickle
import sys

from dendrites import parameters1 as params
from dendrites import sequences
from dendrites import comp_model

seed = sys.argv[1]
model = 'act1'
results_file = './outputs/sta/sta_'+model+'_data/sta_'+model+'_'+seed

### Simulation parameters ###
T = 10000            # simulation time (ms)
dt = 0.1            # time step (ms)
v_init = -65        # initial voltage (mV)
reps = 10           # number of reps
w_jitter = 0.5      # perturbation to initial weights
t_window = 151	    # analysis window (ms)
offset = 2		    # spike time offset (ms)
mu = 0              # mean parameter for lognormal rate dist
sigma = 1           # sd parameter for lognormal rate dist
np.random.seed(int(seed))


def init_rand_sequence(rates_e, rates_i, T):
    """
    build sequences of Poisson presynaptic input to excitatory and inhibitory
    synapses

    Parameters
    ----------
    rates_e, rates_i : list
        excitatory and inhibitory input rates
    T : int
        total simulation time (ms)
    Returns
    -------
    S_e, S_i :  list
        excitatory and inhibitory presynaptic spike times
    """
    S_e = sequences.build_rate_seq(rates_e, 0, T)
    S_i = sequences.build_rate_seq(rates_i, 0, T)
    return S_e, S_i


def init_weights(P):
    """
    Initialise synaptic weights by perturbing initial values.

    Parameters
    ----------
    P : dict
        model parameters

    Returns
    -------
    w_e, w_i :  ndarray
        excitatory and inhibitory weight vectors
    """
    w_e = np.ones(P['N_e'])*[P['g_max_A'] + P['g_max_N']] + w_jitter*(P['g_max_A'] +
            P['g_max_N'])*(np.random.rand(P['N_e']) - 1/2)
    w_i = np.ones(P['N_i'])*[P['g_max_G']] + w_jitter*P['g_max_G']*(
            np.random.rand(P['N_i']) - 1/2)
    return w_e, w_i


def spike_times(dt, v):
    """ Get spike times from voltage trace.

    Parameters
    ----------
    dt : float
        simulation timestep
    v : ndarray
        compartment voltages v=v[compartment, time]
    Returns
    -------
    t_spike : ndarray
        spike times
    """
    thresh_cross = np.where(v[0, :] > 0)[0]
    if thresh_cross.size > 0:
        spikes = np.where(np.diff(thresh_cross) > 1)[0] + 1
        spikes = np.insert(spikes, 0, 0)
        spikes = thresh_cross[spikes]
        t_spike = spikes*dt - offset
    else:
        t_spike = np.array([])
    return t_spike


def get_grad(cell, t0, t1, dt, S_e, S_i, soln, stim):
    """ Get gradients (dv_soma/dw) for individual synaptic activations by
    solving variational equations between times t0 and t1 with fast somatic
    conductances set to zero.

    Parameters
    ----------
    cell : dendrites.comp_model.CModel object
        compartmental model instance used for simulation
    t0, t1 :  float
        initial and final times for computing gradients
    dt : float
        simulation time step
    S_e, S_i : array_like
        input spike patterns for E and I synapses
    soln : list
        model states [v, m, h, n, p] (output from cell.simulate)
    stim : list
        synapse indices and states [ind_e, ind_i, A_r, A_d, N_r, N_d, G_r, G_d]

    Returns
    -------
    v_pre : list
        voltage throughout the morphology preceding somatic spike
    F_e, F_i : list
        computed gradients with associated synapse indices and presynaptic
        spike times F_e = [f_e, z_ind_e, z_e]
    """
    IC_sub = cell.set_IC(soln, stim, int(t0/dt))
    g_na_temp, g_k_temp = P['g_na'], P['g_k']
    cell.P['g_na'] = 0
    cell.P['g_k'] = 0
    t_s, soln_s, stim_s = cell.simulate(t0, t1, dt, IC_sub, S_e, S_i)
    Z_e = sequences.subsequence(S_e, t0, t1)
    Z_i = sequences.subsequence(S_i, t0, t1)
    Z_e, z_ind_e = sequences.rate2temp(Z_e)
    Z_i, z_ind_i = sequences.rate2temp(Z_i)
    f_e, f_i = cell.grad_w(soln_s, stim_s, t_s, dt, Z_e, Z_i, z_ind_e, z_ind_i)
    cell.P['g_na'] = g_na_temp
    cell.P['g_k'] = g_k_temp
    v_pre = soln[0][:, int(t1 / dt)]
    f_e = f_e[:, -1]
    f_i = f_i[:, -1]
    z_e = Z_e - t1
    z_i = Z_i - t1
    return [v_pre], [f_e, z_ind_e, z_e], [f_i, z_ind_i, z_i]


P = params.init_params()
cell = comp_model.CModel(P)
np.random.seed(int(seed))

F_e = []
F_i = []
V = []
W_e = []
W_i = []
for rep in range(reps):
    rates_e, rates_i = sequences.lognormal_rates(1, P['N_e'], P['N_i'], mu, sigma)
    w_e, w_i = init_weights(P)
    cell.set_weights(w_e, w_i)
    S_e, S_i = init_rand_sequence(rates_e[0], rates_i[0], T)
    t, soln, stim = cell.simulate(0, T, dt, v_init, S_e, S_i)
    v = soln[0]
    t_spikes = spike_times(dt, v)
    num_spikes = len(t_spikes)
    if num_spikes > 0:
        incl = np.where(np.diff(np.insert(t_spikes, 0, 0)) > t_window)[0]
        t1 = t_spikes[incl]
        t0 = t1 - t_window
        num_test_spikes = len(t1)
        for k in range(num_test_spikes):
            v_pre, f_e, f_i = get_grad(cell, t0[k], t1[k], dt, S_e, S_i, soln, stim)
            V.append(v_pre)
            F_e.append(f_e)
            F_i.append(f_i)
            W_e.append(np.array(w_e[f_e[1]]))
            W_i.append(np.array(w_i[f_i[1]]))

if seed == '0':
    pickle.dump([cell, V, F_e, F_i, W_e, W_i], open(results_file, 'wb'))
else:
    pickle.dump([[], V, F_e, F_i, W_e, W_i], open(results_file, 'wb'))
