#!/usr/bin/env python3
"""
Example script to demonstrate simulation of the model and calculation of
dv_soma/dw.
"""

import numpy as np
import pickle

from matplotlib import pyplot
from neuron import h, gui

from dendrites import comp_model
from dendrites import neuron_model
from dendrites import parameters1
from dendrites import plot_raster
from dendrites import sequences
from dendrites import training

T = 2000        # simulation time (ms)
dt = 0.1        # time step (ms)
v_init = -75    # initial voltage (mV)
seed = 1        # random seed
kernel_fit = './input/kernel_fit_act1'  # fitted plasticity kernel
P = parameters1.init_params()            # stores model parameters in dict P

c_model = False # True for custom compartmental model with explicit gradient
                 # calculations, False for Neuron model using fitted approximation
num_spikes = 3   # max number of somatic spikes for which to compute gradients
np.random.seed(seed)


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
        t_spike = spikes*dt - 2
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
        model states [v, m, h, n, p, hcn] (output from cell.simulate)
    stim : list
        synapse indices and states [ind_e, ind_i, A_r, A_d, N_r, N_d, G_r, G_d]

    Returns
    -------
    F_e, F_i : list
        computed gradients with associated synapse indices and presynaptic
        spike times F_e = [f_e, z_ind_e, z_e]
    """
    IC_sub = cell.set_IC(soln, stim, int(t0 / dt))
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
    z_e = Z_e - t1
    z_i = Z_i - t1
    F_e = [f_e[:, -1], z_ind_e, z_e]
    F_i = [f_i[:, -1], z_ind_i, z_i]
    return F_e, F_i


def get_k_grad(t0, t1, dt, S_e, S_i, v, kernel_params):
    """ Get gradients (dv_soma/dw) for individual synaptic activations from
    fitted approximations.

    Parameters
    ----------
    t0, t1 :  float
        initial and final times for computing gradients
    dt : float
        simulation time step
    S_e, S_i : array_like
        input spike patterns for E and I synapses
    v : ndarray
        compartment voltages from simulation
    kernel_params : list
        parameters for fitted kernels (see dendrites.kernels)

    Returns
    -------
    F_e, F_i : list
        approximated gradients with associated synapse indices and presynaptic
        spike times F_e = [f_e, z_ind_e, z_e]
    """
    Z_e = sequences.subsequence(S_e, t0, t1)
    Z_i = sequences.subsequence(S_i, t0, t1)
    Z_e, z_ind_e = sequences.rate2temp(Z_e)
    Z_i, z_ind_i = sequences.rate2temp(Z_i)
    f_e, f_i = training.kernel_grad_ss(int(t0 / dt), int(t1 / dt), S_e, S_i,
                        cell.seg_e, cell.seg_i, cell.b_type_e, cell.b_type_i, v,
                        kernel_params, dt)
    F_e = [f_e, z_ind_e, Z_e - t1]
    F_i = [f_i, z_ind_i, Z_i - t1]
    return F_e, F_i


### Run Simulation ###
rates_e, rates_i = sequences.lognormal_rates(1, P['N_e'], P['N_i'], 0, 1)
S_e = sequences.build_rate_seq(rates_e[0], 0, T)
S_i = sequences.build_rate_seq(rates_i[0], 0, T)
if c_model:
    cell = comp_model.CModel(P)
    t, soln, stim = cell.simulate(0, T, dt, v_init, S_e, S_i)
    v = soln[0]
else:
    cell = neuron_model.NModel(P)
    _, kernel_params = pickle.load(open(kernel_fit, 'rb'))
    cell.kernel = kernel_params
    t, v = cell.simulate(T, dt, v_init, S_e, S_i)

### Compute Gradients ###
t_window = 100  # synaptic plasticity window (fixed parameter)
t1 = spike_times(dt, v)
if len(t1) > 0:
    t0 = t1 - t_window
    E_data = []
    I_data = []
    for spike in range(min(num_spikes, len(t1))):
        if c_model:
            F_e, F_i = get_grad(cell, t0[spike], t1[spike], dt, S_e,
                                S_i, soln, stim)
        else:
            F_e, F_i = get_k_grad(t0[spike], t1[spike], dt, S_e, S_i, v,
                                  kernel_params)
        E_data.append(F_e)
        I_data.append(F_i)

### Plot Results ###
fig, ax = pyplot.subplots(figsize=(8, 2.5))
ax.plot(t, v[0, :], 'k')
ax.set_xlim([0, T])
ax.set_ylim([-80, 40])
ax.set_yticks(np.arange(-75, 50, 25))
ax.set_xlabel('time (ms)', fontsize=14)
ax.set_ylabel('V' + r'$_{soma}$' + ' (mV)', fontsize=14)
ax.tick_params(labelsize=12)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
pyplot.tight_layout()

if len(t1) > 0:
    i_positions, index, boundaries = plot_raster.raster_params(cell)
    for e_dat, i_dat in zip(E_data, I_data):
        f_e, e_ind, s_e = e_dat
        f_i, i_ind, s_i = i_dat
        ff_e = np.array(np.abs(f_e)) / np.max(np.abs(f_e))
        ff_i = np.array(f_i)
        if np.min(ff_i) < 0:
            ff_i = -ff_i / np.min(ff_i)
        plot_raster.plot_grad_example(ff_e, ff_i, e_ind, i_ind, s_e, s_i,
                                     t_window, i_positions, index, boundaries)

pyplot.show()
