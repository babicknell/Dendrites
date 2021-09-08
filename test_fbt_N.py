#!/usr/bin/env python3
"""
Test trained models. Requires output from 'run_fbt_N'.
"""

import numpy as np
import pickle

from dendrites import sequences
from dendrites import neuron_model


model = './outputs/fbt/fbt_act_20_1_1p25_4'
seeds = [0]     # seeds of trained model to test
reps = 20       # number of testing reps per pattern
test_seed = 0   # random seed for testing simulations


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
        t_spikes = spikes*dt - offset
    else:
        t_spikes = []
    return t_spikes


def clas_error(T_spikes, Labels):
    """Compute classificition errors from testing simulations

    Parameters
    ----------
    T_spikes : list
        spike times from testing simualtions
    Labels : list
        classifcation labels
    Returns
    -------
    E : list
        errors for each rep of each input pattern
    """
    E = []
    for k, t_s in enumerate(T_spikes):
        num_spikes = np.array([len(t) for t in t_s])
        num_spikes[num_spikes > 0] = 1
        if Labels[k] > 0:
            err = 1 - num_spikes
        else:
            err = num_spikes
        E.append(err)
    return E


for seed in seeds:
    model_file = model+'_'+str(seed)
    try:
        rates_e, rates_i, S_E, S_I, Labels, E, E_P, \
            W_E, W_I, P, kernel = pickle.load(open(model_file + '_complete', 'rb'))
    except:
        continue
    cell = neuron_model.NModel(P)
    t_on, t_off, stim_on, stim_off, jitter, r_0, dt, v_init, offset, num_t,\
    r_max, s = (P['t_on'], P['t_off'], P['stim_on'], P['stim_off'], P['jitter'],
        P['r_0'], P['dt'], P['v_init'], P['offset'], P['num_t'], P['r_max'],
        P['s'])
    cell.set_weights(W_E[-1], W_I[-1])
    np.random.seed(int(test_seed))
    T_spikes = []
    if num_t > 0:
        sigma = jitter*s*1e-3*r_max*(stim_off - stim_on)/num_t
    else:
        sigma = jitter
    pre_syn = sequences.PreSyn(r_0, sigma)
    T = max(t_off, stim_off+50)
    for ind, _ in enumerate(Labels):
        t_s_rep = []
        for rep in range(reps):
            S_e = [pre_syn.spike_train(t_on, t_off, stim_on, stim_off, s,
                    rates_e[ind][k], S_E[ind][k]) for k in range(P['N_e'])]
            S_i = [pre_syn.spike_train(t_on, t_off, stim_on, stim_off, s,
                    rates_i[ind][k], S_I[ind][k]) for k in range(P['N_i'])]
            t, v = cell.simulate(T, dt, v_init, S_e, S_i)
            t_spikes = spike_times(dt, v)
            t_s_rep.append(t_spikes)
        T_spikes.append(t_s_rep)

    E = clas_error(T_spikes, Labels)

    pickle.dump([T_spikes, Labels, E, P, rates_e, rates_i, S_E, S_I, W_E[-1],
                W_I[-1]], open(model_file + '_test', 'wb'))
