#!/usr/bin/env python3
"""
Train neuron on nonlinear feature binding task.
"""

import numpy as np
import pickle

from dendrites import parameters1 as params
from dendrites import training
from dendrites import sequences

model = 'act'				# 'act', 'pas', 'pn'
input = 'opt'				# 'rate', 'temp', 'opt'
num_patterns = 4	        # 4, 9, ..., 100
seed = 0		            # random seed
param_sets = {'rate':[40, 0, 0.], 'temp':[2.5, 1, 1.], 'opt':[20, 1, 1.]}
r_max, num_t, s = param_sets[input]

### Simulation parameters ###
stim_dur = 400                          # stimulus duration (ms)
stim_on = 100							# stimulus on (ms)
stim_off = stim_on + stim_dur           # stimulus off (ms)
t_on = 0								# background on (ms)
t_off = stim_on 						# background off (ms)
r_0 = 1.25								# background rate (Hz)
dt = 0.1                    			# time step (ms)
v_init = -75.0							# initial voltage (mV)
t_window = 100							# plasticity window (ms, fixed)
offset = 2								# spike time offset (ms, fixed)
r_mean = 2.5							# mean input rate (Hz)
epochs = 1001       					# max training epochs
alpha0 = 2*1e-6/r_max 				    # initial learning rate
alpha_d = 1/125							# learning rate decay constant
w_jitter = 0.5							# perturbation to initial weights
jitter = 2.5							# stdev of Gaussian spike jitter (ms)
p = int(num_patterns**0.5)


def filenames(model, r_max, num_t, r_0, num_patterns, seed):
    """ Create filename for specified conditions.

    Parameters
    ----------
    model : str
        model type 'act', 'pas', 'pn'
    r_max : float
        max time-averaged input rate
    num_t : int
        number of precisely timed events per synapse
    r_0: float
        background rate
    num_patterns: int
        number of patterns to be classified
    seed : int
        random seed

    Returns
    -------
    model_file : str
        model filename
    kernel_file : str
        plasticity kernel filename
    """

    r_0_str = str(r_0)
    r_0_str = r_0_str.replace('.', 'p', 1)
    r_max_str = str(r_max)
    r_max_str = r_max_str.replace('.', 'p', 1)
    model_file = './outputs/fbt/fbt_'+model+'_'+r_max_str+'_'+str(num_t)+'_'+\
                 r_0_str+'_'+str(num_patterns)+'_'+str(seed)
    if model == 'act':
        kernel = 'kernel_fit_act1'
    elif model == 'pas':
        kernel = 'kernel_fit_pas1'
    else:
        kernel = 'kernel_fit_soma1'
    kernel_file = './input/' + kernel
    return model_file, kernel_file


def init_input(P, num_patterns, stim_on, stim_off, r_mean, r_max, num_t, s):
    """ Initialise input rates and spike time sequences for feature-binding task
    Parameters
    ----------
    P : dict
        model parameters
    num_patterns: int
        number of patterns to be classified
    stim_on/stim_off : int
        times of stimulus onset/offset
    r_mean: float
        average presynaptic population rate
    r_max : float
        max time-averaged input rate
    num_t : int
        number of precisely timed events per synapse
    s : float
        interpolates between rate and temporal patterns (mostly unused --
         to be removed)

    Returns
    -------
    rates_e, rates_i : list
        input rates for excitatory and inhibitory synapses
    S_E, S_I : list
        precisely timed events for excitatory and inhibitory synapses
    """
    N_e, N_i = P['N_e'], P['N_i']
    ind_e = np.arange(N_e)
    ind_i = np.arange(N_i)
    np.random.shuffle(ind_e)
    np.random.shuffle(ind_i)
    rates_e, rates_i = sequences.assoc_rates(num_patterns, N_e, N_i, r_mean,
                                             r_max)
    rates_e = [r[ind_e] for r in rates_e]
    rates_i = [r[ind_i] for r in rates_i]
    if s > 0:
        S_E, S_I = sequences.assoc_seqs(num_patterns, N_e, N_i, stim_on,
                                        stim_off, num_t)
        S_E = [s[ind_e] for s in S_E]
        S_I = [s[ind_i] for s in S_I]
        for s_e, r_e in zip(S_E, rates_e):
            s_e[r_e == 0] = np.inf
        for s_i, r_i in zip(S_I, rates_i):
            s_i[r_i == 0] = np.inf
    else:
        S_E, S_I = sequences.build_seqs(num_patterns, N_e, N_i, stim_on,
                                        stim_off, 0)
    return rates_e, rates_i, S_E, S_I


def init_weights(N_e, N_i, w_jitter):
    """ Initialise synaptic weights. Perturbed independently from initial values
     with noise w_jitter
    Parameters
    ----------
    N_e, N_i : int
        number of excitatory and inhibitory synapses
    w_jitter: float
        noise term
    Returns
    -------
    w_e, w_i : ndarray
        excitatory and inhibitory weights
    """
    w_e = np.ones(N_e)*[P['g_max_A'] + P['g_max_N']] + w_jitter*(P['g_max_A'] +
            P['g_max_N'])*(np.random.rand(N_e) - 1/2)
    w_i = np.ones(N_i)*[P['g_max_G']] + w_jitter*P['g_max_G']*(
            np.random.rand(N_i) - 1/2)
    return w_e, w_i


def init_labels(num_patterns):
    """
    Initialise classification labels. Nonlinear contingencies for 2x2 task,
    random assignment for 3x3 and above.

    Parameters
    ----------
    num_patterns : int
        number of input patterns to be classified
    Returns
    -------
    L : ndarray
        classification labels (+1/-1 for preferred/non-preferred input patterns).
    """
    p = int(num_patterns**0.5)
    if num_patterns > 4:
        L = sequences.assign_labels(num_patterns)
    else:
        L = np.ones(p) - 2*np.eye(p)
    return L.flatten()


### Initialise ###
model_file, kernel_file = filenames(model, r_max, num_t, r_0, num_patterns, seed)
try:  # check for checkpointed file
    rates_e, rates_i, S_E, S_I, Labels, E, E_P, W_E, W_I, P, kernel = \
        pickle.load(open(model_file, 'rb'))
except FileNotFoundError:
    P = params.init_params()
    (
        P['v_init'], P['epochs'], P['alpha0'], P['alpha_d'], P['offset'],
        P['t_window'], P['jitter'], P['dt'], P['s'], P['model_file'], P['seed'],
        P['t_on'], P['t_off'], P['stim_on'], P['stim_off'], P['r_0'],
        P['r_mean'], P['r_max'], P['num_t']
    ) = (
            v_init, epochs, alpha0, alpha_d, offset, t_window, jitter, dt, s,
            model_file, seed, t_on, t_off, stim_on, stim_off, r_0, r_mean,
            r_max, num_t
        )

    if model == 'pas':
        P['active_n'] = False
        P['g_max_A'] /= 5
        P['g_max_N'] /= 5
    elif model == 'pn':
        P['locs_e'] = np.array(P['soma'])
        P['locs_i'] = np.array(P['soma'])
    if t_off > stim_on:
        P['g_max_A'] /= (r_0 + r_mean)/r_mean
        P['g_max_N'] /= (r_0 + r_mean)/r_mean
        P['g_max_G'] /= (r_0 + r_mean)/r_mean
    np.random.seed(seed)
    _, kernel = pickle.load(open(kernel_file, 'rb'))
    rates_e, rates_i, S_E, S_I, = init_input(P, num_patterns, stim_on, stim_off,
                                            r_mean, r_max, num_t, s)
    Labels = init_labels(num_patterns)
    w_e, w_i = init_weights(P['N_e'], P['N_i'], w_jitter)
    E = [np.nan]
    E_P = [[0.5] for k in range(num_patterns)]
    W_E = [np.array(w_e)]
    W_I = [np.array(w_i)]

### Train ###
E, E_P, W_E, W_I, P = training.train(rates_e, rates_i, S_E, S_I, Labels, E, E_P,
                                    W_E, W_I, P, kernel)

### Save ###
pickle.dump([rates_e, rates_i, S_E, S_I, Labels, E, E_P, W_E, W_I, P, kernel],
            open(model_file + '_complete', 'wb'))
