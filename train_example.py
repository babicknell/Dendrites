#!/usr/bin/env python3
"""
Train neuron on nonlinear feature binding task.
"""

import numpy as np
import pickle

from matplotlib import pyplot

from dendrites import neuron_model
from dendrites import parameters1
from dendrites import training
from dendrites import sequences

model = 'act'			 	# act, pas, pn
input = 'opt'				# rate, temp, opt
num_patterns = 4	 		# 4, 9, ..., 100
seed = 1		 			# random seed
model_file = './outputs/example_1'  # for saving output
save_output = False
param_sets = {'rate':[40., 0, 0.], 'temp':[2.5, 1, 1.], 'opt':[20., 1, 1.]}
r_max, num_t, s = param_sets[input]


### Simulation Parameters ###
stim_dur = 400							# stimulus duration
stim_on = 100							# stimulus on
stim_off = stim_on + stim_dur           # stimulus off
t_on = 0								# background on
t_off = stim_on							# background off
r_0 = 1.25								# background rate
dt = 0.1            					# time step
v_init = -75.0							# initial voltage
t_window = 100							# plasticity window (fixed parameter)
offset = 2								# spike time offset (fixed parameter)
r_mean = 2.5							# mean input rate (Hz)
epochs = 1001       					# max training epochs
alpha0 = 2*1e-6/r_max 					# initial learning rate
alpha_d = 1/125							# learning rate decay constant
w_jitter = 0.5							# perturbation to initial weights
jitter = 2.5							# stdev of spike time jitter (ms)


def init_input(P, num_patterns, stim_on, stim_off, r_mean, r_max, num_t, s):
    """
    Initialise input rates and spike time sequences for feature-binding task.

    Parameters
    ----------
    P : dict
        model parameters
    num_patterns : int
        number of input patterns to be classified
    stim_on, stim_off : int
        time of stimulus onset and termination (ms)
    r_mean : float
        average presynaptic population rate (Hz)
    r_max : float
        time averaged input rate to active synapses
    num_t : int
        number of precisely timed events per active synapse
    s : float
        interpolates between rate (s=0) and temporal (s=1) input signals (mostly
        unused parameter -- to be removed)

    Returns
    -------
    rates_e, rates_i : list
        excitatory and inhibitory input rates for all patterns
    S_E, S_I : list
        times of precisely timed events for all patterns
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
        S_E, S_I = sequences.assoc_seqs(num_patterns, N_e, N_i, stim_on, stim_off,
                                        num_t)
        S_E = [s[ind_e] for s in S_E]
        S_I = [s[ind_i] for s in S_I]
        for s_e, r_e in zip(S_E, rates_e):
            s_e[r_e == 0] = np.inf
        for s_i, r_i in zip(S_I, rates_i):
            s_i[r_i == 0] = np.inf
    else:
        S_E, S_I = sequences.build_seqs(num_patterns, N_e, N_i, stim_on, stim_off,
                                        0)
    return rates_e, rates_i, S_E, S_I


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
    L = L.flatten()
    return L


### Initialise ###
P = parameters1.init_params()
(
    P['v_init'], P['epochs'], P['alpha0'], P['alpha_d'], P['offset'],
    P['t_window'], P['jitter'], P['dt'], P['s'], P['model_file'], P['seed'],
    P['t_on'], P['t_off'], P['stim_on'], P['stim_off'], P['r_0'], P['r_mean'],
    P['r_max'], P['num_t']
) = (
        v_init, epochs, alpha0, alpha_d, offset, t_window, jitter, dt, s,
        model_file, seed, t_on, t_off, stim_on, stim_off, r_0, r_mean, r_max,
        num_t
    )

if model == 'pas':
    P['active_n'] = False
    P['g_max_A'] /= 5
    P['g_max_N'] /= 5
    kernel_file = './input/kernel_fit_pas1'
elif model == 'pn':
    P['locs_e'] = np.array(P['soma'])
    P['locs_i'] = np.array(P['soma'])
    kernel_file = './input/kernel_fit_soma1'
else:
    kernel_file = './input/kernel_fit_act1'
_, kernel = pickle.load(open(kernel_file, 'rb'))
np.random.seed(seed)
rates_e, rates_i, S_E, S_I, = init_input(P, num_patterns, stim_on, stim_off,
                                        r_mean, r_max, num_t, s)
Labels = init_labels(num_patterns)
w_e, w_i = init_weights(P)
E = [num_patterns/2]
E_P = [[0.5] for k in range(num_patterns)]
W_E = [np.array(w_e)]
W_I = [np.array(w_i)]

### Train Model ###
E, E_P, W_E, W_I, P = training.train(rates_e, rates_i, S_E, S_I, Labels, E, E_P,
                                    W_E, W_I, P, kernel)
E = E[1:]
E_sm = np.convolve(E, np.ones(20)/20, mode='valid')/num_patterns
if save_output:
    pickle.dump([rates_e, rates_i, S_E, S_I, Labels, E, E_P, W_E, W_I, P, kernel],
        open(model_file, 'wb'))

#### Test Trained Model ####
reps = 10
cell = neuron_model.NModel(P)
np.random.seed(seed)
cell.set_weights(W_E[-1], W_I[-1])
if num_t > 0:
    sigma = jitter*s*1e-3*r_max*(stim_off - stim_on)/num_t
else:
    sigma = jitter
pre_syn = sequences.PreSyn(r_0, sigma)
T = max(t_off, stim_off + 100)
V = []
for ind, _ in enumerate(Labels):
    v_rep = []
    for rep in range(reps):
        S_e = [pre_syn.spike_train(t_on, t_off, stim_on, stim_off, s,
                            rates_e[ind][k], S_E[ind][k]) for k in range(P['N_e'])]
        S_i = [pre_syn.spike_train(t_on, t_off, stim_on, stim_off, s,
                            rates_i[ind][k], S_I[ind][k]) for k in range(P['N_i'])]
        t, v = cell.simulate(T, dt, v_init, S_e, S_i)
        v_rep.append(v[0, :])
    V.append(v_rep)
V = np.array(V)

#### Plot Results ####
fig, ax = pyplot.subplots(1, 2, figsize=(9, 2.5))
for ind, label in enumerate(Labels):
    color = next(ax[0]._get_lines.prop_cycler)['color']
    if label < 0:
        for rep in range(reps):
            ax[0].plot(t, V[ind, rep].T, color=color)
    else:
        for rep in range(reps):
            ax[1].plot(t, V[ind, rep].T, color=color)

for a in ax:
    a.set_xlim([0, T])
    a.set_ylim([-80, 35])
    a.set_yticks(np.arange(-75, 50, 25))
    a.set_xlabel('time (ms)', fontsize=14)
    a.set_ylabel('V' + r'$_{soma}$'+' (mV)', fontsize=14)
    a.tick_params(labelsize=12)
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.yaxis.set_ticks_position('left')
    a.xaxis.set_ticks_position('bottom')
ax[0].title.set_text(r'$\ominus$' + ' patterns')
ax[1].title.set_text(r'$\oplus$' + ' patterns')
pyplot.tight_layout()

fig1, ax1 = pyplot.subplots(figsize=(4, 3))
ax1.plot(range(1, len(E_sm)+1), E_sm)
ax1.set_ylim([-0.05, 0.55])
ax1.set_yticks(np.arange(0, 0.6, 0.1))
ax1.set_xlabel('epoch', fontsize=14)
ax1.set_ylabel('training error (smoothed)', fontsize=14)
ax1.tick_params(labelsize=12)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')
pyplot.tight_layout()
pyplot.show()
