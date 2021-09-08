"""
Training algorithm and helper functions for spike pattern classification.
"""

import numpy as np
import pickle

from dendrites import kernels
from dendrites import sequences
from dendrites import neuron_model
from dendrites import comp_model


def spike_inds(v, offset):
    """Get spike time indices from voltage trace v.

    Parameters
    ----------
    v : ndarray
        vector of model compartment voltages (v[0] = somatic voltage)
    offset : int
        time before zero-crossing that defines 'spike time'

    Returns
    -------
    t_spikes : ndarray
        spike time indices

    """
    thresh_cross = np.where(v[0, :] > 0)[0]
    if thresh_cross.size > 0:
        spikes = np.where(np.diff(thresh_cross) > 1)[0] + 1
        spikes = np.insert(spikes, 0, 0)
        spikes = thresh_cross[spikes]
        t_spikes = spikes - offset
    else:
        t_spikes = []
    t_spikes = np.array(t_spikes)
    return t_spikes


def kernel_grad(t0_ind, t1_ind, S_e, S_i, seg_e, seg_i, b_type_e, b_type_i, v, kernel,
                dt):
    """Look up fitted plasticity kernels to approximate gradients. Sums over all
    inputs to a synapse.

    Parameters
    ----------
    t0, t1 : int
        time indices to define range for computing gradients
    S_e, S_i : list
        presynaptic spike times for E and I synapses
    seg_e, seg_i : ndarray
        segment locations of E and I synapses
    b_type_e, b_type_i : list
        branch types for all synapses (soma=-1, basal=0, oblique=1, apical=2)
    v : ndarray
        vector of model compartment voltages
    kernel : list
        parameters of plasticity kernel
    dt : float
        timestep

    Returns
    -------
    f_e, f_i : ndarray
        gradients (dv_soma/dw) for E and I synapses
    """
    t0 = t0_ind*dt
    t1 = t1_ind*dt
    Z_e = sequences.subsequence(S_e, t0, t1)
    Z_i = sequences.subsequence(S_i, t0, t1)
    Z_e, z_ind_e = sequences.rate2temp(Z_e)
    Z_i, z_ind_i = sequences.rate2temp(Z_i)
    s_e = t1 - Z_e
    s_i = t1 - Z_i
    f_e = np.zeros(len(S_e))
    f_i = np.zeros(len(S_i))
    if len(kernel) == 2:  # somatic only
        p_es, p_is = kernel
        for k, s in enumerate(s_e):
            f_e[z_ind_e[k]] += p_es(s)
        for k, s in enumerate(s_i):
            f_i[z_ind_i[k]] += p_is(s)
    else:  # basal and apical
        p_eb, p_ea, p_ib, p_ia = kernel
        if np.size(p_eb[0]) == 1:
            for k, s in enumerate(s_e):
                if b_type_e[z_ind_e[k]] == 2:
                    f_e[z_ind_e[k]] += p_ea(s)
                else:
                    f_e[z_ind_e[k]] += p_eb(s)
            for k, s in enumerate(s_i):
                if b_type_i[z_ind_i[k]] == 2:
                    f_i[z_ind_i[k]] += p_ia(s)
                else:
                    f_i[z_ind_i[k]] += p_ib(s)
        else:
            if v.ndim == 1:
                v_dend_e = v[seg_e[z_ind_e]]
                v_dend_i = v[seg_i[z_ind_i]]
            else:
                v_dend_e = v[seg_e[z_ind_e], t1_ind]
                v_dend_i = v[seg_i[z_ind_i], t1_ind]

            for k, (s, u) in enumerate(zip(s_e, v_dend_e)):
                if b_type_e[z_ind_e[k]] == 2:
                    f_e[z_ind_e[k]] += kernels.eval_poly(s, u, p_ea)
                else:
                    f_e[z_ind_e[k]] += kernels.eval_poly(s, u, p_eb)
            for k, (s, u) in enumerate(zip(s_i, v_dend_i)):
                if b_type_i[z_ind_i[k]] == 2:
                    f_i[z_ind_i[k]] += kernels.eval_poly(s, u, p_ia)
                else:
                    f_i[z_ind_i[k]] += kernels.eval_poly(s, u, p_ib)
    return f_e, f_i


def kernel_grad_ss(t0_ind, t1_ind, S_e, S_i, seg_e, seg_i, b_type_e, b_type_i, v,
                    kernel, dt):
    """Look up fitted plasticity kernels to approximate gradients. Treat each
    synaptic activation separately.

    Parameters
    ----------
    t0, t1 : int
        time indices to define range for computing gradients
    S_e, S_i : list
        presynaptic spike times for E and I synapses
    seg_e, seg_i : ndarray
        segment locations of E and I synapses
    b_type_e, b_type_i : list
        branch types for all synapses (soma=-1, basal=0, oblique=1, apical=2)
    v : ndarray
        vector of model compartment voltages
    kernel : list
        parameters of plasticity kernel
    dt : float
        timestep

    Returns
    -------
    f_e, f_i : ndarray
        gradients (dv_soma/dw) for E and I individual synaptic inputs
    """
    t0 = t0_ind*dt
    t1 = t1_ind*dt
    Z_e = sequences.subsequence(S_e, t0, t1)
    Z_i = sequences.subsequence(S_i, t0, t1)
    Z_e, z_ind_e = sequences.rate2temp(Z_e)
    Z_i, z_ind_i = sequences.rate2temp(Z_i)
    s_e = t1 - Z_e
    s_i = t1 - Z_i
    f_e = np.zeros(len(Z_e))
    f_i = np.zeros(len(Z_i))
    if len(kernel) == 2:
        p_es, p_is = kernel
        for k, s in enumerate(s_e):
            f_e[k] += p_es(s)
        for k, s in enumerate(s_i):
            f_i[k] += p_is(s)
    else:
        p_eb, p_ea, p_ib, p_ia = kernel
        if np.size(p_eb[0]) == 1:
            for k, s in enumerate(s_e):
                if b_type_e[z_ind_e[k]] == 2:
                    f_e[k] += p_ea(s)
                else:
                    f_e[k] += p_eb(s)
            for k, s in enumerate(s_i):
                if b_type_i[z_ind_i[k]] == 2:
                    f_i[k] += p_ia(s)
                else:
                    f_i[k] += p_ib(s)
        else:
            v_dend_e = v[seg_e[z_ind_e], t1_ind]
            v_dend_i = v[seg_i[z_ind_i], t1_ind]
            for k, (s, u) in enumerate(zip(s_e, v_dend_e)):
                if b_type_e[z_ind_e[k]] == 2:
                    f_e[k] += kernels.eval_poly(s, u, p_ea)
                else:
                    f_e[k] += kernels.eval_poly(s, u, p_eb)
            for k, (s, u) in enumerate(zip(s_i, v_dend_i)):
                if b_type_i[z_ind_i[k]] == 2:
                    f_i[k] += kernels.eval_poly(s, u, p_ia)
                else:
                    f_i[k] += kernels.eval_poly(s, u, p_ib)
    return f_e, f_i


def get_grad(cell, t0_ind, t1_ind, dt, S_e, S_i, soln, stim, I_inj):
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
        input spike patterns fro E and I synapses
    soln : list
        model states [v, m, h, n, p] (output from cell.simulate)
    stim : list
        synapse indices and states [ind_e, ind_i, A_r, A_d, N_r, N_d, G_r, G_d]

    Returns
    -------
    F_e, F_i : list
        computed gradients with associated synapse indices and presynaptic
        spike times F_e = [f_e, z_ind_e, z_e]
    """
    t0 = t0_ind*dt
    t1 = t1_ind*dt
    F_e = np.zeros(len(S_e))
    F_i = np.zeros(len(S_i))
    IC_sub = cell.set_IC(soln, stim, t0_ind)
    g_na_temp, g_k_temp = cell.P['g_na'], cell.P['g_k']

    cell.P['g_na'] = 0
    cell.P['g_k'] = 0
    t_s, soln_s, stim_s = cell.simulate(t0, t1, dt, IC_sub, S_e, S_i, I_inj=I_inj)
    Z_e = sequences.subsequence(S_e, t0, t1)
    Z_i = sequences.subsequence(S_i, t0, t1)
    z_ind_e = stim_s[0]
    z_ind_i = stim_s[1]
    Z_e = pad_S(Z_e)[z_ind_e]
    Z_i = pad_S(Z_i)[z_ind_i]
    f_e, f_i = cell.grad_w(soln_s, stim_s, t_s, dt, Z_e, Z_i, z_ind_e, z_ind_i)
    cell.P['g_na'] = g_na_temp
    cell.P['g_k'] = g_k_temp

    for k in range(len(z_ind_e)):
        F_e[z_ind_e[k]] += f_e[k, -1]
    for k in range(len(z_ind_i)):
        F_i[z_ind_i[k]] += f_i[k, -1]
    return F_e, F_i


def pad_S(S0):
    l = np.max([len(s) for s in S0])
    S = np.full((len(S0), l), np.inf)
    for k, s in enumerate(S0):
        S[k, :len(s)] = s
    return S


def train(rates_e, rates_i, S_E, S_I, Labels, E, E_P, W_E, W_I, P, kernel, cmodel=False):
    """Training algorithm for pattern classification, using fitted plasticity
    kernels to approximate gradients. Learning rule is spike-time and dendritic
    voltage dependent with error feedback. Pickles updated weights and errors
    after each epoch.

    Parameters
    ----------
    rates_e, rates_i : list
        sets of rate vectors defining each pattern
    S_E, S_I : list
        sets of precisely timed event times
    Labels : list
        binary classification labels
    E : list
        total errors for each training epoch
    E_P : list
        pattern errors for each training epoch
    W_E, W_I : list
        weight vectors for each training epoch
    P : dict
        model parameters
    kernel : list
        plasticity kernel parameters

    Returns
    -------
    E : list
        updated total errors for each training epoch
    E_P : list
        updated pattern errors for each training epoch
    W_E, W_I : list
        updated weight vectors for each training epoch
    P : dict
        model parameters (can be removed -- remains from previous version)
    """

    if cmodel:
        cell = comp_model.CModel(P)
    else:
        cell = neuron_model.NModel(P)

    (
        v_init, epochs, alpha0, alpha_d, offset, t_window, jitter, dt, s,
        model_file, seed, t_on, t_off, stim_on, stim_off, r_0, r_mean, r_max,
        active_n, num_t
    ) = (
        P['v_init'], P['epochs'], P['alpha0'], P['alpha_d'], P['offset'], P['t_window'],
        P['jitter'], P['dt'], P['s'], P['model_file'], P['seed'], P['t_on'],
        P['t_off'], P['stim_on'], P['stim_off'], P['r_0'], P['r_mean'], P['r_max'],
        P['active_n'], P['num_t'],
        )

    if num_t > 0:
        sigma = jitter*s*1e-3*r_max*(stim_off - stim_on)/num_t
    else:
        sigma = jitter
    np.random.seed(seed)

    pre_syn_e = sequences.PreSyn(r_0, sigma)
    pre_syn_i = sequences.PreSyn(r_0, sigma)

    if active_n:
        sf_e = 1.0
    else:
        sf_e = 0.2

    alpha = alpha0/(1 + alpha_d*len(E))
    w_e_max = 10e-3
    w_i_max = 10e-3
    f_e_max = 3e3
    f_i_min = -3e3

    w_e = np.array(W_E[-1])
    w_i = np.array(W_I[-1])
    cell.set_weights(w_e, w_i)

    num_patterns = len(Labels)
    rand_ind = np.arange(num_patterns)
    T = max(t_off, stim_off+50)
    while (len(E) < epochs) and (np.sum(E[-10:]) != 0):
        np.random.shuffle(rand_ind)
        E_trial = 0
        for ind in rand_ind:
            label = np.array(Labels[ind])

            if label == 1:
                I_inj = 0.1*np.mean(E_P[ind][-10:])
            else:
                I_inj = 0

            S_e = [pre_syn_e.spike_train(t_on, t_off, stim_on, stim_off, s,
                    rates_e[ind][k], S_E[ind][k]) for k in range(len(rates_e[ind]))]
            S_i = [pre_syn_i.spike_train(t_on, t_off, stim_on, stim_off, s,
                    rates_i[ind][k], S_I[ind][k]) for k in range(len(rates_i[ind]))]
            S_e = pad_S(S_e)
            S_i = pad_S(S_i)

            if cmodel:
                t, soln, stim = cell.simulate(0, T, dt, v_init, S_e, S_i, I_inj,
                                              break_flag=True)
                v = soln[0]
            else:
                t, v = cell.simulate(T, dt, v_init, S_e, S_i, I_inj,
                                     break_flag=True)

            t_inds = spike_inds(v, int(offset/dt))
            t_inds = t_inds[t_inds > stim_on/dt]
            num_spikes = len(t_inds)
            if label == -1 and num_spikes > 0:
                alpha_t = alpha*np.mean(E_P[ind][-10:])
                E_P[ind].append(1)
                E_trial += 1
            elif label == -1 and num_spikes == 0:
                E_P[ind].append(0)
                continue
            elif label == 1 and num_spikes > 0:
                alpha_t = alpha*np.mean(E_P[ind][-10:])
                E_P[ind].append(0)
            else:
                E_trial += 1
                E_P[ind].append(1)
                continue
            t0_inds = t_inds - int(t_window/dt)
            t0_inds[t0_inds < 0] = 0

            for spike in range(num_spikes):

                if cmodel:
                    f_e, f_i = get_grad(cell, t0_inds[spike], t_inds[spike],
                                        dt, S_e, S_i, soln, stim, I_inj)
                else:
                    f_e, f_i = kernel_grad(t0_inds[spike], t_inds[spike], S_e,
                            S_i, cell.seg_e, cell.seg_i, cell.b_type_e,
                            cell.b_type_i, v, kernel, dt)

                f_e[f_e < 0] = 0
                f_i[f_i > 0] = 0
                f_e[f_e > f_e_max] = f_e_max
                f_i[f_i < f_i_min] = f_i_min
                delta_e = sf_e*label*alpha_t*f_e
                delta_i = label*alpha_t*f_i
                w_e += delta_e
                w_i += delta_i
                w_e[w_e < 0] = 0
                w_i[w_i < 0] = 0
                w_e[w_e > w_e_max] = w_e_max
                w_i[w_i > w_i_max] = w_i_max
            cell.set_weights(w_e, w_i)
        E.append(E_trial)
        alpha = alpha0/(1 + alpha_d*len(E))
        W_E.append(np.array(w_e))
        W_I.append(np.array(w_i))
        if len(E) % 10 == 0:
            print('epoch: ' + str(len(E)) + '  error: ' + str(np.nanmean(E[-10:])/num_patterns))
            # uncomment below to checkpoint on long simulations
            # pickle.dump([rates_e, rates_i, S_E, S_I, Labels, E, E_P, W_E, W_I, P,
            #             kernel], open(model_file, 'wb'))
    return E, E_P, W_E, W_I, P


def train_online(rates_e, rates_i, S_E, S_I, Labels, E, E_P, W_E, W_I, P):
    """Online training algorithm for pattern classification, using fitted plasticity
    kernels to approximate gradients. Learning rule is spike-time and dendritic
    voltage dependent with error feedback. Pickles updated weights and errors
    after each epoch.

    Parameters
    ----------
    rates_e, rates_i : list
        sets of rate vectors defining each pattern
    S_E, S_I : list
        sets of precisely timed event times
    Labels : list
        binary classification labels
    E : list
        total errors for each training epoch
    E_P : list
        pattern errors for each training epoch
    W_E, W_I : list
        weight vectors for each training epoch
    P : dict
        model parameters
    kernel : list
        plasticity kernel parameters

    Returns
    -------
    E : list
        updated total errors for each training epoch
    E_P : list
        updated pattern errors for each training epoch
    W_E, W_I : list
        updated weight vectors for each training epoch
    """

    cell = neuron_model.NModelOnline(P)
    (
        v_init, epochs, offset, alpha0, alpha_d, jitter, dt, s, model_file,
        seed, t_on, t_off, stim_on, stim_off, r_0, r_mean, r_max, num_t, active_n
    ) = (
        P['v_init'], P['epochs'], P['offset'], P['alpha0'], P['alpha_d'],
        P['jitter'], P['dt'], P['s'], P['model_file'], P['seed'], P['t_on'],
        P['t_off'], P['stim_on'], P['stim_off'], P['r_0'], P['r_mean'],
        P['r_max'], P['num_t'], P['active_n']
    )
    np.random.seed(seed)

    if num_t > 0:
        sigma = jitter*s*1e-3*r_max*(P['period'])/num_t
    else:
        sigma = jitter

    pre_syn = sequences.PreSyn(r_0, sigma)
    w_e = np.array(W_E[-1])
    w_i = np.array(W_I[-1])
    cell.set_weights(w_e, w_i)
    cell.P['alpha'] = alpha0/(1 + alpha_d*len(E))
    T = max(t_off, stim_off+50)
    num_patterns = len(Labels)
    rand_ind = np.arange(num_patterns)
    while (len(E) < epochs) and (np.sum(E[-10:]) != 0):
        np.random.shuffle(rand_ind)
        E_trial = 0
        for ind in rand_ind:
            S_e = [pre_syn.spike_train(t_on, t_off, stim_on, stim_off, s,
                    rates_e[ind][k], S_E[ind][k]) for k in range(len(rates_e[ind]))]
            S_i = [pre_syn.spike_train(t_on, t_off, stim_on, stim_off, s,
                    rates_i[ind][k], S_I[ind][k]) for k in range(len(rates_i[ind]))]

            t, v = cell.train(T, dt, v_init, S_e, S_i, Labels[ind])

            t_inds = spike_inds(v, int(offset / dt))
            t_inds = t_inds[t_inds > (stim_on+P['delay']) / dt]
            num_spikes = len(t_inds)
            if Labels[ind] == -1 and num_spikes > 0:
                E_trial += 1
            elif Labels[ind] == -1 and num_spikes == 0:
                pass
            elif Labels[ind] == 1 and num_spikes/(T-stim_on-P['delay'])*1e3 >= P['s_th']:
                pass
            else:
                E_trial += 1
            E_P[ind].append(num_spikes)

        E.append(E_trial)
        cell.P['alpha'] = alpha0/(1 + alpha_d*len(E))
        W_E.append(np.array(cell.w_e))
        W_I.append(np.array(cell.w_i))
        if len(E) % 10 == 0:
            print('epoch: ' + str(len(E)) + '  error: ' + str(np.nanmean(E[-10:])/num_patterns))
            pickle.dump([rates_e, rates_i, S_E, S_I, Labels, E, E_P, W_E, W_I, P], open(model_file, 'wb'))
    return E, E_P, W_E, W_I
