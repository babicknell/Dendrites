"""
Functions for plotting input spike rasters.
"""

from matplotlib import pyplot
import numpy as np


def plot_simple_raster(S, S_i, T):
    """ Plot raster

    Parameters
    ----------
    S : list
        spike times for a set of synapses
    T : int
        maximum time (ms)
    """

    fig, ax = pyplot.subplots(figsize=(5, 3.5))
    for k, t_k in enumerate(S):
        ax.plot(t_k, len(t_k)*[k+1], 'ko', markersize=3)
    for k, t_k in enumerate(S_i):
        ax.plot(t_k, len(t_k)*[k+1], 'mo', markersize=3)
    ax.set_xlim([0, T])
    ax.set_ylim([0, len(S)+2])
    ax.set_yticks([1, len(S)])
    ax.set_xlabel('time (ms)', fontsize=14)
    ax.set_ylabel('synapse', fontsize=14)
    ax.tick_params(labelsize=12)
    pyplot.tight_layout()


def raster_params(cell):
    """ Spatial positions and boundaries for spike raster."""
    P = cell.P
    soma, basal, oblique, apical = P['soma'], P['basal'], P['oblique'], P['apical']
    w_sec = cell.sec_e[0, :]
    w_soma = [k for k in range(P['N_e']) if w_sec[k] in soma]
    w_basal = [k for k in range(P['N_e']) if w_sec[k] in basal]
    w_oblique = [k for k in range(P['N_e']) if w_sec[k] in oblique]
    w_apical = [k for k in range(P['N_e']) if w_sec[k] in apical]
    w_sorted = np.array(w_soma + w_basal + w_oblique + w_apical)
    index = [np.where(w_sorted == k)[0][0] for k in range(P['N_e'])]
    boundaries = []
    for b in [oblique[-1], apical[-1]]:
        position = np.where(cell.sec_e[0, :] == b)[0]
        if len(position) > 0:
            position = position[-1]
        else:
            position = 0
        boundaries.append(index[position])
    i_positions = []
    for k in range(P['N_i']):
        i_positions.append(np.where(cell.seg_e == cell.seg_i[k])[0][0])
    return i_positions, index, boundaries


def plot_grad_example(f_e, f_i, e_ind, i_ind, s_e, s_i, t_window, i_positions,
                    index, boundaries):
    """ Plot example raster with scaled markers to represent dv_soma/dw (for 800
    excitatory and 200 inhibitory synapses)

    Parameters
    ----------
    f_e : ndarray
        excitatory input scale factors
    f_e : ndarray
        inhibitory input scale factors
    e_ind : list
        indices of E synapses corresponding to f_e
    i_ind : list
        indices of I synapses corresponding to f_i
    s_e : ndarray
        excitatory presynaptic spike times
    s_i : ndarray
        inhibitory presynaptic spike times
    t_window :  int
        time window for plotting
    i_positions : list
        index of closest E synapse to each I synapse
    index : list
        E synapse indices sorted from basal to apical dendrites
    boundaries : list
        indices separating basal and apical domains
    """

    max_marker = 12
    f_e = np.array(f_e)
    f_i = np.array(f_i)
    f_e[f_e < 0] = 0
    f_i[f_i > 0] = 0
    f_i *= -1
    fig, ax1 = pyplot.subplots(figsize=(5, 3.25))
    for k, t_k in enumerate(s_e):
            ax1.plot(t_k, index[e_ind[k]], 'ko', markersize=1+max_marker*f_e[k], markeredgewidth=0)
    ax2 = ax1.twinx()
    for k, t_k in enumerate(s_i):
            ax1.plot(t_k, index[i_positions[i_ind[k]]], 'mo', markersize=1+max_marker*f_i[k], markeredgewidth=0)

    pyplot.plot([-t_window, 0], [0, 0], 'b-', linewidth=1)
    for b in boundaries:
        pyplot.plot([-t_window, 0], [b+0.5, b+0.5], 'b-', linewidth=1)
    ax1.set_xlim([-t_window, 0])
    ax1.set_ylim([-20, 820])
    ax1.set_xlabel('time before spike (ms)', fontsize=14)
    ax1.set_ylabel('excitatory synapse', fontsize=14)
    ax2.set_ylabel('inhibitory synapse', fontsize=14, color='m')
    ax2.set_ylim([-20, 820])
    ax1.set_xticks(np.arange(-100, 20, 20))
    ax1.set_xticklabels(np.arange(100, -20, -20))
    ax1.set_yticks([1, 200, 400, 600, 800])
    ax1.set_yticklabels([1, 200, 400, 600, 800])
    ax2.set_yticks([1, 200, 400, 600, 800])
    ax2.set_yticklabels([1, 50, 100, 150, 200], color='m')
    ax1.tick_params(labelsize=13)
    ax2.tick_params(labelsize=13)
    pyplot.tight_layout()
