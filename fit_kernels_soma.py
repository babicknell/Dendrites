"""
Fit polynomials to STA data and interpolate over unvisited areas of the
parameter space (point neuron version). Requires the output of 'process_sta'.
"""

import numpy as np
import pickle


results = './cluster_outputs/sta/sta_soma1_proc'
kernel_fit = './outputs/kernel_fit_soma1'
save_fit = True
cv = False  # fits on 75% of data when True.
t = np.arange(0, 101)  # time vector for binning
model = 'pn'


def mean(x):
    if len(x) > 0:
        m = np.mean(x)
    else:
        m = np.nan
    return m


def std(x):
    if len(x) > 0:
        sd = np.std(x)
    else:
        sd = np.nan
    return sd


def get_avg(t, X):
    """ Compute spike-triggered average of dv_soma/dw as a function of
     synaptic activation time.

    Parameters
    ----------
    t : ndarray
        vector of time bins
    X : ndarray
        simulation data (see 'process_sta')
    Returns
    -------
    Y_m : ndarray
        spike-triggered average
    """
    Y = []
    for tt in t:
        ind_t = np.where((-X[:, 2] >= tt) & (-X[:, 2] < tt+1))[0]
        Y.append(X[ind_t, 3])
    Y_m = np.array([mean(y) for y in Y])
    return Y_m


def fit(t, Z, deg_t):
    """Polynomial fit to STA.

     Parameters
     ----------
     t : ndarray
         vector of time bins
     Z : ndarray
         simulation data (see 'process_sta')
     deg_t : int
         degree of polynomial for fitting
     Returns
     -------
     Z_m : ndarray
         spike-triggered average
     pt : list
         coefficients of polynomial fit
     z : ndarray
         polynomial evaluated on t
     """
    Z_m = get_avg(t, Z)
    Z_mz = np.array(Z_m)
    pad = 1
    t_pad = np.hstack((np.arange(t[0]-pad, t[0]), t))
    Z_mz = np.hstack((np.zeros((pad)), Z_mz))
    pt = np.poly1d(np.polyfit(t_pad, Z_mz, deg_t))
    z = pt(t)
    return Z_m, pt, z


if model == 'pn':
    E_s, _, I_s, _ = pickle.load(open(results, 'rb'))
    if cv:
        E_s_m, p_es, z_es = fit(t, E_s[:int(E_s.shape[0]*3/4), :], 10)
        I_s_m, p_is, z_is = fit(t, I_s[:int(I_s.shape[0]*3/4), :], 10)
    else:
        E_s_m, p_es, z_es = fit(t, E_s, 10)
        I_s_m, p_is, z_is = fit(t, I_s, 10)
    if save_fit:
        data = [E_s_m, I_s_m]
        params = [p_es, p_is]
        pickle.dump([data, params], open(kernel_fit, 'wb'))
else:
    E_b, E_a, I_b, I_a = pickle.load(open(results, 'rb'))
    E_b_m, p_eb, z_eb = fit(t, E_b, 10)
    E_a_m, p_ea, z_ea = fit(t, E_a, 10)
    I_b_m, p_ib, z_ib = fit(t, I_b, 10)
    I_a_m, p_ia, z_ia = fit(t, I_a, 10)
    if save_fit:
        data = [E_b_m, E_a_m, I_b_m, I_a_m]
        params = [p_eb, p_ea, p_ib, p_ia]
        pickle.dump([data, params], open(kernel_fit, 'wb'))
