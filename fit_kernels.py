"""
Fit polynomials to STA data and interpolate over unvisited areas of the
parameter space. Requires the output of 'process_sta'.
"""

import numpy as np
import pickle

from matplotlib import pyplot

from dendrites import kernels

results = './outputs/sta/sta_act1_proc'
kernel_fit = './outputs/kernel_fit_act1'
save_fit = True
cv = False  # fits on 75% of data when True.

t = np.arange(0, 101)  # time and voltage vectors for binning
v = np.arange(-80, 1)


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


def get_2d_avg(t, v, X):
    """ Compute 2d spike-triggered average of dv_soma/dw as a function of
     synaptic activation time and local dendritic voltage.

    Parameters
    ----------
    t : ndarray
        vector of time bins
    v : ndarray
        vector of voltage bins
    X : ndarray
        simulation data (see 'process_sta')
    Returns
    -------
    Z_m : ndarray
        spike-triggered average
    """
    Z_m = np.zeros((len(v), len(t)))
    Z = []
    for tt in t:
        ind_t = np.where((-X[:, 2] >= tt) & (-X[:, 2] < tt+1))[0]
        Z_v = []
        for vv in v:
            ind_v = np.where((X[ind_t, 1] >= vv) & (X[ind_t, 1] < vv+1))[0]
            Z_v.append(X[ind_t, 3][ind_v])
        Z.append(Z_v)

    for i in range(len(v)):
        for j in range(len(t)):
            Z_m[i, j] = mean(Z[j][i])
    return Z_m


def fit(t, v, Z, deg_t, deg_v):
    """Polynomial fit to 2d STA.

    Parameters
    ----------
    t : ndarray
        vector of time bins
    v : ndarray
        vector of voltage bins
    Z : ndarray
        simulation data (see 'process_sta')
    deg_t, deg_v : int
        degrees of polynomial for fitting along time and voltage axes
    Returns
    -------
    Z_m : ndarray
        spike-triggered average
    p : list
        coefficients of polynomial fit
    z : ndarray
        polynomial evaluated on t x v mesh
    """
    Z_m = get_2d_avg(t, v, Z)
    Z_mz = np.array(Z_m)
    pad = 10  # v dimension padding
    pad2 = 1  # t dimension padding
    v_pad = np.hstack((np.arange(v[0]-pad, v[0]), v, np.arange(v[-1]+1, v[-1]+pad+1)))
    Z_mz = np.vstack((np.zeros((pad, len(t))), Z_mz, np.zeros((pad, len(t)))))
    t_pad = np.hstack((np.arange(t[0]-pad2, t[0]), t))
    Z_mz = np.hstack((np.zeros((len(v_pad), pad2)), Z_mz))
    p = kernels.fit_poly(t_pad, v_pad, Z_mz, len(t_pad), deg_t, deg_v)
    z = kernels.eval_poly_mesh(t, v, p)
    return Z_m, p, z



E_b, E_a, I_b, I_a = pickle.load(open(results, 'rb'))


if cv:
    E_b_m, p_eb, z_eb = fit(t, v, E_b[:int(E_b.shape[0]*3/4), :], 8, 8)
    E_a_m, p_ea, z_ea = fit(t, v, E_a[:int(E_a.shape[0]*3/4), :], 8, 8)
    I_b_m, p_ib, z_ib = fit(t, v, I_b[:int(I_b.shape[0]*3/4), :], 8, 8)
    I_a_m, p_ia, z_ia = fit(t, v, I_a[:int(I_a.shape[0]*3/4), :], 8, 8)
else:
    E_b_m, p_eb, z_eb = fit(t, v, E_b, 8, 8)
    E_a_m, p_ea, z_ea = fit(t, v, E_a, 8, 8)
    I_b_m, p_ib, z_ib = fit(t, v, I_b, 8, 8)
    I_a_m, p_ia, z_ia = fit(t, v, I_a, 8, 8)

if save_fit:
    data = [E_b_m,  E_a_m, I_b_m, I_a_m]
    params = [p_eb, p_ea, p_ib, p_ia]
    pickle.dump([data, params], open(kernel_fit, 'wb'))
