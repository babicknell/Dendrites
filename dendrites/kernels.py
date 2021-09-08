#!/usr/bin/env python3
"""
Functions for fitting and evaluating optimal plasticity kernels.
"""

import numpy as np


def fit_poly(t, v, Z, t_max, deg_t, deg_v):
    """Fit 2d spike-triggered average with polynomial.

    Parameters
    ----------
    t : ndarray
        time vector
    v : ndarray
        voltage vector
    Z : ndarray
        spike-triggered average, Z = Z(v, t)
    t_max : int
        max time index to fit up to
    deg_t, deg_v : int
        degree of polynomials in time and voltage variables

    Returns
    -------
    pt : list
        polynomial fit as a function of time for each voltage slice
    """

    v_ind = [np.where(~np.isnan(Z[:, k]))[0] for k in range(t_max)]
    pv = np.array([np.polyfit(v[v_ind[k]], Z[v_ind[k], k], deg_v) for k in range(t_max)])
    pt = [np.poly1d(np.polyfit(t, pv[:, k], deg_t)) for k in range(deg_v+1)]
    return pt


def eval_poly(t, v, pt):
    """Evaluate fitted kernel for a sequence of points

    Parameters
    ----------
    t : ndarray
        sequences of presynaptic spike times
    v : ndarray
        sequence of voltages
    pt : list
        fitted polynomial for plasticity kernel

    Returns
    -------
    z :  ndarray
        values of kernels evaluated at t and v
    """

    z = 0
    for k, p in enumerate(pt):
        z += p(t)*v**(len(pt)-1-k)
    z = np.array(z)
    return z


def eval_poly_mesh(t, v, pt):
    """Evaluate kernel for a mesh of points

    arameters
    ----------
    t : ndarray
        mesh of presynaptic spike times
    v : ndarray
        mesh of voltages
    pt : list
        fitted polynomial for plasticity kernel

    Returns
    -------
    z :  ndarray
        values of kernels evaluated at (t, v)
    """

    z = []
    for u in v:
        z_k = 0
        for k, p in enumerate(pt):
            z_k += p(t)*u**(len(pt)-1-k)
        z.append(z_k)
    z = np.array(z)
    return z
