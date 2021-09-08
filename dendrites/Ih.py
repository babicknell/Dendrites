#!/usr/bin/env python3
"""
Functions and derivatives for HCN channel model. Hard-coded model
parameters are from Kole, Hallermann and Stuart 2006.
"""

import numpy as np


def alpha_m(v):
    return 0.00643*(v + 154.9)/(np.exp((v + 154.9)/11.9) - 1)


def d_alpha_m(v):
    return 0.00643*(np.exp((v + 154.9)/11.9) - 1 - 1/11.9*(v + 154.9)*
                    np.exp((v + 154.9)/11.9))/(np.exp((v + 154.9)/11.9) - 1)**2


def beta_m(v):
    return 0.193*np.exp(v/33.1)


def d_beta_m(v):
    return 0.193/33.1*np.exp(v/33.1)


def m_inf(v):
    return alpha_m(v)/(alpha_m(v) + beta_m(v))


def tau_m(v):
    return 1/(alpha_m(v) + beta_m(v))
