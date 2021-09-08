#!/usr/bin/env python3
"""
Functions and derivatives for somatic Na and K channel models. Hard-coded model
parameters from are Pospichil et al 2008.
"""

import numpy as np

v_th = -60  # Traub and Miles threshold parameter
t_max = 0.2e3  # Slow K+ channel adaptation time constant


def alpha_m(v):
	return 0.32*(v - v_th - 13.)/(1. - np.exp(-(v - v_th - 13.)/4.))


def d_alpha_m(v):
	return 0.32*(1. - np.exp(-(v - v_th - 13.)/4.) - 1/4.*(v - v_th - 13.)*
			np.exp(-(v - v_th - 13.)/4.))/(1. - np.exp(-(v - v_th - 13.)/4.))**2


def beta_m(v):
	return -0.28*(v - v_th - 40.)/(1. - np.exp((v - v_th - 40.)/5.))


def d_beta_m(v):
	return -0.28*(1. - np.exp((v - v_th - 40.)/5.) + 1/5.*(v - v_th - 40.)*
			np.exp((v - v_th - 40.)/5.))/(1. - np.exp((v - v_th - 40.)/5.))**2


def alpha_h(v):
	return 0.128*np.exp(-(v - v_th - 17.)/18.)


def d_alpha_h(v):
	return -0.128/18.*np.exp(-(v - v_th - 17.)/18.)


def beta_h(v):
	return 4./(1. + np.exp(-(v - v_th - 40.)/5.))


def d_beta_h(v):
	return (4./5.)*np.exp(-(v - v_th - 40.)/5.)/(1 + np.exp(-(v - v_th - 40.)/5.))**2


def alpha_n(v):
	return 0.032*(v - v_th - 15.)/(1. - np.exp(-(v - v_th - 15.)/5.))


def d_alpha_n(v):
	return 0.032*(1. - np.exp(-(v - v_th - 15.)/5.) - 1/5.*(v - v_th - 15.)*
			np.exp(-(v - v_th - 15.)/5.))/(1. - np.exp(-(v - v_th - 15.)/5.))**2


def beta_n(v):
	return 0.5*np.exp(-(v - v_th - 10.)/40.)


def d_beta_n(v):
	return -0.5/40.*np.exp(-(v - v_th - 10.)/40.)


def m_inf(v):
	return alpha_m(v)/(alpha_m(v) + beta_m(v))


def tau_m(v):
	return 1/(alpha_m(v) + beta_m(v))


def n_inf(v):
	return alpha_n(v)/(alpha_n(v) + beta_n(v))


def tau_n(v):
	return 1/(alpha_n(v) + beta_n(v))


def h_inf(v):
	return alpha_h(v)/(alpha_h(v) + beta_h(v))


def tau_h(v):
	return 1/(alpha_h(v) + beta_h(v))


def p_inf(v):
	return 1./(1. + np.exp(-(v + 35.)/10.))


def tau_p(v):
	return t_max/(3.3*np.exp((v + 35.)/20.) + np.exp(-(v + 35.)/20.))


def alpha_p(v):
	return p_inf(v)/tau_p(v)


def beta_p(v):
	return (1 - p_inf(v))/tau_p(v)


def d_p_inf(v):
	return 1./10.*np.exp(-(v + 35.)/10.)/(1. + np.exp(-(v + 35.)/10.))**2


def d_tau_p(v):
	return -t_max*(3.3/20.*np.exp((v + 35.)/20.) - 1/20.*np.exp(-(v + 35.)/20.)
					)/(3.3*np.exp((v + 35.)/20.) + np.exp(-(v + 35.)/20.))**2


def d_alpha_p(v):
	return (d_p_inf(v)*tau_p(v) - p_inf(v)*d_tau_p(v))/tau_p(v)**2


def d_beta_p(v):
	return (-d_p_inf(v)*tau_p(v) - (1 - p_inf(v))*d_tau_p(v))/tau_p(v)**2
