#!/usr/bin/env python3

#modified from https://neuron.yale.edu/neuron/static/docs/neuronpython/ballandstick3.html

import numpy as np
from neuron import h
from matplotlib import pyplot

def set_recording_vectors(cell):
	""" set recording vectors
	:param cell: cell to record from
	:return: soma and dendrite voltage and time vectors as tuple
	"""
	v_vec = h.Vector()
	t_vec = h.Vector()
	v_vec.record(cell.soma(0.5)._ref_v)
	t_vec.record(h._ref_t)
	return v_vec, t_vec
	
def normalise(cell, v_vec, t_vec, e_exc):
	
	tau_m = cell.soma.cm / cell.soma.g_pas * 1e-3
	t_vec = np.array(t_vec) / tau_m
	v_vec = (np.array(v_vec) - cell.soma.e_pas) / (e_exc - cell.soma.e_pas)
	return v_vec, t_vec

def show_output(v_vec, t_vec, color_str, new_fig=True):
	"""
	:param new_fig: flag to create new figure instead of drawing over previous
	"""
	if new_fig:
		pyplot.figure(figsize=(8,4))
	v_plot = pyplot.plot(t_vec, v_vec, color=color_str)
	pyplot.xlabel('$t/\\tau$')
	pyplot.ylabel('$(V-E_r)/(E_e-E_r)$')
	return v_plot