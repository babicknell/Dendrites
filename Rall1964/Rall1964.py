#!/usr/bin/env python3

import numpy as np
from neuron import h, gui
from matplotlib import pyplot
import cell_props as cp
import plotting

# create three ball and stick models to stimulate in OUT and IN directions, and ALL at once
cell_out = cp.BallStick()
cell_in = cp.BallStick()
cell_all = cp.BallStick()

#membrane time constant and electrotonic length
tau_m = cell_out.soma.cm / cell_out.soma.g_pas * 1e-3
lam = 50 * (cell_out.dend.diam / (cell_out.dend.g_pas * cell_out.dend.Ra))**0.5

# Position 8 synapses in the first 8/9 dendrite segments. Delay between activation of
# adjacent pairs of synapses is tau_m/4
num_syn = 8
syn_loc = np.arange(1/9, 1, 1/9) - 1/18
syn_delay = 0.25*tau_m
syn_onset = np.arange(0, 4*syn_delay, syn_delay)
syn_onset = np.array([syn_onset, syn_onset]).T.reshape(1,8)
syn_onset = syn_onset[0]
syn_gmax = cell_out.dend.g_pas * cell_out.dend(0.5).area()/100
syn_e = 0

# Attach synapses and simulate in OUT direction
syns = []
for i in range(num_syn):
	syn = cp.attach_gpulse(cell_out, syn_onset[i], syn_delay, syn_gmax, syn_e,
	cell_out.dend(syn_loc[i]))
	syns.append(syn)
	
v_vec, t_vec = plotting.set_recording_vectors(cell_out)

def simulate(tstop=12.5, v_init=-70):
	h.v_init = v_init
	h.tstop = tstop
	h.run()
	
simulate()
v_vec, t_vec = plotting.normalise(cell_out, v_vec, t_vec, syn_e)
v_plot_out = plotting.show_output(v_vec, t_vec, 'black', True)

# Attach synapses and simulate in IN direction
syns = []
for i in range(num_syn):
	syn = cp.attach_gpulse(cell_in, syn_onset[7 - i], syn_delay, syn_gmax, syn_e,
	cell_in.dend(syn_loc[i]))
	syns.append(syn)

v_vec, t_vec = plotting.set_recording_vectors(cell_in)

simulate()
v_vec, t_vec = plotting.normalise(cell_in, v_vec, t_vec, syn_e)
v_plot_in = plotting.show_output(v_vec, t_vec, 'red', False)

# activate all synapses at once for tau_m and 1/4 the max conductance
syns = []
for i in range(num_syn):
	syn = cp.attach_gpulse(cell_all, syn_onset[0], 4*syn_delay, syn_gmax/4, syn_e,
	cell_all.dend(syn_loc[i]))
	syns.append(syn)

v_vec, t_vec = plotting.set_recording_vectors(cell_all)

simulate()
v_vec, t_vec = plotting.normalise(cell_all, v_vec, t_vec, syn_e)
v_plot_all = plotting.show_output(v_vec, t_vec, 'blue', False)

#Plot all simulations. t and V have been normalised by tau_m and the resting and 
#excitatory conductances
pyplot.legend(v_plot_out + v_plot_in +v_plot_all, ['OUT', 'IN', 'ALL'])
pyplot.axis([0, 2.5, 0, 0.2])
pyplot.xticks([0, 0.5, 1, 1.5, 2])
pyplot.yticks([0, 0.05, 0.1, 0.15, 0.2])
pyplot.show()