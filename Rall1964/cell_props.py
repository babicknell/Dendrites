#!/usr/bin/env python3

#modified from https://neuron.yale.edu/neuron/static/docs/neuronpython/ballandstick3.html

from neuron import h
import numpy as np

class BallStick():
	
	def __init__(self):
		self.create_sections()
		self.build_topology()
		self.build_subsets()
		self.define_geometry()
		self.define_biophysics()
		
	def create_sections(self):
		self.soma = h.Section(name='soma', cell=self)
		self.dend = h.Section(name='dend', cell=self)
	
	def build_topology(self):
		self.dend.connect(self.soma(1))
	
	def build_subsets(self):
		self.all = h.SectionList()
		self.all.wholetree(sec=self.soma)
	
	def define_geometry(self):
		self.soma.L = 100
		self.soma.diam = 1
		self.dend.L = 900
		self.dend.diam = 1
		self.dend.nseg = 9
		h.define_shape()
	#
	def define_biophysics(self):
		for sec in self.all:
			sec.Ra = 50
			sec.cm = 1
			sec.insert('pas')
			sec.g_pas = 0.0002
			sec.e_pas = -70
	
def attach_gpulse(cell, onset, dur, gmax, e, loc):
	
	syn = h.gpulse(loc)
	syn.onset = onset
	syn.dur = dur
	syn.gmax = gmax
	syn.e = e
	return syn