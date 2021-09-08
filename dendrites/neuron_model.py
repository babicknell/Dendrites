#!/usr/bin/env python3
"""
Create NEURON model and associated functions.
"""

import numpy as np

from neuron import h

from dendrites import morphology
from dendrites import training

h.load_file("stdrun.hoc")


class NModel:
    """Neuron model object for simulation of dendritic computation and learning.
    Designed to construct identical model to comp_model.CModel (hence the
    non-conventional approach to processing the morphology and creating sections).

    Parameters
    ----------
        P : dict
            model and simulation parameters

    Attributes (additional to those accessible via neuron.h)
    ----------
        P : dict
            model and simulation paramaters
        a_s : ndarray
            segment radii
        sec_e, sec_i : ndarray
            synapse locations
        seg_e, seg_i : ndarray
            synapse segment numbers
        seg_list : ndarray
            sections and positions for each segment
        b_type_e, b_type_i : list
            branch types (basal/apical/soma) for all synapses
        tree : list
            section objects (tree[0]=soma)
        AMPA, NMDA, GABA : list
            AMPA, NMDA, GABA objects
        w_1, w_2, w_3 : float
            AMPA, NMDA, GABA weights
        r_na : float
            NMDA/AMPA ratio
    """
    def __init__(self, P):
        h.celsius = 36
        self.P = P
        A, nseg, L, a, sec_points, self.sec_e, self.sec_i, self.b_type_e, self.b_type_i = \
        self.define_morphology()
        _, a_s, _ = morphology.seg_geometry(L, a, nseg)
        self.tree, self.seg_list, self.seg_e, self.seg_i = \
            self.create_sections(A, L, a_s, nseg, sec_points)
        self.build_tree(A)
        self.define_biophysics()
        self.insert_active()
        if P['active_d']:
            self.insert_active_dend()
        self.AMPA = self.attach_ampa(self.sec_e)
        self.GABA = self.attach_gaba(self.sec_i)
        if P['active_n']:
            self.NMDA = self.attach_nmda(self.sec_e)
        else:
            self.NMDA = self.attach_pas_nmda(self.sec_e)
        self.w_1 = self.sec_e.shape[1]*[P['g_max_A']]
        self.w_2 = self.sec_e.shape[1]*[P['g_max_N']]
        self.w_3 = self.sec_i.shape[1]*[P['g_max_G']]
        self.r_na = P['r_na']
        self.set_weights(np.array(self.w_1)+np.array(self.w_2), np.array(self.w_3))

    def Section(self, name):
        h("create " + name)
        return h.__getattribute__(name)

    def define_morphology(self):
        """ Create adjacency matrix, lists of segment numbers and dimensions,
        and synapse locations.

        Returns
        -------
        A : ndarray
            adjacency matrix for dendritic sections
        nseg : ndarray
            number of segments in each section
        L : ndarray
            section lengths
        a : ndarray
            section radii
        sec_points : list
            section coordinates
        sec_e, sec_i :  ndarray
            locations of excitatory and inhibitory synapses
        b_type_e, b_type_i : list
            branch types (basal/apical/soma) for all synapses
        """
        P = self.P
        N_e, N_i, l_seg, locs_e, locs_i, branch_ids = \
        (P['N_e'], P['N_i'], P['l_seg'], P['locs_e'], P['locs_i'],
        [P['basal'], P['oblique'], P['apical']])
        A, L, a, sec_points, secs, *_ = morphology.reconstruction(P['tree'])
        nseg = np.array(len(L)*[1])
        dseg = L[secs]//(l_seg*1e-4)+1
        dseg[dseg == 1] = 2
        nseg[secs] = dseg
        sec_e = morphology.synapse_locations_rand(locs_e, N_e, nseg[locs_e], 0)
        sec_i = morphology.synapse_locations_rand(locs_i, N_i, nseg[locs_i], 0)
        b_type_e = morphology.branch_type(sec_e, branch_ids)
        b_type_i = morphology.branch_type(sec_i, branch_ids)
        return A, nseg, L, a, sec_points, sec_e, sec_i, b_type_e, b_type_i

    def create_sections(self, A, L, a_s, nseg, sec_points):
        """Create sections from adjacency matrix A.

        Parameters
        ----------
        A : ndarray
            adjacency matrix for dendritic sections
        L : ndarray
            section lengths
        a_s : ndarray
            segment radii
        nseg : ndarray
            number of segments in each section
        sec_points : list
            section coordinates

        Returns
        -------
        seg_list : ndarray
            sections and positions for each segment
        seg_e, seg_i : ndarray
            synapse segment numbers
        tree : list
            section objects (tree[0]=soma)

        """
        soma = self.Section('soma')
        soma.L = L[0]*1e4			#(um)
        tree = [soma]
        for k in range(1, A.shape[0]):
            sec = self.Section('dend_%d' % k)
            sec.L = L[k]*1e4		#(um)
            sec.nseg = nseg[k]
            tree.append(sec)
        for k, branch in enumerate(tree):
            h.pt3dclear(sec=branch)
            for pt in sec_points[k]:
                h.pt3dadd(pt[0], pt[1], pt[2], 2*pt[3], sec=branch)
        j = 0
        for branch in tree:
            for seg in branch:
                seg.diam = 2*a_s[j]*1e4
                j += 1
        soma.L = L[0]*1e4

        j = 0
        seg_list = []
        for num, sec in enumerate(tree):
            for k in range(sec.nseg):
                seg_list.append([num, (k+1/2)/sec.nseg, j])
                j += 1
        seg_list = np.array(seg_list)

        seg_e = np.array([np.where(np.sum(np.abs(seg_list[:, :2]-sec), 1)<1e-15)[0][0]
                        for sec in self.sec_e.T])
        seg_i = np.array([np.where(np.sum(np.abs(seg_list[:, :2]-sec), 1)<1e-15)[0][0]
                        for sec in self.sec_i.T])
        return tree, seg_list, seg_e, seg_i

    def build_tree(self, A):
        """Connect sections to form tree (tree[0]=soma)

        Parameters
        ----------
        A : ndarray
            adjacency matrix
        """
        if A.shape[0] > 1:
            for i in range(A.shape[0]):
                for j in range(i, A.shape[0]):
                    if A[i, j] == 1:
                        self.tree[j].connect(self.tree[i](1))

    def define_biophysics(self):
        """Set biophysical parameters with unit conversions."""
        if 'c_m_d' not in self.P:
            self.P['c_m_d'] = self.P['c_m']
        for sec in self.tree:
            sec.Ra = self.P['R_a']*1e3		#(ohm cm)
            sec.cm = self.P['c_m_d']
            sec.insert('pas')
            sec.g_pas = self.P['c_m_d']/self.P['tau_m']*1e-3	      #(S/cm^2)
            sec.e_pas = self.P['E_r']
        self.tree[0].cm = self.P['c_m']
        self.tree[0].g_pas = self.P['c_m']/self.P['tau_m']*1e-3

    def attach_ampa(self, E):
        """Attach double exponential AMPA synapses.

        Parameters
        ----------
        E : ndarray
            synapse locations [sec, loc]

        Returns
        -------
        AMPA : list
            synapse objects
        """
        AMPA = []
        for k in range(E.shape[1]):
                syn = h.Exp2Syn(self.tree[int(E[0, k])](E[1, k]))
                syn.tau1 = self.P['tauA'][0]
                syn.tau2 = self.P['tauA'][1]
                syn.e = self.P['E_e']
                AMPA.append(syn)
        return AMPA

    def attach_nmda(self, E):
        """Attach double exponential NMDA synapses with sigmoid voltage
        dependence.

        Parameters
        ----------
        E : ndarray
            synapse locations [sec, loc]

        Returns
        -------
        NMDA : list
            synapse objects
        """
        NMDA = []
        for k in range(E.shape[1]):
                syn = h.Exp2Syn_NMDA(self.tree[int(E[0, k])](E[1, k]))
                syn.tau1 = self.P['tauN'][0]
                syn.tau2 = self.P['tauN'][1]
                syn.e = self.P['E_e']
                NMDA.append(syn)
        return NMDA

    def attach_gaba(self, I):
        """Attach double exponential GABA synapses.

        Parameters
        ----------
        I : ndarray
            synapse locations [sec, loc]

        Returns
        -------
        GABA : list
            synapse objects
        """
        GABA = []
        for k in range(I.shape[1]):
                syn = h.Exp2Syn(self.tree[int(I[0, k])](I[1, k]))
                syn.tau1 = self.P['tauG'][0]
                syn.tau2 = self.P['tauG'][1]
                syn.e = self.P['E_i']
                GABA.append(syn)
        return GABA

    def attach_pas_nmda(self, E):
        """Attach double exponential NMDA synapses without voltage dependence.

        Parameters
        ----------
        E : ndarray
            synapse locations [sec, loc]

        Returns
        -------
        NMDA : list
            synapse objects
        """
        NMDA = []
        for k in range(E.shape[1]):
                syn = h.Exp2Syn(self.tree[int(E[0, k])](E[1, k]))
                syn.tau1 = self.P['tauN'][0]
                syn.tau2 = self.P['tauN'][1]
                syn.e = self.P['E_e']
                NMDA.append(syn)
        return NMDA

    def insert_active(self):
        """Insert Na and K (fast and slow) channels at soma."""
        self.tree[0].insert('hh2')
        self.tree[0].gnabar_hh2 = self.P['g_na']*1e-3  # S/cm^2
        self.tree[0].gkbar_hh2 = self.P['g_k']*1e-3  # S/cm^2
        self.tree[0].vtraub_hh2 = self.P['v_th']
        self.tree[0].insert('im')
        self.tree[0].gkbar_im = self.P['g_km']*1e-3  # S/cm^2
        self.tree[0].taumax_im = self.P['t_max']  # ms
        self.tree[0].ek = self.P['E_k']
        self.tree[0].ena = self.P['E_na']
        self.tree[0].insert('Ih')
        self.tree[0].gIhbar_Ih = self.P['g_Ih']*1e-3  # S/cm^2

    def insert_active_dend(self):
        """Insert Na and K (fast and slow) channels in dendrites."""
        for dend in self.tree[1:]:
            dend.insert('hh2')
            dend.gnabar_hh2 = self.P['g_na_d']*1e-3  # S/cm^2
            dend.gkbar_hh2 = self.P['g_k_d']*1e-3  # S/cm^2
            dend.vtraub_hh2 = self.P['v_th']
            dend.insert('im')
            dend.gkbar_im = self.P['g_km_d']*1e-3  # S/cm^2
            dend.taumax_im = self.P['t_max']  # ms
            dend.ek = self.P['E_k']
            dend.ena = self.P['E_na']
            dend.insert('Ih')
            dend.gIhbar_Ih = self.P['g_Ih_d'] * 1e-3  # S/cm^2

    def set_weights(self, w_e, w_i):
        """Assign AMPA and NMDA weights with ratio r_n, and GABA weights.

        Parameters
        ----------
        w_e, w_i : E and I synaptic weights
        """
        r = self.r_na
        self.w_e = w_e
        self.w_i = w_i
        self.w_1 = w_e/(1 + r)
        self.w_2 = w_e*r/(1 + r)
        self.w_3 = w_i

    def create_stim(self, S, syns, weights):
        """Create vecstim and netcon objects for simulation.

        Parameters
        ----------
        S : ndarray
            presynaptic spike times
        syns : list
            neuron synapse objects
        weights : ndarray
            synaptic weights

        Returns
        -------
        t_vec_list : list
            spike time vectors
        stim_list : list
            vec_stim objects
        con_list : list
            netcon objects
        """
        t_vec_list = []
        stim_list = []
        con_list = []
        for k, (t_list, syn) in enumerate(zip(S, syns)):
            stim = h.VecStim()
            t_vec = h.Vector(t_list[~np.isinf(t_list)])
            stim.play(t_vec)
            t_vec_list.append(t_vec)
            stim_list.append(stim)
            con_list.append(h.NetCon(stim, syn, 0, 0, weights[k]))
        return t_vec_list, stim_list, con_list

    def simulate(self, T, dt, v_init, S_e, S_i, I_inj=0, break_flag=False):
        """Run simulation and record voltage from every compartment.

        Parameters
        ----------
        T : int
            total simulation time (ms)
        dt : float
            time step (ms)
        v_init : float
            initial voltage (mV)
        S_e, S_i : array_like
            presynaptic spike times for each E and I synapse
        I_inj : int
            injected current at soma (default 0)
        break_flag : bool
            interupt simulation at time of first spike (default False)

        Returns
        -------
        t : ndarray
            time vector
        v : ndarray
            voltage vector
        """

        h.dt = dt
        h.steps_per_ms = 1.0/h.dt
        t_a, stim_a, self.ampa = self.create_stim(S_e, self.AMPA, self.w_1)
        t_n, stim_n, self.nmda = self.create_stim(S_e, self.NMDA, self.w_2)
        t_g, stim_g, self.gaba = self.create_stim(S_i, self.GABA, self.w_3)
        t = h.Vector()
        t.record(h._ref_t, dt)
        v = []
        for sec in self.tree:
            for k in range(sec.nseg):
                v_temp = h.Vector()
                v_temp.record(sec((k+1/2)/sec.nseg)._ref_v, dt)
                v.append(v_temp)
        if np.abs(I_inj) > 0:
            stim = h.IClamp(0.5, sec=self.tree[0])
            stim.dur = T-150
            stim.amp = I_inj
            stim.delay = 100
        if break_flag:
            nc = h.NetCon(self.tree[0](0.5)._ref_v, None, sec=self.tree[0])
            nc.threshold = 0
            nc.record(self.halt)

        h.v_init = v_init
        h.tstop = T
        h.run()
        t = np.array(t)
        v = np.array(v)
        return t, v

    def halt(self):
        """Interupt simulation. Use with Netcon.record """
        if h.t > 100:
            h.stoprun = 1


class NModelOnline(NModel):
    def __init__(self, P):
        super().__init__(P)
        self.O = {}

    def train(self, T, dt, v_init, S_e, S_i, label):
        """Run simulation and record voltage from every compartment.

        Parameters
        ----------
        T : int
            total simulation time (ms)
        dt : float
            time step (ms)
        v_init : float
            initial voltage (mV)
        S_e, S_i : array_like
            presynaptic spike times for each E and I synapse
        I_inj : int
            injected current at soma (default 0)
        break_flag : bool
            interupt simulation at time of first spike (default False)

        Returns
        -------
        t : ndarray
            time vector
        v : ndarray
            voltage vector
        """

        h.dt = dt
        h.steps_per_ms = 1.0/h.dt
        self.O['t_s'] = np.arange(0, T + dt, dt)
        self.O['s'] = np.zeros(len(self.O['t_s']))
        self.O['E'] = np.zeros(len(self.O['t_s']))
        self.O['label'] = label
        self.O['S_e'] = S_e
        self.O['S_i'] = S_i
        t_a, stim_a, self.O['ampa'] = self.create_stim(S_e, self.AMPA, self.w_1)
        t_n, stim_n, self.O['nmda'] = self.create_stim(S_e, self.NMDA, self.w_2)
        t_g, stim_g, self.O['gaba'] = self.create_stim(S_i, self.GABA, self.w_3)
        t = h.Vector()
        t.record(h._ref_t, dt)
        v = []
        for sec in self.tree:
            for k in range(sec.nseg):
                v_temp = h.Vector()
                v_temp.record(sec((k+1/2)/sec.nseg)._ref_v, dt)
                v.append(v_temp)

        self.avg_err(0)
        if label > 0:
            self.stim = h.IClamp(0.5, sec=self.tree[0])
            self.stim.dur = 1e9
            self.stim.delay = 0
            self.I_inj = h.Vector(self.P['beta']*self.O['E'])
            self.t_inj = h.Vector(self.O['t_s'])
            self.I_inj.play(self.stim._ref_amp, self.t_inj, 1)

        nc = h.NetCon(self.tree[0](0.5)._ref_v, None, sec=self.tree[0])
        nc.threshold = 0
        nc.record(self.update_weights)

        h.v_init = v_init
        h.tstop = T
        h.run()
        t = np.array(t)
        v = np.array(v)
        return t, v

    def update_weights(self):
        """ Update synaptic weights. Triggered by somatic spike in 'train' """
        O = self.O
        P = self.P
        t_ind = int(h.t / h.dt)
        t0_ind = t_ind - int(P['t_window'] / h.dt)
        O['s'][t_ind:] += 1e3*1/P['tau_s']*np.exp(-1/P['tau_s']*(
                O['t_s'][t_ind:] - O['t_s'][t_ind]))
        self.avg_err(t_ind)
        if O['label'] > 0:
            I_inj_new = h.Vector(P['beta']*O['E'])
            self.I_inj.copy(I_inj_new)

        if h.t > P['delay']:
            v_dend = np.array([self.tree[int(seg[0])](seg[1]).v for seg in
                               self.seg_list])
            f_e, f_i = training.kernel_grad(t0_ind, t_ind, O['S_e'], O['S_i'],
                    self.seg_e, self.seg_i, self.b_type_e, self.b_type_i,
                    v_dend, P['kernel'], h.dt)
            f_e[f_e < 0] = 0
            f_i[f_i > 0] = 0
            delta_e = O['E'][t_ind]*f_e
            delta_i = O['E'][t_ind]*f_i
            w_e = self.w_e + P['sf_e']*P['alpha']*delta_e
            w_i = self.w_i + P['alpha']*delta_i
            w_e[w_e < 0] = 0
            w_i[w_i < 0] = 0
            w_e[w_e > P['w_e_max']] = P['w_e_max']
            w_i[w_i > P['w_i_max']] = P['w_i_max']
            self.set_weights(w_e, w_i)

            for k in range(self.P['N_e']):
                O['ampa'][k].weight[0] = self.w_1[k]
                O['nmda'][k].weight[0] = self.w_2[k]
            for k in range(self.P['N_i']):
                O['gaba'][k].weight[0] = self.w_3[k]

    def avg_err(self, t_ind):
        """Updates exponentially weighted moving average of classification
        error from index t_ind onwards."""
        O = self.O
        P = self.P
        steps = len(O['E'][t_ind:])
        label = (O['label']+ 1)/2
        s_bin = np.array(O['s'])
        if label == 0:
            s_bin[s_bin < 0.1] = 0
            s_bin[s_bin >= 0.1] = 1
        else:
            s_bin[s_bin < P['s_th']] = 0
            s_bin[s_bin >= P['s_th']] = 1
        for k in range(1, steps):
            O['E'][t_ind+k] = 1/(1 + h.dt/P['tau_e'])*(O['E'][t_ind+k-1] +
                            h.dt/P['tau_e']*(label - s_bin[t_ind+k]))
