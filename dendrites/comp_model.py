#!/usr/bin/env python3
"""
Compartmental model class and functions for simulation and training.
"""
import numpy as np
import numba as nb

from dendrites import nak, Ih
from dendrites import morphology


class CModel:
    """Compartmental model object for simulation of dendritic computation and
    learning.

    Parameters
    ----------
        P : dict
        model and simulation parameters

    Attributes
    ----------
        P : dict
            model and simulation paramaters
        L_s, a_s : ndarray
            segment lengths and radii
        C : ndarray
            adjacency matrix for segments
        Q : ndarray
            axial conductance matrix
        sec_e, sec_i : ndarray
            synapse locations
        seg_e, seg_i : ndarray
            synapse segment numbers
        seg2sec : ndarray
            sections and positions for each segment
        b_type_e, b_type_i : list
            branch types (basal/apical/soma) for all synapses
        H_e, H_i : ndarray
            projection matrices (inputs->compartments)
        w_e, w_i : ndarray
            synaptic weights
        g_ops : ndarray
            list of Gaussian elimination steps for solve
        f_ops : ndarray
            list of forward sub steps for solve
    """

    def __init__(self, P):
        self.P = P
        A, nseg, L, a, self.sec_e, self.sec_i, self.b_type_e, self.b_type_i = \
        self.define_morphology()
        self.L_s, self.a_s, self.area = morphology.seg_geometry(L, a, nseg)
        self.C, self.seg2sec = self.build_comp_matrix(A, nseg)
        self.Q = self.build_axial_mat()
        self.H_e, self.seg_e = self.syn_mat(self.sec_e)
        self.H_i, self.seg_i = self.syn_mat(self.sec_i)
        self.w_e = np.array(self.sec_e.shape[1]*[P['g_max_A'] + P['g_max_N']])
        self.w_i = np.array(self.sec_i.shape[1]*[P['g_max_G']])
        self.g_ops = self.gauss_ops()
        self.f_ops = self.fsub_ops()

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
        sec_e, sec_i :  ndarray
            locations of excitatory and inhibitory synapses
        b_type_e, b_type_i : list
            branch types (soma/basal/oblique/apical) for all synapses
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
        return A, nseg, L, a, sec_e, sec_i, b_type_e, b_type_i

    def build_comp_matrix(self, A, nseg):
        """ Expand adjacency matric for sections into adjacency matrix for all
        segments. Uses Wye-delta transformation at branch points.

        Parameters
        ----------
        A : ndarray
            adjacency matrix for dendritic sections
        nseg : ndarray
            number of segments in each section

        Returns
        -------
        C : ndarray
            adjacency matrix for dendritic segments
        seg2sec : ndarray
            array mapping the new locations [sec, sec_location].
        """
        n_secs = A.shape[0]
        secs = [nseg[k]*[k] for k in range(n_secs)]
        secs = [s for sub in secs for s in sub]
        sec_locs = [np.arange(1/(2*nseg[k]), 1, 1/nseg[k]) for k in
                    range(n_secs)]
        sec_locs = [s for sub in sec_locs for s in sub]
        seg2sec = np.vstack((secs, sec_locs)).T
        A = np.array(A + A.T)
        branch_points = np.array([k for k in range(n_secs) if sum(A[k, k:]) >=
                                2])
        terminal_points = np.array([k for k in range(n_secs-1) if sum(A[k, k:])
                                == 0])
        M = sum(nseg)
        C = np.diag(np.ones(M - 1), 1)
        for tp in terminal_points:
            tp_new = np.where(seg2sec[:, 0] == tp)[0][-1]
            C[tp_new, tp_new + 1] = 0
        for bp in branch_points:
            bp_new = np.where(seg2sec[:, 0] == bp)[0][-1]
            C[bp_new, bp_new + 1] = 0
        for bp in branch_points:
            bp_new = np.where(seg2sec[:, 0] == bp)[0][-1]
            daughters = np.where(A[bp, :])[0]
            daughters = daughters[daughters > bp]
            d_new = [np.where(seg2sec[:, 0] == d)[0][0] for d in daughters]
            C[bp_new, d_new] = 1
            for n, d_i in enumerate(d_new):
                for d_j in d_new[n+1:]:
                    C[d_i, d_j] = 1
        C = C + C.T
        return C, seg2sec

    def g_axial(self, a_i, a_j, L_i, L_j, R_a):
        """Axial conductance from compartment j to i in unbranched section."""
        return (a_i*a_j**2)/(R_a*L_i*(L_j*a_i**2 + L_i*a_j**2))

    def g_axial_b(self, a_i, a_j, L_i, L_j, a_k, L_k, R_a):
        """Axial conductance from compartment j to i through branch point."""
        return ((a_i*a_j**2)/(L_i**2*L_j))/(R_a*(a_i**2/L_i + a_j**2/L_j +
                sum(a_k**2 / L_k)))

    def build_axial_mat(self):
        """Build and return axial conductance matrix Q."""
        R_a = self.P['R_a']
        L_s = self.L_s
        a_s = self.a_s
        C = self.C
        Q = np.zeros(C.shape)
        ind_i, ind_j = np.where(C)
        for i, j in zip(ind_i, ind_j):
                adj = np.intersect1d(np.where(C[i, :])[0], np.where(C[j, :])[0])
                a_k = a_s[adj]
                L_k = L_s[adj]
                Q[i, j] = self.g_axial_b(a_s[i], a_s[j], L_s[i], L_s[j],
                                        a_k, L_k, R_a)
        Q = Q + np.diag(-np.sum(Q, axis=1))
        return Q

    def set_weights(self, w_e, w_i):
        """Set synaptic weights."""
        self.w_e = w_e
        self.w_i = w_i

    def syn_mat(self, syns):
        """Matrix to project conductances to specified compartments with
        conversion to current in units of mS/area.

        Parameters
        ----------
        syns : ndarray
            locations of synapses

        Returns
        -------
        H : ndarray
            synapse->segment projection matrix
        syn_segs : ndarray
            segment locations of synapses

        """
        H = np.zeros((self.C.shape[0], syns.shape[1]))
        seg_soma = [0 for c in syns.T if c[0] == 0]
        seg_dend = [int(np.where(np.sum(np.abs(self.seg2sec - c), axis=1) <
                                1e-5)[0][0])
                    for c in syns.T if c[0] > 0]
        for k, s in enumerate(seg_soma + seg_dend):
            H[s, k] = 1/(1e3*self.area[s])
        syn_segs = np.array(seg_soma + seg_dend)
        return H, syn_segs

    def gauss_ops(self):
        """Returns sequence of pivots and targets for Gaussian elimination in
        solve.
        """
        targets = [np.where(self.C[:k, k])[0] for k in range(self.C.shape[0])]
        g_ops = []
        for k in range(1, self.C.shape[0]):
            for target in targets[k]:
                g_ops.append([k, target])
        g_ops = np.array(g_ops[::-1])
        return g_ops

    def fsub_ops(self):
        """Return array of non-zero elements for forward substitution in solve.
        """
        Q = self.C + np.diag(np.arange(self.C.shape[0]))
        row_reduce(Q, self.g_ops)
        Q[np.abs(Q) < 1e-10] = 0
        np.fill_diagonal(Q, 0)
        f_ops = np.vstack((np.where(Q)[0], np.where(Q)[1])).T
        return f_ops

    def init_IC(self, v_init):
        """ Inititialse voltage, gating and synaptic variables.

        Parameters
        ----------
        v_init : int
            initial voltage

        Returns
        -------
        v0 : list
            initial voltage in all compartments
        gate0 : list
            initial states of gating variables
        syn0 : list
            initial states of synapse kinetics states
        """
        N_e, N_i = self.P['N_e'], self.P['N_i']
        v0 = len(self.C)*[v_init]
        gate0 = [nak.m_inf(v_init), nak.h_inf(v_init), nak.n_inf(v_init),
                 nak.p_inf(v_init), Ih.m_inf(v_init)]
        syn0 = [np.zeros((2, N_e)), np.zeros((2, N_e)), np.zeros((2, N_i))]
        return v0, gate0, syn0

    def set_IC(self, soln, stim, t0):
        """ Set conditions from specific time point in previous simulation.

        Parameters
        ----------
        soln :  list
            solution returned by `simulate`
        stim : list
            conductance states returned by `simulate`
        t0 : int
            time index to extract model states

        Returns
        -------
        v0 : list
            initial voltage in all compartments
        gate0 : list
            initial states of gating variables
        syn0 : list
            initial states of synapse kinetics states
        """
        ind_e, ind_i = stim[0], stim[1]
        v0 = soln[0][:, t0]
        gate0 = [soln[1][:, t0], soln[2][:, t0], soln[3][:, t0], soln[4][:, t0], soln[5][:, t0]]
        syn0 = [np.zeros((2, self.P['N_e'])), np.zeros((2, self.P['N_e'])),
            np.zeros((2,self.P['N_i']))]
        syn0[0][:, ind_e] = np.vstack((stim[2][:, t0], stim[3][:, t0]))
        syn0[1][:, ind_e] = np.vstack((stim[4][:, t0], stim[5][:, t0]))
        syn0[2][:, ind_i] = np.vstack((stim[6][:, t0], stim[7][:, t0]))
        return v0, gate0, syn0

    def simulate(self, t_0, t_1, dt, IC, S_e, S_i, I_inj=0, break_flag=False):
        """Simulate instance of CModel using input sequences S_e and S_i from
        initial conditions IC. Records detailed synaptic state variables.

        Parameters
        ----------
        t_0, t_1 : int
            start and end times of simulation
        dt : float
            timestep
        IC : array_like
            initial conditions for all state variables (v0, gate0, syn0)
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
        soln : list
            arrays of model states (voltage and gating variables) [v, m, h, n, p, hcn]
        stim :  list
            arrays of synaptic conductance and kinetic states and associated
            indices [ind_e, ind_i, A_r, A_d, N_r, N_d, G_r, G_d]

         """
        P = self.P
        E_r, E_e, E_i, E_na, E_k, E_hcn, g_na, g_k, g_km, g_Ih, g_na_d, g_k_d, \
        g_km_d, g_Ih_d, r_na, tau_m, cm_s, cm_d, tauA, tauN, tauG, active_d, \
        active_n = (P['E_r'], P['E_e'], P['E_i'], P['E_na'], P['E_k'], P['E_hcn'],
                    P['g_na'], P['g_k'],P['g_km'], P['g_Ih'], P['g_na_d'],
                    P['g_k_d'], P['g_km_d'], P['g_Ih_d'], P['r_na'], P['tau_m'],
                    P['c_m'], P['c_m_d'], P['tauA'], P['tauN'], P['tauG'],
                    P['active_d'], P['active_n'])

        t = np.arange(t_0, t_1+dt, dt)
        if isinstance(IC, (int, float)):
            v_0, gate_0, syn_0 = self.init_IC(IC)
        else:
            v_0, gate_0, syn_0 = IC

        M = self.Q.shape[0]
        cm = np.hstack((cm_s, (M-1)*[cm_d]))

        Id = np.eye(M)
        d_inds = np.diag_indices(M)
        ind_e = np.where(S_e[:, 0] < t_1)[0]
        ind_i = np.where(S_i[:, 0] < t_1)[0]
        w_e = self.w_e[ind_e]
        w_i = self.w_i[ind_i]
        H_e = self.H_e[:, ind_e]
        H_i = self.H_i[:, ind_i]
        A_r, A_d, N_r, N_d, G_r, G_d = build_stim2(t, dt, syn_0[0][:, ind_e],
                                    syn_0[1][:, ind_e], syn_0[2][:, ind_i],
                                    S_e[ind_e], S_i[ind_i], tauA, tauN, tauG)
        I_inj *= 1/(self.area[0]*1e3)

        if active_d:
            a_inds = np.arange(M)
        else:
            a_inds = [0]
        M_active = len(a_inds)

        v = np.zeros((M, len(t)))
        n = np.zeros((M_active, len(t)))
        m = np.zeros((M_active, len(t)))
        h = np.zeros((M_active, len(t)))
        p = np.zeros((M_active, len(t)))
        hcn = np.zeros((M_active, len(t)))
        v[:, 0] = v_0
        m[:, 0], h[:, 0], n[:, 0], p[:, 0], hcn[:, 0] = gate_0
        g_na = np.hstack((g_na, (M_active-1)*[g_na_d]))
        g_k = np.hstack((g_k, (M_active-1)*[g_k_d]))
        g_km = np.hstack((g_km, (M_active-1)*[g_km_d]))
        g_Ih = np.hstack((g_Ih, (M_active-1)*[g_Ih_d]))

        J = dt*(self.Q.T*1/cm).T
        q = self.Q[d_inds]
        g_a = H_e@(w_e/(1 + r_na)*(A_d - A_r).T).T
        g_n = H_e@(w_e*r_na/(1 + r_na)*(N_d - N_r).T).T
        g_g = H_i@(w_i*(G_d - G_r).T).T

        if active_n:
            update_J = update_jacobian
            rhs = dvdt
        else:
            update_J = update_jacobian_pas
            rhs = dvdt_pas

        for k in range(1, len(t)):
            m[:, k] = m[:, k-1] + (1 - np.exp(-dt/nak.tau_m(v[a_inds, k - 1])))*(
                    nak.m_inf(v[a_inds, k - 1]) - m[:, k - 1])
            h[:, k] = h[:, k-1] + (1 - np.exp(-dt/nak.tau_h(v[a_inds, k - 1])))*(
                    nak.h_inf(v[a_inds, k - 1]) - h[:, k - 1])
            n[:, k] = n[:, k-1] + (1 - np.exp(-dt/nak.tau_n(v[a_inds, k - 1])))*(
                    nak.n_inf(v[a_inds, k - 1]) - n[:, k - 1])
            p[:, k] = p[:, k-1] + (1 - np.exp(-dt/nak.tau_p(v[a_inds, k - 1])))*(
                    nak.p_inf(v[a_inds, k - 1]) - p[:, k - 1])
            hcn[:, k] = hcn[:, k-1] + (1 - np.exp(-dt/Ih.tau_m(v[a_inds, k - 1])))*(
                    Ih.m_inf(v[a_inds, k - 1]) - hcn[:, k - 1])

            update_J(J, q, v[:, k-1], g_a[:, k], g_n[:, k], g_g[:, k],
                            E_e, tau_m, cm, dt, d_inds)
            f = rhs(v[:, k-1], g_a[:, k], g_n[:, k], g_g[:, k], self.Q, E_r,
                    E_e, E_i, tau_m, cm)
            f[0] += I_inj
            f *= dt/cm
            a = Id - J	 # note to future self: J multiplied by dt in update step
            a[a_inds, a_inds] += dt/cm[a_inds]*(g_na*m[:, k]**3*h[:, k] + g_k*n[:, k]**4 +
            g_km*p[:, k] + g_Ih*hcn[:, k])
            b = v[:, k-1] + f - J@v[:, k-1]
            b[a_inds] += dt/cm[a_inds]*(g_na*m[:, k]**3*h[:, k]*E_na + g_k*n[:, k]**4*E_k +
            g_km*p[:, k]*E_k + g_Ih*hcn[:, k]*E_hcn)
            v[:, k] = solve(a, b, self.g_ops, self.f_ops)
            if v[0, k] > 0 and break_flag:
                break
        soln = [v, m, h, n, p, hcn]
        stim = [ind_e, ind_i, A_r, A_d, N_r, N_d, G_r, G_d]
        return t, soln, stim

    def grad_w(self, soln, stim, t, dt, Z_e, Z_i, z_ind_e, z_ind_i):
        """Compute gradients associated with individual input
        spikes using solution from `simulate2'. Z_e and Z_i are expanded copies
        of the input pattern between those times (one spike per synapse; see
        `sequences.rate2temp`).

        Parameters
        ----------
        soln : list
            arrays of model states (voltage and gating variables)
            [v, m, h, n, p]. See `simulate2`.
        stim : list
            arrays of synaptic conductance states and associated indices
            [ind_e, ind_i, A_r, A_d, N_r, N_d, G_r, G_d]
        t : ndarray
            simulation time vector
        dt : float
            timestep from original forward simulation
        Z_e, Z_i : array_like
            presynaptic spike time for dummy copies of E and I synapses
        z_ind_e, z_ind_i :
            original indices of dummy synapses

        Returns
        -------
        f_e, f_i : ndarray
            dv_soma/dw for E and I synapses as a function of time
        """

        P = self.P
        E_e, E_i, tau_m, E_k, E_na, E_hcn, g_k, g_na, g_km, g_Ih, g_na_d, g_k_d, g_km_d, g_Ih_d, \
        r_na, cm_s, cm_d, tauA, tauN, tauG, active_d, active_n = (P['E_e'],
            P['E_i'], P['tau_m'], P['E_k'], P['E_na'], P['E_hcn'], P['g_k'],
            P['g_na'], P['g_km'], P['g_Ih'], P['g_na_d'], P['g_k_d'], P['g_km_d'],
            P['g_Ih_d'], P['r_na'], P['c_m'], P['c_m_d'], P['tauA'], P['tauN'],
            P['tauG'], P['active_d'], P['active_n'])
        ind_e, ind_i = stim[0], stim[1]
        M = self.Q.shape[0]
        cm = np.hstack((cm_s, (M - 1) * [cm_d]))
        if active_d:
            a_inds = np.arange(M)
        else:
            a_inds = [0]
            g_Ih = 0
        M_active = len(a_inds)
        ZA, ZN, ZG = build_stim(t, dt, Z_e, Z_i, tauA, tauN, tauG)

        N_e = len(Z_e)
        N_i = len(Z_i)
        Hz_e = np.zeros((self.H_e.shape[0], len(z_ind_e)))
        Hz_i = np.zeros((self.H_i.shape[0], len(z_ind_i)))
        Hz_e[np.where(self.H_e)[0][z_ind_e], np.arange(len(z_ind_e))] = self.H_e[
        np.where(self.H_e)[0][z_ind_e], np.where(self.H_e)[1][z_ind_e]]
        Hz_i[np.where(self.H_i)[0][z_ind_i], np.arange(len(z_ind_i))] = self.H_i[
        np.where(self.H_i)[0][z_ind_i], np.where(self.H_i)[1][z_ind_i]]
        he_inds = (np.where(Hz_e)[0], np.where(Hz_e)[1])
        hi_inds = (np.where(Hz_i)[0], np.where(Hz_i)[1] + N_e)

        g_na = np.hstack((g_na, (M_active-1)*[g_na_d]))
        g_k = np.hstack((g_k, (M_active-1)*[g_k_d]))
        g_km = np.hstack((g_km, (M_active-1)*[g_km_d]))
        g_Ih = np.hstack((g_Ih, (M_active-1)*[g_Ih_d]))

        v, m, h, n, p, hcn = soln
        GA = stim[3] - stim[2]
        GN = stim[5] - stim[4]
        GG = stim[7] - stim[6]

        w_e = self.w_e[ind_e]
        w_i = self.w_i[ind_i]
        H_e = self.H_e[:, ind_e]
        H_i = self.H_i[:, ind_i]
        dhQ = dt*(self.Q.T*1/cm).T

        Y_m = np.zeros((N_e + N_i, M_active))
        Y_h = np.zeros((N_e + N_i, M_active))
        Y_n = np.zeros((N_e + N_i, M_active))
        Y_p = np.zeros((N_e + N_i, M_active))
        Y_hcn = np.zeros((N_e + N_i, M_active))
        B = np.zeros((M, N_e + N_i))
        f_soma = B[0, :]
        f_e = np.zeros((N_e, v.shape[1]))
        f_i = np.zeros((N_i, v.shape[1]))

        a_m = nak.d_alpha_m(v[a_inds, :])*(1 - m) - nak.d_beta_m(v[a_inds, :])*m
        a_h = nak.d_alpha_h(v[a_inds, :])*(1 - h) - nak.d_beta_h(v[a_inds, :])*h
        a_n = nak.d_alpha_n(v[a_inds, :])*(1 - n) - nak.d_beta_n(v[a_inds, :])*n
        a_p = nak.d_alpha_p(v[a_inds, :])*(1 - p) - nak.d_beta_p(v[a_inds, :])*p
        a_hcn = Ih.d_alpha_m(v[a_inds, :])*(1 - hcn) - Ih.d_beta_m(v[a_inds, :])*hcn

        b_m = nak.alpha_m(v[a_inds, :]) + nak.beta_m(v[a_inds, :])
        b_h = nak.alpha_h(v[a_inds, :]) + nak.beta_h(v[a_inds, :])
        b_n = nak.alpha_n(v[a_inds, :]) + nak.beta_n(v[a_inds, :])
        b_p = nak.alpha_p(v[a_inds, :]) + nak.beta_p(v[a_inds, :])
        b_hcn = Ih.alpha_m(v[a_inds, :]) + Ih.beta_m(v[a_inds, :])

        c_m = (g_na*3*(m**2*h).T*(E_na - v[a_inds, :].T)).T
        c_h = (g_na*(m**3).T*(E_na - v[a_inds, :].T)).T
        c_n = (g_k*4*(n**3).T*(E_k - v[a_inds, :].T)).T
        c_p = (g_km*(E_k - v[a_inds, :].T)).T
        c_hcn = (g_Ih*(E_hcn - v[a_inds, :].T)).T

        if active_n:
            g_s = (H_e@(w_e/(1 + r_na)*GA.T).T + H_e@(w_e*r_na/(1 + r_na)*GN.T).T*sigma(v) -
                H_e@(w_e*r_na/(1 + r_na)*GN.T).T*d_sigma(v)*(E_e - v) +
                H_i@(w_i*GG.T).T)
            g_s = (g_s.T + cm/tau_m).T
            gw_e = 1/(1 + r_na)*(Hz_e.T@(E_e - v))*ZA + r_na/(1 + r_na)*(Hz_e.T@(
            (E_e - v)*sigma(v)))*ZN
        else:
            g_s = (H_e@(w_e/(1 + r_na)*GA.T).T + H_e@(w_e*r_na/(1 + r_na)*GN.T).T +
                H_i@(w_i*GG.T).T)
            g_s = (g_s.T + cm/tau_m).T
            gw_e = 1/(1 + r_na)*(Hz_e.T@(E_e - v))*ZA + r_na/(1 + r_na)*(Hz_e.T@(
            E_e - v))*ZN

        g_s[a_inds, :] += (g_k*(n**4).T + g_na*(m**3*h).T + g_km*p.T + g_Ih*hcn.T).T
        gw_i = (Hz_i.T@(E_i - v))*ZG

        for k in range(1, v.shape[1]):
            Y_m += (a_m[:, k-1]/b_m[:, k-1]*B[a_inds, :].T - Y_m)*(
                    1 - np.exp(-dt*b_m[:, k-1]))
            Y_h += (a_h[:, k-1]/b_h[:, k-1]*B[a_inds, :].T - Y_h)*(
                    1 - np.exp(-dt*b_h[:, k-1]))
            Y_n += (a_n[:, k-1]/b_n[:, k-1]*B[a_inds, :].T - Y_n)*(
                    1 - np.exp(-dt*b_n[:, k-1]))
            Y_p += (a_p[:, k-1]/b_p[:, k-1]*B[a_inds, :].T - Y_p)*(
                    1 - np.exp(-dt*b_p[:, k-1]))
            Y_hcn += (a_hcn[:, k-1]/b_hcn[:, k-1]*B[a_inds, :].T - Y_hcn)*(
                    1 - np.exp(-dt*b_hcn[:, k-1]))
            A = np.diag(1 + dt/cm*g_s[:, k]) - dhQ
            B[he_inds] += dt/cm[self.seg_e[z_ind_e]]*gw_e[:, k]
            B[hi_inds] += dt/cm[self.seg_i[z_ind_i]]*gw_i[:, k]
            B[a_inds, :] += (dt/cm[a_inds]*(c_m[:, k]*Y_m + c_h[:, k]*Y_h +
                            c_n[:, k]*Y_n + c_p[:, k]*Y_p + c_hcn[:, k]*Y_hcn)).T
            solve_grad(A, B, self.g_ops, self.f_ops)
            f_e[:, k] = f_soma[:N_e]
            f_i[:, k] = f_soma[N_e:]
        return f_e, f_i


@nb.jit(nopython=True, cache=True)
def kernel(t, tau_r, tau_d):
    """Returns value of double-exponential synaptic kernel.

    Parameters
    ---------
    t : float
        time
    tau_r, tau_d : float
        rise and decay time constants
    """
    t_peak = tau_r*tau_d/(tau_d - tau_r)*np.log(tau_d/tau_r)
    Z = np.exp(-t_peak/tau_d) - np.exp(-t_peak/tau_r)
    return 1/Z*(np.exp(-t/tau_d) - np.exp(-t/tau_r))


@nb.jit(nopython=True, cache=True)
def g(t, S, tau_r, tau_d):
    """Returns vector of synaptic conductances for sequence S at time t.

    Parameters
    ----------
    t : float
        time
    S : ndarray
        presynaptic spike times
    tau_r, tau_d : float
        rise and decay time constants
    """
    s_vec = (t - S)
    for i in range(s_vec.shape[0]):
            for j in range(s_vec.shape[1]):
                if ~(s_vec[i, j] > 0):
                    s_vec[i, j] = 0
    return np.sum(kernel(s_vec, tau_r, tau_d), axis=1)


def sigma(v):
    """NMDA voltage nonlinearity.

    Parameters
    ----------
    v : array_like
        voltage (mV)
    """
    return 1/(1 + 1/3.75*np.exp(-0.062*v))


def d_sigma(v):
    """Derivative of NMDA nonlinearity with respect to v.

    Parameters
    ----------
    v : array_like
        voltage (mV)
    """
    return 0.062*sigma(v)*(1 - sigma(v))


@nb.jit(nopython=True, cache=True)
def build_stim(t, dt, S_e, S_i, tauA, tauN, tauG):
    """AMPA, NMDA and GABA conductance time series.

    Parameters
    ----------
    t : ndarray
        time vector
    dt : float
        timestep
    S_e, S_i : ndarray
        presynaptic spike times for each E and I synapse
    tauA, tauN, tauG : list
        rise and decay time constants for AMPA, NMDA, GABA receptors

    Returns
    -------
    GA, GN, GG : ndarray
        conductance states as a function of time for AMPA, NMDA, GABA receptors
    """
    GA = build_G(t, dt, S_e, tauA[0], tauA[1])
    GN = build_G(t, dt, S_e, tauN[0], tauN[1])
    GG = build_G(t, dt, S_i, tauG[0], tauG[1])
    return GA, GN, GG


@nb.jit(nopython=True, cache=True)
def build_G(t, dt, S, tau_r, tau_d):
    """Build synaptic conductance time series using two-state scheme

    Parameters
    ----------
    t : ndarray
        time vector
    dt : float
        timestep
    S : ndarray
        presynaptic spike times for set of synapses
    tau_r, tau_d : float
        rise and decay time constants

    Returns
    -------
    G : ndarray
        conductance states as a function of time
    """
    G = np.zeros((len(S), len(t)))
    r = np.zeros(len(S))
    d = np.zeros(len(S))
    alpha_r = np.exp(-dt/tau_r)
    alpha_d = np.exp(-dt/tau_d)
    t_peak = tau_r*tau_d/(tau_d - tau_r)*np.log(tau_d/tau_r)
    Z = np.exp(-t_peak/tau_d) - np.exp(-t_peak/tau_r)
    for k, t_k in enumerate(t):
        r *= alpha_r
        d *= alpha_d
        dt_k = S - t_k
        ind = np.where((dt_k > 0) & (dt_k < dt))
        for j, i in enumerate(ind[0]):
            r[i] += 1/Z*np.exp(-dt_k[i, ind[1][j]]/tau_r)
            d[i] += 1/Z*np.exp(-dt_k[i, ind[1][j]]/tau_d)
        G[:, k] = d - r
    return G


@nb.jit(nopython=True, cache=True)
def build_stim2(t, dt, a_init, n_init, g_init, S_e, S_i, tauA, tauN, tauG):
    """AMPA, NMDA and GABA conductance time series with detailed kinetic states.

    Parameters
    ----------
    t : ndarray
        time vector
    dt : float
        timestep
    a_init, n_init, g_init : list
        initial conditions for AMPA, NMDA, GABA receptors
    S_e, S_i : ndarray
        presynaptic spike times for each E and I synapse
    tauA, tauN, tauG : list
        rise and decay time constants for AMPA, NMDA, GABA receptors

    Returns
    -------
    A_r, A_d, N_r, N_d, G_r, G_d : ndarray
        kinetic states as a function of time for AMPA, NMDA, GABA receptors
    """
    A_r, A_d = build_G2(t, dt, S_e, tauA[0], tauA[1], a_init[0], a_init[1])
    N_r, N_d = build_G2(t, dt, S_e, tauN[0], tauN[1], n_init[0], n_init[1])
    G_r, G_d = build_G2(t, dt, S_i, tauG[0], tauG[1], g_init[0], g_init[1])
    return A_r, A_d, N_r, N_d, G_r, G_d


@nb.jit(nopython=True, cache=True)
def build_G2(t, dt, S, tau_r, tau_d, r_init, d_init):
    """Build synaptic conductance time series using two-state scheme, and return
    kinetic states

    Parameters
    ----------
    t : ndarray
        time vector
    dt : float
        timestep
    S : ndarray
        presynaptic spike times for set of synapses
    tau_r, tau_d : float
        rise and decay time constants
    r_init, d_init : ndarray
        kinetics state initial conditions

    Returns
    -------
    R, D : ndarray
        kinetic states as a function of time
    """
    R = np.zeros((len(S), len(t)))
    D = np.zeros((len(S), len(t)))
    r = r_init
    d = d_init
    R[:, 0] = r_init
    D[:, 0] = d_init
    alpha_r = np.exp(-dt/tau_r)
    alpha_d = np.exp(-dt/tau_d)
    t_peak = tau_r*tau_d/(tau_d - tau_r)*np.log(tau_d/tau_r)
    Z = np.exp(-t_peak/tau_d) - np.exp(-t_peak/tau_r)
    for k, t_k in enumerate(t[1:]):
        r *= alpha_r
        d *= alpha_d
        dt_k = S - t_k
        ind = np.where((dt_k > 0) & (dt_k < dt))
        for j, i in enumerate(ind[0]):
            r[i] += 1/Z*np.exp(-dt_k[i, ind[1][j]]/tau_r)
            d[i] += 1/Z*np.exp(-dt_k[i, ind[1][j]]/tau_d)

        R[:, k+1] = r
        D[:, k+1] = d
    return R, D


def dvdt(v, g_a, g_n, g_g, Q, E_r, E_e, E_i, tau_m, cm):
    """ Returns right-hand side of ODE system.

    Parameters
    ----------
    v : ndarray
        voltage in all compartments
    g_a, g_n, gg : ndarray
        AMPA, NMDA GABA conductance states
    Q : ndarray
        axial conductance matrix
    E_r, E_e, E_i : int
        resting, E, and I reversal potentials
    tau_m : float
        membrane time constant
    I_inj : float
        injected current at soma
    """
    return (g_a*(E_e - v) + g_n*sigma(v)*(E_e - v) + g_g*(E_i - v) + cm/tau_m*(E_r - v) + Q@v)


def update_jacobian(J, q, v, g_a, g_n, g_g, E_e, tau_m, cm, dt, d_inds):
    """Update ODE Jacobian matrix.

    Parameters
    ----------
    J : ndarry
        Jacobian matrix
    q : ndarray
        diagonal of axial conductance matrix
    v : ndarray
        voltage in all compartments
    g_a, g_n, gg : ndarray
        AMPA, NMDA GABA conductance states
    E_e : int
        resting potential
    tau_m : float
        membrane time constant
    dt : float
        time step
    d_inds : tuple
        diagonal indices of J
    """
    J[d_inds] = dt/cm*(-g_a - g_n*sigma(v) + g_n*d_sigma(v)*(E_e - v) - g_g - cm/tau_m + q)


def dvdt_pas(v, g_a, g_n, g_g, Q, E_r, E_e, E_i, tau_m, cm):
    """ Returns right-hand side of ODE system with Ohmic NMDA receptors

    Parameters
    ----------
    v : ndarray
        voltage in all compartments
    g_a, g_n, gg : ndarray
        AMPA, NMDA GABA conductance states
    Q : ndarray
        axial conductance matrix
    E_r, E_e, E_i : int
        resting, E, and I reversal potentials
    tau_m : float
        membrane time constant
    I_inj : float
        injected current at soma
    """
    return (g_a*(E_e - v) + g_n*(E_e - v) + g_g*(E_i - v)) + (cm/tau_m*(E_r - v) +
            Q@v)


def update_jacobian_pas(J, q, v, g_a, g_n, g_g, E_e, tau_m, cm, dt, d_inds):
    """Update ODE Jacobian matrix with Ohmic NMDa receptors

    Parameters
    ----------
    J : ndarry
        Jacobian matrix
    q : ndarray
        diagonal of axial conductance matrix
    v : ndarray
        voltage in all compartments (unused)
    g_a, g_n, gg : ndarray
        AMPA, NMDA GABA conductance states
    E_e : int
        resting potential (unused)
    tau_m : float
        membrane time constant
    dt : float
        time step
    d_inds : tuple
        diagonal indices of J
    """
    J[d_inds] = dt/cm*(-g_a - g_n - g_g - cm/tau_m + q)


def solve(Q, b, g_ops, f_ops):
    """Solve linear system of equations Qx=b with Gaussian elimination
    (using v[0] as soma requires clearing upper triangle first).

    Parameters
    ----------
    Q : ndarray
        coefficient matrix
    b : ndarray
        right-hand side
    g_ops : ndarray
        sequence of operations for Gaussian elimination
    g_ops : ndarray
        sequence of operations for forward substitution

    Returns
    -------
    x : ndarray
        solution
    """
    gauss_elim(Q, b, g_ops)
    x = b
    forward_sub(Q, x, f_ops)
    return x


def solve_grad(Q, B, g_ops, f_ops):
    """Solve linear system of matrix equations QX=B with Gaussian elimination
    (using v[0] as soma requires clearing upper triangle first). Note: modifies
    B in place to produce solution X.

    Parameters
    ----------
    Q : ndarray
        coefficient matrix
    B : ndarray
        right-hand side
    g_ops : ndarray
        sequence of operations for Gaussian elimination
    f_ops : ndarray
        sequence of operations for forward substitution

    """
    gauss_elim_mat(Q, B, g_ops)
    X = B
    forward_sub_mat(Q, X, f_ops)


@nb.jit(nopython=True, cache=True)
def gauss_elim(Q, b, g_ops):
    """Gaussian elimination (upper triangle cleared)

    Parameters
    ----------
    Q : ndarray
        coefficient matrix
    b : ndarray
        right-hand side
    g_ops : ndarray
        sequence of operations for Gaussian elimination
    """
    for k in range(g_ops.shape[0]):
        p = g_ops[k][0]
        t = g_ops[k][1]
        if t != p-1:
            b[t] -= Q[t, p]/Q[p, p]*b[p]
            Q[t, :] = Q[t, :] - Q[t, p]/Q[p, p]*Q[p, :]
        else:
            b[t] -= Q[t, p]/Q[p, p]*b[p]
            Q[t, t] = Q[t, t] - Q[t, p]/Q[p, p]*Q[p, t]
            Q[t, p] = 0


@nb.jit(nopython=True, cache=True)
def gauss_elim_mat(Q, B, g_ops):
    """Gaussian elimination (upper triangle cleared) for matrix system

    Parameters
    ----------
    Q : ndarray
        coefficient matrix
    B : ndarray
        right-hand side
    g_ops : ndarray
        sequence of operations for Gaussian elimination
    """
    for k in range(g_ops.shape[0]):
        p = g_ops[k][0]
        t = g_ops[k][1]
        if t != p-1:
            B[t, :] = B[t, :] - Q[t, p]/Q[p, p]*B[p, :]
            Q[t, :] = Q[t, :] - Q[t, p]/Q[p, p]*Q[p, :]
        else:
            B[t, :] = B[t, :] - Q[t, p]/Q[p, p]*B[p, :]
            Q[t, t] = Q[t, t] - Q[t, p]/Q[p, p]*Q[p, t]
            Q[t, p] = 0


@nb.jit(nopython=True, cache=True)
def row_reduce(Q, g_ops):
    """Row reduction of Q to precompute forward subtitution operations.

    Parameters
    ----------
    Q : ndarray
        matrix to be reduced
    g_ops : ndarray
        sequence of operations for Gaussian elimination of Q
    """
    for k in range(g_ops.shape[0]):
        p = g_ops[k][0]
        t = g_ops[k][1]
        Q[t, :] = Q[t, :] - Q[t, p]/Q[p, p]*Q[p, :]


@nb.jit(nopython=True, cache=True)
def forward_sub(Q, x, f_ops):
    """Forward substitution after gauss_elim.

    Parameters
    ----------
    Q : ndarray
        row-reduced matrix
    x : ndarray
        view to rhs b.
    f_ops : ndarray
        sequence of operations for forward substitution
    """
    x /= np.diag(Q)
    for k in range(f_ops.shape[0]):
        r = f_ops[k][0]
        c = f_ops[k][1]
        x[r] -= Q[r, c]/Q[r, r]*x[c]


@nb.jit(nopython=True, cache=True)
def forward_sub_mat(Q, X, f_ops):
    """Forward substitution for matrix system after gauss_elim_mat

    Parameters
    ----------
    Q : ndarray
        row-reduced matrix
    X : ndarray
        view to rhs B.
    f_ops : ndarray
        sequence of operations for forward substitution
    """
    q = np.expand_dims(np.diag(Q), 1)
    X /= q
    for k in range(f_ops.shape[0]):
        r = f_ops[k][0]
        c = f_ops[k][1]
        X[r, :] -= Q[r, c]/Q[r, r]*X[c, :]
