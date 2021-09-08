"""
Parameters for simulation and training with morphology 2.
"""
import numpy as np


def init_params():
    """ Create dictionary of model parameters """
    tree = './input/Allen2_smth.swc'
    N_e = 800  # number of excitatory synapses
    N_i = 200  # number of inhibitory synapses
    soma = [0]
    basal = [3, 5, 7, 15, 29, 37, 89, 94, 97, 100, 102, 103]
    oblique = [41, 81, 82]
    apical = [47, 48, 50, 51, 56, 57, 65, 72, 74, 77]
    locs_e = np.array(basal + oblique + apical)  # location of excitatory synapses
    locs_i = np.array(basal + oblique + apical)  # location of inhibitory synapses
    l_seg = 10  # maximum segment size (um)
    c_m = 1.0  # somatic specific capacitance (uF/cm^2)
    c_m_d = 1.0 # dendritic specific capacitance (uF/cm^2)
    R_a = 150 * 1e-3  # axial resistivity (k ohm cm)
    tau_m = 10. # membrane time constant (ms)
    E_r = -75.  # resting potential (mv)
    E_e = 0.  # excitatory reversal potential (mv)
    E_i = -75.  # inhibitory reversal potential (mv)
    tauA = np.array([0.1, 2.])  # AMPA synapse rise and decay time (ms)
    g_max_A = 0.175*1e-3  # AMPA conductance (uS)
    tauN = np.array([2., 75.])  # NMDA synapse rise and decay time (ms)
    g_max_N = 0.35*1e-3  # NMDA conductance (uS)
    tauG = np.array([1., 5.])  # GABA synapse rise and decay time (ms)
    g_max_G = 0.8 * 1e-3  # GABA conductance (uS)
    r_na = 2.  # NMDA/AMPA ratio
    E_na = 50.  # Na reversal potential (mV)
    E_k = -80.  # K reversal potential (mV)
    E_hcn = -45. # HCN reversal potential (mV)
    g_na = 80.  # somatic Na conductance (mS/cm^2)
    g_k = 40.  # somatic K conductance (mS/cm^2)
    g_km = 3.  # somatic K slow conductance (mS/cm^2)
    g_Ih = 0  # somatic Ih conductance (mS/cm^2)
    g_na_d = 2  # dendritc Na conductance (mS/cm^2)
    g_k_d = 1.  # dendritic K conductance (mS/cm^2)
    g_km_d = 0.15  # dendritic K slow conductance (mS/cm^2)
    g_Ih_d = 0  # dendritic Ih conductance (mS/cm^2)
    v_th = -60.  # Traub and Miles threshold parameter (mV)
    t_max = 0.2e3  # slow K adaptation time scale (ms)
    active_d = False  # active or passive dendrites
    active_n = True  # active or passive NMDA receptors

    P = {'tree': tree,
         'N_e': N_e,
         'N_i': N_i,
         'soma': soma,
         'basal': basal,
         'oblique': oblique,
         'apical': apical,
         'locs_e': locs_e,
         'locs_i': locs_i,
         'l_seg': l_seg,
         'c_m': c_m,
         'c_m_d': c_m_d,
         'R_a': R_a,
         'tau_m': tau_m,
         'E_r': E_r,
         'E_e': E_e,
         'E_i': E_i,
         'tauA': tauA,
         'tauN': tauN,
         'tauG': tauG,
         'g_max_A': g_max_A,
         'g_max_N': g_max_N,
         'g_max_G': g_max_G,
         'r_na': r_na,
         'E_na': E_na,
         'E_k': E_k,
         'E_hcn': E_hcn,
         'g_na': g_na,
         'g_k': g_k,
         'g_km': g_km,
         'g_Ih': g_Ih,
         'g_na_d': g_na_d,
         'g_k_d': g_k_d,
         'g_km_d': g_km_d,
         'g_Ih_d': g_Ih_d,
         'v_th': v_th,
         't_max': t_max,
         'active_d': active_d,
         'active_n': active_n}
    return P
