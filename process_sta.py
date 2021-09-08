""" Process STA data to create data structures with measures of synaptic
 activity. Requires output from 'sta_data' """

import numpy as np
import pickle

results = './outputs/sta/sta_act1'
processed = results + '_proc'


t_window = 101  # analysis window (max time preceding somatic spikes)
cell, V, F_e, F_i, W_e, W_i = pickle.load(open(results, 'rb'))
basal, oblique, apical = cell.P['basal'], cell.P['oblique'], cell.P['apical']
locs_e = cell.sec_e
locs_i = cell.sec_i

E_b = []    # excitatory basal
E_a = []    # excitatory apical
I_b = []    # inhibitory basal
I_a = []    # inhibitory apical

# Recorded Variables
# seg: dendritic segment, v_dend: local voltage, s_e/s_i: time of presynaptic spike,
# f_e/f_i: voltage gradient, w_e/w_i: synaptic weight,
# synch_e/synch_i: number of other presynaptic spikes on same branch,
# j: somatic spike index
num_output_spikes = len(V)
for j in range(num_output_spikes):
    f_e, e_ind, s_e = F_e[j]
    f_i, i_ind, s_i = F_i[j]
    w_e = W_e[j]
    w_i = W_i[j]
    branches_e = cell.sec_e[0, e_ind]
    branches_i = cell.sec_i[0, i_ind]
    num_spikes = len(s_e)
    for k in range(num_spikes):
        seg = cell.seg_e[e_ind[k]]
        branch = branches_e[k]
        synch_e = sum((branches_e == branch) & (s_e[:, 0] >= -t_window))
        synch_i = sum((branches_i == branch) & (s_i[:, 0] >= -t_window))
        v_dend = V[j][0][seg]
        if branch in apical:
            E_a.append([seg, v_dend, s_e[k][0], f_e[k], w_e[k], synch_e, synch_i, j])
        else:
            E_b.append([seg, v_dend, s_e[k][0], f_e[k], w_e[k], synch_e, synch_i, j])
    num_spikes = len(s_i)
    for k in range(num_spikes):
        seg = cell.seg_i[i_ind[k]]
        branch = branches_i[k]
        synch_e = sum((branches_e == branch) & (s_e[:, 0] >= -t_window))
        synch_i = sum((branches_i == branch) & (s_i[:, 0] >= -t_window))
        v_dend = V[j][0][seg]
        if branch in apical:
            I_a.append([seg, v_dend, s_i[k][0], f_i[k], w_i[k], synch_e, synch_i, j])
        else:
            I_b.append([seg, v_dend, s_i[k][0], f_i[k], w_i[k], synch_e, synch_i, j])

E_b = np.array(E_b)
E_a = np.array(E_a)
I_b = np.array(I_b)
I_a = np.array(I_a)
pickle.dump([E_b, E_a, I_b, I_a], open(processed, 'wb'))
