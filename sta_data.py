""" Combine STA data files and save."""

import pickle

model = 'act1'
results = './outputs/sta/sta_'+model+'_data/sta_'+model

files = range(0, 500)
cell, V, data_e, data_i, W_e, W_i = pickle.load(open(results+'_'+str(files[0]), 'rb'))
for file in files[1:]:
        try:
            sta = pickle.load(open(results+'_'+str(file), 'rb'))
            V = V + sta[1]
            data_e = data_e + sta[2]
            data_i = data_i + sta[3]
            W_e = W_e + sta[4]
            W_i = W_i + sta[5]
        except:
            continue

pickle.dump([cell, V, data_e, data_i, W_e, W_i], open('./outputs/sta/sta_'+model, 'wb'))
