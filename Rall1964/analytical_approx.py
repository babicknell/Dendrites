#!/usr/bin/env python3

import numpy as np
import math
from matplotlib import pyplot

m = 80 # approximation parameters for temporal and spatial discretisation of stimulus
n = 80 # both n and m should be divisible by 4

#conductance parameters
g = 1/(m*n)
E = 1

#time vectors (stimulation and plotting)
t = np.linspace(0, 1, m, endpoint=False)
t_steps = 100;
tt = np.linspace(0,2.5,t_steps)

#location matrix
L = 2;
x_0 = 0.1*L
deltaX = 0.8*L
x = np.linspace(x_0, x_0 + deltaX, n, endpoint=False)
x = np.tile(x, (m,1))

#Green's function (semi-infinite, sealed end at 0)
def greens(x,y,t,s):
	gr = np.exp(-(t - s))/(4*(t - s))**0.5*(np.exp(-(x - y)**2/(4*(t - s))) 
	+ np.exp(-(x + y)**2/(4*(t - s))))
	return gr

#create look-up tables of function values
G = np.zeros((m,n,m,n))
for i in range(m):
		for k in range(i):
				G[i,:,k,:] = greens(x[i,:], x[k,:], t[i], t[k])

G_sol = np.zeros((m,n,t_steps))
for t_ind in range(t_steps):
	i = 0
	while (i < m) and (t[i] < tt[t_ind]):
		G_sol[i,:,t_ind] = greens(0, x[i,:], tt[t_ind], t[i])
		i+=1
	
#create coefficient matrix
A_in = np.zeros((m, n))
A_out = np.zeros((m, n))
A_all = np.zeros((m, n))

for i in range(0,m):
	s_in = (3 - math.floor(4*i/m))*n//4
	s_out =  math.floor(4*i/m)*n//4
	for j in range(n):
		A_all[i,j] = g*E/4 - g/4*np.sum(A_all[:(i-1),:]*G[i,j,:(i-1),:])
		
		if j >= s_out and j < (s_out + n//4):
			A_out[i,j] = g*E - g*np.sum(A_out[:(i-1),:]*G[i,j,:(i-1),:])
		
		if j >= s_in and j < (s_in + n//4):
			A_in[i,j] = g*E - g*np.sum(A_in[:(i-1),:]*G[i,j,:(i-1),:])
						
#recursive Green’s function expansion of depolarisation
V_in = np.zeros((t_steps,))
V_out = np.zeros((t_steps,))
V_all = np.zeros((t_steps,))

for t_ind in range(t_steps):
	vij_in = 0
	vij_out = 0
	vij_all = 0
	i = 0
	while (i < m) and (t[i] < tt[t_ind]):
		
		vij_all += np.sum(A_all[i,:]*G_sol[i,:,t_ind])
		
		vij_out += np.sum(A_out[i,:]*G_sol[i,:,t_ind])
		
		vij_in += np.sum(A_in[i,:]*G_sol[i,:,t_ind])
		
		i+=1
		
	V_in[t_ind] = vij_in
	V_out[t_ind] = vij_out
	V_all[t_ind] = vij_all

#plotting 
pyplot.figure(figsize=(8,4))
v_plot_out = pyplot.plot(tt, V_out, 'r-')
v_plot_in = pyplot.plot(tt, V_in, 'k-')
v_plot_all = pyplot.plot(tt, V_all, 'b-')
pyplot.xlabel('$t/\\tau$')
pyplot.ylabel('$(V-E_r)/(E_e-E_r)$')
pyplot.legend(v_plot_out + v_plot_in +v_plot_all, ['OUT', 'IN', 'ALL'])
pyplot.axis([0, 2.5, 0, 0.2])
pyplot.xticks([0, 0.5, 1, 1.5, 2])
pyplot.yticks([0, 0.05, 0.1, 0.15, 0.2])
pyplot.show()
