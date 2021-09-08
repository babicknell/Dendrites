"""Plot 2d STA plasticity kernels for active or passive models
fitted with fit_kernels.py """

from matplotlib import pyplot
import numpy as np
import pickle

from dendrites import kernels

kernel_fit = './input/kernel_fit_act1'

t = np.arange(0, 101)
v = np.arange(-80, 1)


def display_kernel(z, min, max, ax, map):
    """ Display 2d plasticity kernel as heatmap.

    Parameters
    ----------
    z : ndarray
        kernel evaluated on t x v mesh
    min, max : float
        min and max values for colormap
    ax : matplotlib axis object
        axis to plot to
    map : str
        name of colormap, e.g. 'viridis'
    Returns
    -------
    im : matplotlib AxesImage
        kernel heatmap
    """
    im = ax.matshow(z, vmin=min, vmax=max, origin='lower', cmap=map)
    ax.set_xlabel('time before spike (ms)', fontsize=11)
    ax.set_ylabel('local voltage (mV)', fontsize=11)
    ax.invert_xaxis()
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks(np.arange(0, 100, 20))
    ax.set_yticklabels(np.arange(-80, 20, 20))
    ax.set_xticks(np.arange(0, 120, 20))
    ax.set_xticklabels(np.arange(0, 120, 20))
    ax.tick_params(labelsize=12)
    return im


_, params = pickle.load(open(kernel_fit, 'rb'))
p_eb, p_ea, p_ib, p_ia = params

z_eb = kernels.eval_poly_mesh(t, v, p_eb)*1e-3
z_ea = kernels.eval_poly_mesh(t, v, p_ea)*1e-3
z_ib = kernels.eval_poly_mesh(t, v, p_ib)*1e-3
z_ia = kernels.eval_poly_mesh(t, v, p_ia)*1e-3


fig, ax = pyplot.subplots(2, 2)
im_b = display_kernel(z_eb, 0, np.max(z_eb), ax[0, 0], 'viridis')
im_a = display_kernel(z_ea, 0, np.max(z_ea), ax[0, 1], 'viridis')
ax[0, 0].set_title('Excitatory: basal', fontsize=12)
ax[0, 1].set_title('Excitatory: apical', fontsize=12)
cb_b = fig.colorbar(im_b, ax=ax[0, 0],  ticks=[0, np.round(np.max(z_eb)-0.01, 2)],
                    fraction=0.0375, pad=0.04, label=r'$\partial v/\partial w$'
                    +' (mVnS' + r'$^{-\mathregular{1}}$)')
cb_a = fig.colorbar(im_a, ax=ax[0, 1],  ticks=[0, np.round(np.max(z_ea)-0.01, 2)],
                    fraction=0.0375, pad=0.04, label=r'$\partial v/\partial w$'
                    +' (mVnS' + r'$^{-\mathregular{1}}$)')
im2_b = display_kernel(z_ib, np.min(z_ib), 0, ax[1, 0], 'viridis_r')
im2_a = display_kernel(z_ia, np.min(z_ia), 0, ax[1, 1], 'viridis_r')
ax[1, 0].set_title('Inhibitory: basal', fontsize=12)
ax[1, 1].set_title('Inhibitory: apical', fontsize=12)
cb_ib = fig.colorbar(im2_b, ax=ax[1, 0], ticks=[np.round(np.min(z_ib)+0.01, 2), 0],
                     fraction=0.0375, pad=0.04, label=r'$\partial v/\partial w$'
                     +' (mVnS' + r'$^{-\mathregular{1}}$)')
cb_ia = fig.colorbar(im2_a, ax=ax[1, 1], ticks=[np.round(np.min(z_ia)+0.01, 2), 0],
                     fraction=0.0375, pad=0.04, label=r'$\partial v/\partial w$'
                    +' (mVnS' + r'$^{-\mathregular{1}}$)')
ax[0, 0].tick_params(labelsize=9.5)
ax[0, 1].tick_params(labelsize=9.5)
ax[1, 0].tick_params(labelsize=9.5)
ax[1, 1].tick_params(labelsize=9.5)
cb_b.ax.tick_params(labelsize=9.5)
cb_a.ax.tick_params(labelsize=9.5)
cb_ib.ax.tick_params(labelsize=9.5)
cb_ia.ax.tick_params(labelsize=9.5)
pyplot.tight_layout()
pyplot.show()
