"""
Functions for defining morphology and synapse placement.
"""
import numpy as np
import neurom


def synapse_locations_rand(secs, N, nseg, seed):
    """Build array of random synapse locations.

    Parameters
    ----------
    secs : list
        sections to contatin synapses
    N : int
        number of synapses
    nseg : int
        number of segments in sections
    seed : int
        random seed

    Returns
    -------
    syns : ndarray
        synapse locations [section, segment]

    """
    np.random.seed(seed)
    secs = [n*[secs[k]] for k, n in enumerate(nseg)]
    secs = np.array([s for sub in secs for s in sub])
    sec_locs = [np.arange(1/(2*n), 1, 1/n) for n in nseg]
    sec_locs = np.array([s for sub in sec_locs for s in sub])
    distal_inds = np.where(sec_locs>0.1)[0]
    secs = secs[distal_inds]
    sec_locs = sec_locs[distal_inds]
    seg2sec = np.vstack((secs, sec_locs)).T
    seg2sec = np.tile(seg2sec, (N//len(seg2sec) + 1, 1))
    rand_inds = np.arange(len(seg2sec))
    np.random.shuffle(rand_inds)
    syns = seg2sec[rand_inds[:N], :]
    syns = syns[syns[:, 0].argsort()].T
    for sec in secs:
        inds = np.where(syns[0, :] == sec)[0]
        syns[1, inds] = syns[1, inds[syns[1, inds].argsort()]]
    return syns


def reconstruction(filename):
    """Build adjacency matrix, lengths, radii and lists of points and sections
    from swc morphology file. Uses the `neurom` package.

    Parameters
    ----------
    filename : str
        path to .swc morphology file

    Returns
    -------
    A : ndarray
        adjacency matrix for dendritic sections
    L : ndarray
        section lengths
    a : ndarray
        section radii
    sec_points : list
        section coordinates
    sec : list
        section indices
    """

    nrn = neurom.load_neuron(filename)

    num_sp = nrn.soma.points.shape[0]
    a_soma = nrn.soma.radius
    if num_sp == 1:
        L_soma = 2*a_soma
    else:
        L_soma = nrn.soma.area/(2*np.pi*a_soma)
    soma = [sec.id for sec in nrn.sections if sec.type.name == 'soma']
    num_ss = len(soma)
    tree_points = [sec.points for sec in nrn.sections[num_ss:]]
    L = [L_soma*1e-4]
    a = [a_soma*1e-4]
    for points in tree_points:
        path = points[:, :3]
        path = (np.sum(np.diff(path, axis=0)**2, axis=1))**0.5
        path = np.insert(np.cumsum(path), 0, 0)*1e-4
        a_sec = points[:, 3]*1e-4				# (cm)
        L.append(path[-1])
        a.append([path, a_sec])
    L = np.array(L)

    if num_sp == 3:
        s_points = np.zeros((2, tree_points[0].shape[1]))
        s_points[0, :4] = nrn.soma.points[1]
        s_points[1, :4] = nrn.soma.points[2]
    elif num_sp > 3:
        s_points = np.zeros((2, tree_points[0].shape[1]))
        s_points[0, :4] = nrn.soma.points[0]
        s_points[1, :4] = np.mean(nrn.soma.points, axis=0)
    else:
        s_points = np.zeros((2, tree_points[0].shape[1]))
        s_points[1, :4] = nrn.soma.points[0]

    sec_points = [s_points] + tree_points
    parents = []
    for sec in nrn.sections[num_ss:]:
        if sec.parent is not None:
            parents.append(sec.parent.id - (num_ss - 1))
        else:
            parents.append(0)
    parents = np.array(parents)

    M = len(sec_points)
    A = np.zeros((M, M))
    for k, p in enumerate(parents):
        A[p, k + 1] = 1
    axon = [sec.id for sec in nrn.sections if sec.type.name == 'axon']
    basal = [sec.id for sec in nrn.sections if sec.type.name == 'basal_dendrite']
    apical = [sec.id for sec in nrn.sections if sec.type.name == 'apical_dendrite']
    trunk = [sec.id for sec in nrn.sections if sec.type.name == 'all']
    relabel = []
    for ind in basal:
        if nrn.sections[ind].parent is not None and nrn.sections[ind].parent.type.name == 'axon':
            relabel.append(ind)
    axon = (axon + relabel)
    axon.sort()
    for ind in relabel:
        basal.remove(ind)

    axon = [a - (num_ss - 1) for a in axon]
    basal = [b - (num_ss - 1) for b in basal]
    apical = [ap - (num_ss - 1) for ap in apical]
    trunk = [tr - (num_ss - 1) for tr in trunk]
    secs = basal + apical + trunk
    return A, L, a, sec_points, secs


def seg_geometry(L, a, nseg):
    """Build lists of length, radius and area for each segment.

    Parameters
    ----------
    L : ndarray
        section lengths
    a : ndarray
        section radii
    nseg : list
        number of segments in each section

    Returns
    -------
    L_s : ndarray
        segment lengths
    a_s : ndarray
        segment radii
    area : ndarray
        segment surface areas
    """
    L_s = [nseg[k]*[L[k]/nseg[k]] for k, l_sec in enumerate(L)]
    L_s = np.array([l_seg for sub in L_s for l_seg in sub])
    a_s = [[a[0]]]
    sub_n = 10
    for n, a_sec in zip(nseg[1:], a[1:]):
        path = a_sec[0]
        radius = a_sec[1]
        nn = sub_n*n
        pp = np.array([path[-1]*(k/nn+1/(2*nn)) for k in range(nn)])
        radius_interp = np.interp(pp, path, radius)
        radius_av = np.array([np.mean(radius_interp[k*sub_n:(k+1)*sub_n]) for k
                                in range(n)])
        a_s.append(radius_av)
    a_s = np.array([a_seg for sub in a_s for a_seg in sub])
    area = np.array([2*np.pi*r*l for r, l in zip(a_s, L_s)])
    return L_s, a_s, area


def branch_type(locs, branch_ids):
    """Assign brach type identifiers to array of synapse locations

    Parameters
    ----------
    locs : ndarray
        synapse locations
    branch_ids : list
        section indices of branches

    Returns
    -------
    b_type : list
        branch type (soma=-1, basal=0, oblique=1, apical=2)
    """

    basal, oblique, apical = branch_ids
    b_type = []
    for sec in locs[0, :]:
        if sec in basal:
            b_type.append(0)
        elif sec in oblique:
            b_type.append(1)
        elif sec in apical:
            b_type.append(2)
        else:
            b_type.append(-1)
    return b_type
