"""
Functions for generating sets of random presynaptic spike sequences.
"""

import numpy as np
import numba as nb


class PreSyn:
    """
    For building trains of non-homogeneous Poisson presynaptic spikes.

    Parameters
    ----------
    r_0 : float
            background spike rate (s^-1)
    sigma : float
            standard deviation of precisely timed spikes (ms)

    Attributes
    ----------
    r_0 : float
        background spike rate (s^-1)
    sigma : float
        standard deviation of precisely timed spikes (ms)
    """

    def __init__(self, r_0, sigma, tau_d=None):
        """ Constructor """
        self.r_0 = r_0
        self.sigma = sigma
        self.tau_d = tau_d

    def rate(self, t, t_on, t_off, stim_on, stim_off, s, r, spike_times):
        """ Define instantaneous rate function.

        Parameters
        ----------
        t : ndarray
            time vector
        t_on, t_off : int
            onset and offset of background activity
        stim_on, stim_off :  int
            onset and offset of stimulus-dependent activity
        s : float
            interpolates between rate (s=0) and temporal code (s=1)
        r : float
            time-averaged stim-dependent firing rate
        spike_times : array_like
            precisely timed elevation in firing rate

        Returns
        -------
        rr : ndarray
            instantaneous rate vector corresponding to times t
        """
        rr = np.zeros(t.shape)
        rr[(t >= t_on) & (t < t_off)] += 1e-3*self.r_0
        rr[(t >= stim_on) & (t < stim_off)] += (1 - s)*r
        if t_off <= stim_off:
            rr[t >= stim_off] += 1e-3*self.r_0
        if (r > 0) & (s*len(spike_times) > 0):
            rr += r*(stim_off - stim_on)/len(spike_times)*s*gauss_spike_time(t,
                                                        spike_times, self.sigma)
        return rr

    def spike_train(self, t_on, t_off, stim_on, stim_off, s, r, spike_times):
        """ Generate Poisson spike train by rejection sampling

        Parameters
        ----------
        t : ndarray
            time vector
        t_on, t_off : int
            onset and offset of background activity
        stim_on, stim_off :  int
            onset and offset of stimulus-dependent activity
        s : float
            interpolates between rate (s=0) and temporal code (s=1)
        r : float
            time-averaged stim-dependent firing rate
        spike_times : array_like
            precisely timed elevation in firing rate

        Returns
        -------
        train : ndarray
            sequence of spike times
        """
        if s*len(spike_times) > 0:
            r_max = 1e-3*self.r_0 + ((1 - s)*r + r*(stim_off - stim_on)/
                len(spike_times)*s*gauss_spike_time(np.array([0]), np.array([0]),
                self.sigma)[0])
        else:
            r_max = 1e-3*self.r_0 + r
        T = max(t_off, stim_off+200)
        num_spikes = np.random.poisson(r_max*(T - t_on))
        train = np.random.uniform(t_on, T, num_spikes)
        accept = np.where(
            self.rate(train, t_on, t_off, stim_on, stim_off, s, r, spike_times)
            / r_max >= np.random.rand(num_spikes))[0]
        train = train[accept]
        train.sort()
        return train


@nb.jit(nopython=True, cache=True)
def gauss_spike_time(t, spike_times, sigma):
    """ Helper for PreSyn class. Creates time-dependent rate
    function peaked about spike_times.

    Parameters
    ----------
    t : ndarray
        time vector
    spike_times : array_like
        precisely timed elevation in firing rate
    sigma : float
        standard deviation of gaussian (ms)

    Returns
    -------
    g : ndarray
        rate function corresponding to times t
    """
    g = np.zeros(t.shape)
    for t_k in spike_times:
        g += 1/(2*np.pi*sigma**2)**0.5*np.exp(-(t - t_k)**2/(2*sigma**2))
    return g


def build_seqs(p, N_e, N_i, T0, T, n):
    """ Generate n inputs per synapse at uniformly distributed times.

    Parameters
    ----------
    p : int
        number of patterns
    N_e, N_i : int
        number of excitatory and inhibitory synapses
    T0, T : int
        initial and final times
    n : int
        spikes per synapse

    Returns
    -------
    S_e, S_i : list
        presynaptic spike patterns
    """
    S_e = []
    S_i = []
    for k in range(p):
        S_e.append(T0 + (T-T0)*np.random.rand(N_e, n))
        S_i.append(T0 + (T-T0)*np.random.rand(N_i, n))
    return S_e, S_i


def build_rate_seq(rates, T0, T):
    """Poisson inputs with prescribed rates.

    Parameters
    ----------
    rates : array_like
        presynaptic firing rates for set of synapses
    T0, T : int
        initial and final times

    Returns
    -------
    s_k_pad : ndarray
        array of spike times padded with infs
    """
    s_k = []
    for rate in rates:
        if rate > 0:
            spike_times = [np.random.exponential(1/rate)]
        else:
            spike_times = [np.inf]
        while sum(spike_times) < T - T0:
            spike_times.append(np.random.exponential(1/rate))
        s_k.append(T0 + np.cumsum(spike_times[:-1]))
    num_spikes = [len(s) for s in s_k]
    s_k_pad = np.full((len(rates), np.max(num_spikes)), np.inf)
    for j, (n, s) in enumerate(zip(num_spikes, s_k)):
        s_k_pad[j, :n] = s
    return s_k_pad


def lognormal_rates(p, N_e, N_i, mu, sigma):
    """Log-normally distributed firing rates with mean exp(mu+sigma^2/2)

    Parameters
    ----------
    p : int
        number of patterns
    N_e, N_i : int
        number of excitatory and inhibitory synapses
    mu : float
        mean of log
    sigma : float
        standard deviation of log

    Returns
    -------
    rates_e, rates_i : list
        sets of E and I rate vectors
    """
    rates_e = []
    rates_i = []
    N = N_e + N_i
    for k in range(p):
        rates = 1e-3*np.random.lognormal(mu, sigma, N)
        rates_e.append(rates[:N_e])
        rates_i.append(rates[N_e:])
    return rates_e, rates_i


def sparse_rates(p, N_e, N_i, mu, r_max):
    """Sparsely distributed firing rates with mean mu and max r_max

    Parameters
    ----------
    p : int
        number of patterns
    N_e, N_i : int
        number of excitatory and inhibitory synapses
    mu : float
        ensemble average rate
    r_max : float
        rate of active synapses

    Returns
    -------
    rates_e, rates_i : list
        sets of E and I rate vectors
    """
    rates_e = []
    rates_i = []
    p_max = mu/r_max
    for k in range(p):
        r = 1e-3*r_max*(np.random.rand(N_e + N_i) < p_max)
        rates_e.append(r[:N_e])
        rates_i.append(r[N_e:])
    return rates_e, rates_i


def assoc_rates(num, N_e, N_i, r_mean, r_max):
    """ Define sets of rate vectors for feature binding task. Draws for two
    stimulus features separately and then forms conjunctions of pairs

    Parameters
    ----------
    num : int
        total number of patterns (num = p*p, for p patterns per feature)
    N_e, N_i : int
        number of excitatory and inhibitory synapses
    r_mean : float
        ensemble average rate
    r_max : float
        rate of active synapses

    Returns
    -------
    rates_e, rates_i : list
        sets of E and I rate vectors
    """
    p = int(num**0.5)
    rates_e = []
    rates_i = []
    n_e1 = int(N_e/2)
    n_i1 = int(N_i/2)
    n_e2 = N_e - n_e1
    n_i2 = N_i - n_i1
    re1, ri1 = sparse_rates(p, n_e1, n_i1, r_mean, r_max)
    re2, ri2 = sparse_rates(p, n_e2, n_i2, r_mean, r_max)
    for j in range(p):
        for k in range(p):
            re = np.hstack((re1[j], re2[k]))
            ri = np.hstack((ri1[j], ri2[k]))
            rates_e.append(re)
            rates_i.append(ri)
    return rates_e, rates_i


def assoc_seqs(num, N_e, N_i, T0, T, n):
    """Create sequences for feature binding task.

    Parameters
    ----------
    num : int
        total number of patterns (num = p*p, for p patterns per feature)
    N_e, N_i : int
        number of excitatory and inhibitory synapses
    T0, T : int
        initial and final times
    n : int
        number of precisely timed events per active synapse

    Returns
    -------
    S_e, S_i : list
        sets of E and I event/spike times
    """
    S_e = []
    S_i = []
    p = int(num**0.5)
    n_e1 = int(N_e/2)
    n_i1 = int(N_i/2)
    n_e2 = N_e - n_e1
    n_i2 = N_i - n_i1
    se1, si1 = build_seqs(p, n_e1, n_i1, T0, T, n)
    se2, si2 = build_seqs(p, n_e2, n_i2, T0, T, n)
    for j in range(p):
        for k in range(p):
            s_e = np.vstack((se1[j], se2[k]))
            s_i = np.vstack((si1[j], si2[k]))
            S_e.append(s_e)
            S_i.append(s_i)
    return S_e, S_i


def superimpose(S_1, S_2):
    """Combine two sequences and sort.

    Parameters
    ----------
    S_1, S_2 :  array_like
        spike sequences for set of synapses
    """
    S = np.full((S_1.shape[0], S_1.shape[1]+S_2.shape[1]), np.inf)
    S[:, :S_1.shape[1]] = S_1
    S[:, S_1.shape[1]:] = S_2
    S.sort(1)
    return S


def assign_labels(p=50):
    """Randomly label p patterns into two categories of equal size."""
    L = np.ones(p)
    rand_inds = np.arange(p)
    np.random.shuffle(rand_inds)
    L[rand_inds[:p//2]] = -1
    return L


def rate2temp(S):
    """Expand sequences with multiple spikes per synapse to single spikes per
    (duplicated) synapse.

    Parameters
    ----------
    S : array_like
        original spike pattern with multiple spikes per synapse

    Returns
    -------
    S_temp : ndarray
        expanded sequence
    syn_ind : list
        original indices of dummy synapses
    """
    S_temp = []
    syn_ind = []
    for k, s in enumerate(S):
        t_k = list(s[s < np.inf])
        S_temp = S_temp + t_k
        syn_ind = syn_ind + len(t_k)*[k]
    S_temp = np.array([S_temp]).T
    return S_temp, syn_ind


def subsequence(S, t_start, t_end):
    """Extract sub-sequence from S between two time points.

    Parameters
    ----------
    S :  array_like
        spike pattern
    t_start, t_end : float
        range for extracting subsequence

    Returns
    -------
    S_sub : list
        extracted subsequence
    """
    S_sub = []
    for k, s in enumerate(S):
        S_sub.append(s[(s >= t_start) & (s < t_end)])
    return S_sub


def translate(S, del_t, T1, T2):
    """ Periodic translation of input sequences.

    Parameters
    ----------
    S :  array_like
        spike pattern
    del_t : int
        translation time (ms)
    T1, T2 : int
        initial and final times for periodic translation

    Returns
    -------
    S_t : ndarray
        translated spike pattern
    """
    S_t = np.array(S)
    S_t -= del_t
    S_t[S_t < T1] += (T2 - T1)
    return S_t


def jitter_phase(S, stim_s, stim_e, jit_mag):
    """Periodic translation of input spike times.

    Parameters
    ----------
    S :  array_like
        spike pattern
    stim_s, stim_e: int
        start and end times for periodic translation (ms)
    jit_mag : float
        translation magnitude for each synapse (ms)

    Returns
    -------
    s : ndarray
        translated spike pattern
    """
    s = np.array(S)
    for k, s_k in enumerate(s):
        if s_k[0] < np.inf:
            s_k += jit_mag[k]
    s[s > stim_e] -= (stim_e - stim_s)
    s[s < stim_s] += (stim_e - stim_s)
    return s


def compress_stim(S, compression, stim_on):
    """ Compress input sequence

    Parameters
    ----------
    S :  array_like
        spike pattern
    compression : float
        factor to compress spike pattern
    stim_on : int
        stimulus onset time (ms)

    Returns
    -------
    s : ndarray
        compressed sequences of spike times
    """
    s = np.array(S)
    s[s < np.inf] -= stim_on
    s[s < np.inf] *= compression
    s[s < np.inf] += stim_on
    return s


def periodic_stim(S, T, periods):
    """ Extend temporal sequence over multiple periods

    Parameters
    ----------
    S :  array_like
        spike pattern
    T : int
        time of one presentation (ms)
    periods :  int
        nuber of periods

    Returns
    -------
    s : ndarray
        periodic sequences of spike times
    """
    s = np.array(S)
    for k in range(periods-1):
        s = np.hstack((s, s[:, -1].reshape(-1, 1)+T))
    return s
