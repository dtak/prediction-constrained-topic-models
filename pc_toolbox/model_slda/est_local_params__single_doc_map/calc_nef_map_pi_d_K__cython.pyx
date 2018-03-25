#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
from libc.math cimport log, exp, abs

def calc_nef_map_pi_d_K__cython(
        double[:] init_pi_d_K,
        double[:,:] topics_KUd,
        double[:,:] ct_topics_KUd,
        double convex_alpha_minus_1=0.0,
        int pi_max_iters=0,
        double pi_converge_thr=0.0,
        double pi_step_size=0.0,
        double pi_step_decay_rate=0.0,
        double pi_min_mass_preserved_to_trust_step=1.0,
        double pi_min_step_size=0.0,
        **kwargs):
    """ Find MAP estimate of the K-dim. proba vector for specific document.

    Uses Natural-parameter Exponential Family (NEF) formulation,
    so the optimization problem is always convex.

    Finds solution via iterative exponentiated gradient steps.

    Returns
    -------
    pi_d_K : 1D array, size K
        Contains non-negative entries that sum to one.
    info_dict : dict
        Contains info about the optimization
    """

    cdef int K = topics_KUd.shape[0]
    cdef int Ud = topics_KUd.shape[1]

    cdef double[:] denom_Ud = np.zeros(Ud)
    cdef double[:] grad_K = np.zeros(K)
    cdef double[:] pi_d_K = np.asarray(init_pi_d_K).copy()

    cdef double cur_L1_diff = 1.0
    cdef double new_pi_sum = 0.0
    cdef double new_pi_k = 0.0
    cdef int giter = 0
    cdef int did_converge = 0
    cdef int n_restarts = 0
    cdef int k = 0
    cdef int u = 0
    cdef double max_val = -1e9
    while giter < pi_max_iters:
        for k in range(K):
            grad_K[k] = 0.0
        for u in xrange(Ud):
            denom_Ud[u] = 0.0
            for k in xrange(K):
                denom_Ud[u] += pi_d_K[k] * topics_KUd[k,u]
            denom_Ud[u] = 1.0 / denom_Ud[u]
            for k in range(K):
                grad_K[k] += denom_Ud[u] * ct_topics_KUd[k, u]
        #np.dot(pi_d_K, topics_KUd, out=denom_Ud)
        #np.divide(1.0, denom_Ud, out=denom_Ud)
        #np.dot(ct_topics_KUd, denom_Ud, out=grad_K)

        max_val = -1e9
        for k in range(K):
            grad_K[k] += convex_alpha_minus_1 / (1e-9 + pi_d_K[k])
            grad_K[k] *= pi_step_size
            if (grad_K[k] > max_val):
                max_val = grad_K[k]

        # Let grad_K now contain the new pi vector
        new_pi_sum = 0.0
        for k in range(K):
            grad_K[k] = pi_d_K[k] * exp(grad_K[k] - max_val)
            new_pi_sum += grad_K[k]

        if new_pi_sum <= pi_min_mass_preserved_to_trust_step:
            if pi_step_size > pi_min_step_size:
                # Retry from previous pi_d_K with smaller step size
                n_restarts += 1
                pi_step_size *= pi_step_decay_rate
                continue
            else:
                # We've reached minimum step size. Abort.
                break

        giter += 1
        if giter % 5 == 0:
            cur_L1_diff = 0.0
            for k in range(K):
                new_pi_k = grad_K[k] / new_pi_sum
                cur_L1_diff += abs(pi_d_K[k] - new_pi_k)
                pi_d_K[k] = new_pi_k
            if cur_L1_diff < pi_converge_thr:
                did_converge = 1
                break
        else:
            for k in range(K):
                pi_d_K[k] = grad_K[k] / new_pi_sum

    return np.asarray(pi_d_K), dict(
        n_iters=giter,
        did_converge=did_converge,
        n_restarts=n_restarts,
        cur_L1_diff=cur_L1_diff,
        pi_max_iters=pi_max_iters,
        pi_converge_thr=pi_converge_thr,
        pi_step_size=pi_step_size,
        pi_min_step_size=pi_min_step_size,
        convex_alpha_minus_1=convex_alpha_minus_1)
