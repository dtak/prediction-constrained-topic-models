import autograd.numpy as np

from calc_nef_map_pi_d_K__defaults import DefaultDocTopicOptKwargs

def calc_nef_map_pi_d_K__autograd(
        word_id_d_Ud=None,
        word_ct_d_Ud=None,
        topics_KUd=None,
        topics_KV=None,
        convex_alpha_minus_1=None,
        init_pi_d_K=None,
        ct_topics_KUd=None,
        pi_max_iters=DefaultDocTopicOptKwargs['pi_max_iters'],
        pi_converge_thr=DefaultDocTopicOptKwargs['pi_converge_thr'],
        pi_step_size=DefaultDocTopicOptKwargs['pi_step_size'],
        pi_min_step_size=DefaultDocTopicOptKwargs['pi_min_step_size'],
        pi_step_decay_rate=DefaultDocTopicOptKwargs['pi_step_decay_rate'],
        pi_min_mass_preserved_to_trust_step=\
            DefaultDocTopicOptKwargs['pi_min_mass_preserved_to_trust_step'],
        **kwargs):
    ''' Find MAP estimate of the K-dim. proba vector for specific document.

    Uses Natural-parameter Exponential Family (NEF) formulation,
    so the optimization problem is always convex.

    Finds solution via iterative exponentiated gradient steps.

    Returns
    -------
    pi_d_K : 1D array, size K
        Contains non-negative entries that sum to one.
    info_dict : dict
    '''
    pi_step_size = float(pi_step_size)
    pi_converge_thr = float(pi_converge_thr)

    if topics_KUd is None:
        topics_KUd = topics_KV[:, word_id_d_Ud]
    K = topics_KUd.shape[0]

    # Precompute some useful things
    if ct_topics_KUd is None:
        ct_topics_KUd = topics_KUd * word_ct_d_Ud[np.newaxis, :]

    # Parse convex_alpha_minus_1
    convex_alpha_minus_1 = float(convex_alpha_minus_1)
    assert convex_alpha_minus_1 < 1.0
    assert convex_alpha_minus_1 >= 0.0

    # Initialize as uniform vector over K simplex
    if init_pi_d_K is None:
        init_pi_d_K = np.ones(K) / float(K)
    else:
        init_pi_d_K = np.asarray(init_pi_d_K)
    assert init_pi_d_K.ndim == 1
    assert init_pi_d_K.size == K

    pi_d_K = 1.0 * init_pi_d_K
    best_pi_d_K = 1.0 * init_pi_d_K
    # Start loop over iterations
    did_converge = 0
    n_restarts = 0
    giter = 0
    cur_L1_diff = 1.0
    while giter < pi_max_iters:
        giter = giter + 1
        denom_Ud = 1.0 / np.dot(pi_d_K, topics_KUd)
        grad_K = pi_step_size * (
            np.dot(ct_topics_KUd, denom_Ud)
            + convex_alpha_minus_1 / (1e-9 + pi_d_K)
            )
        grad_K = grad_K - np.max(grad_K)
        new_pi_d_K = pi_d_K * np.exp(grad_K)
        new_pi_d_K_sum = np.sum(new_pi_d_K)
        if new_pi_d_K_sum <= pi_min_mass_preserved_to_trust_step:
            if pi_step_size > pi_min_step_size:
                # Undo the latest update to pi_d_K
                # and continue from previous pi_d_K with smaller step size
                giter = giter - 1
                n_restarts = n_restarts + 1
                pi_step_size = pi_step_size * pi_step_decay_rate
                pi_d_K = 1.0 * best_pi_d_K
                continue
            else:
                pi_d_K = 1.0 * best_pi_d_K
                break
        pi_d_K = new_pi_d_K / new_pi_d_K_sum
        # Check for convergence every few iters
        if giter % 5 == 0:
            cur_L1_diff = np.sum(np.abs(best_pi_d_K - pi_d_K))
            if cur_L1_diff < pi_converge_thr:
                did_converge = 1
                break
        best_pi_d_K = 1.0 * pi_d_K

    return pi_d_K, dict(
        n_iters=giter,
        pi_max_iters=pi_max_iters,
        did_converge=did_converge,
        cur_L1_diff=cur_L1_diff,
        pi_converge_thr=pi_converge_thr,
        n_restarts=n_restarts,
        pi_step_size=pi_step_size,
        pi_min_step_size=pi_min_step_size,
        convex_alpha_minus_1=convex_alpha_minus_1)