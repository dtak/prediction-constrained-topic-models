import numpy as np

from calc_nef_map_pi_d_K__defaults import DefaultDocTopicOptKwargs

def calc_nef_map_pi_d_K__numpy_linesearch(
        word_id_d_Ud=None,
        word_ct_d_Ud=None,
        topics_KUd=None,
        topics_KV=None,
        nef_alpha=None,
        init_pi_d_K=None,
        pi_max_iters=DefaultDocTopicOptKwargs['pi_max_iters'],
        pi_converge_thr=DefaultDocTopicOptKwargs['pi_converge_thr'],
        pi_step_size=DefaultDocTopicOptKwargs['pi_step_size'],
        pi_max_step_size=DefaultDocTopicOptKwargs['pi_max_step_size'],
        pi_min_step_size=DefaultDocTopicOptKwargs['pi_min_step_size'],
        pi_step_decay_rate=DefaultDocTopicOptKwargs['pi_step_decay_rate'],
        pi_min_mass_preserved_to_trust_step=\
            DefaultDocTopicOptKwargs['pi_min_mass_preserved_to_trust_step'],
        verbose=False,
        verbose_pi=False,
        track_stuff=False,
        **kwargs):
    ''' Find MAP estimate of the K-dim. proba vector for specific document.

    Uses Natural-parameter Exponential Family (NEF) formulation.

    Returns
    -------
    pi_d_K : 1D array, size K
        Contains non-negative entries that sum to one.
    info_dict : dict
    '''
    raise ValueError("TODO: NEEDS CHECKING/FIXING")
    
    if topics_KUd is None:
        topics_KUd = topics_KV[:, word_id_d_Ud]

    # Precompute some useful things
    ct_topics_KUd = topics_KUd * word_ct_d_Ud[np.newaxis, :]
    K = topics_KUd.shape[0]

    # Parse nef_alpha
    nef_alpha = float(nef_alpha)
    assert nef_alpha >= 1.0
    convex_alpha_minus_1 = float(nef_alpha) - 1.0
    assert convex_alpha_minus_1 < 1.0
    assert convex_alpha_minus_1 >= 0.0
    
    # Initialize as uniform vector over K simplex
    if init_pi_d_K is None:
        init_pi_d_K = np.ones(K) / float(K)
    else:
        init_pi_d_K = np.asarray(init_pi_d_K)
    assert init_pi_d_K.ndim == 1
    assert init_pi_d_K.size == K

    best_pi_d_K = 1.0 * init_pi_d_K
    best_denom_Ud = np.dot(best_pi_d_K, topics_KUd)
    best_loss = -1.0 * np.inner(word_ct_d_Ud, np.log(best_denom_Ud))

    if track_stuff:
        pi_list = list()
        loss_list = list()
        step_list = list()

    # Start loop over iterations
    did_converge = 0
    n_restarts = 0
    n_improve_in_a_row = 0
    giter = 0
    cur_step_size = pi_step_size * pi_step_decay_rate
    while giter < pi_max_iters:
        giter = giter + 1
        #denom_Ud = 1.0 / np.dot(pi_d_K, topics_KUd)
        grad_K = (
            np.dot(ct_topics_KUd, 1.0 / best_denom_Ud)
            # Purposefully not using alpha here.
            )
        grad_K = grad_K - np.max(grad_K)

        if n_improve_in_a_row > 2:
            # Increase step size (since we seem to be improving regularly)
            cur_step_size = cur_step_size / pi_step_decay_rate

            # But scale it down slightly (between 0.9 and 1.0)
            # so that we avoid too much oscillation around optimum
            cur_step_size *= 1.0 - 0.1 * (float(giter)/float(pi_max_iters))

        cur_step_size = np.minimum(
            max_pi_step_size,
            cur_step_size)
        did_improve = False
        while cur_step_size >= pi_min_step_size:
            new_pi_d_K = best_pi_d_K * np.exp(cur_step_size * grad_K)
            new_pi_d_K_sum = np.sum(new_pi_d_K)
            new_pi_d_K = new_pi_d_K / new_pi_d_K_sum

            new_denom_Ud = np.dot(new_pi_d_K, topics_KUd)
            new_loss = -1.0 * np.inner(word_ct_d_Ud, np.log(new_denom_Ud))
            if new_loss > best_loss:
                # Try smaller stepsize
                cur_step_size = cur_step_size * pi_step_decay_rate
                n_restarts += 1
                n_improve_in_a_row = 0
            else:
                n_improve_in_a_row += 1
                did_improve = True
                break

        # Check for convergence
        delta_mass = np.sum(np.abs(best_pi_d_K - new_pi_d_K))
        if delta_mass < pi_converge_thr:
            did_converge = 1
        if did_improve:
            if verbose:
                delta_loss = (best_loss - new_loss) / np.abs(best_loss)
                msg_str = \
                     "iter %4d step_size %.5f  loss %.8e" \
                      + "  delta_pi %.5f  delta_loss %.9e <<< keep"
                msg_str = msg_str % (
                    giter, cur_step_size, new_loss, delta_mass, delta_loss)
                print msg_str
            if verbose_pi:
                print '  '.join(["%.5e" % a for a in new_pi_d_K])
            best_pi_d_K = 1.0 * new_pi_d_K
            best_denom_Ud = new_denom_Ud
            best_loss = new_loss
        if track_stuff:
            pi_list.append(best_pi_d_K)
            step_list.append(cur_step_size)
            loss_list.append(best_loss)
        if did_converge or not did_improve:
            break

    info_dict = dict(
        did_converge=did_converge,
        n_iters=giter,
        n_iters_try=giter + n_restarts,
        pi_max_iters=pi_max_iters,
        n_restarts=n_restarts,
        pi_converge_thr=pi_converge_thr,
        pi_step_size=cur_step_size,
        pi_min_step_size=pi_min_step_size)
    if track_stuff:
        info_dict['pi_list'] = pi_list
        info_dict['step_list'] = step_list
        info_dict['loss_list'] = loss_list

    return best_pi_d_K, info_dict