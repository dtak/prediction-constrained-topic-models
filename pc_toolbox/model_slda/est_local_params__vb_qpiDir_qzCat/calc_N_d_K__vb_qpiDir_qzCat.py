import argparse
import numpy as np
import os
from scipy.special import gammaln, digamma
from scipy.misc import logsumexp

def calc_N_d_K__vb_coord_ascent__many_tries(
        word_id_d_Ud=None,
        word_ct_d_Ud=None,
        topics_KV=None,
        alpha_K=None,
        init_pi_d_K=None,
        init_name=None,
        init_name_list=None,
        coldstart_initname='prior_mean',
        prng=np.random,
        verbose=False,
        do_trace_elbo=True,
        **lstep_kwargs):
    """ Estimate token-assignment counts for VB approximate posterior.

    Returns
    -------
    N_d_K : 1D array, size K
        N_d_K[k] : count of usage of topic k in document d
    """
    K = alpha_K.size
    if init_name is not None:
        init_name_list = init_name.split("+")
    if init_name_list is None:
        init_name_list = [coldstart_initname]
    assert isinstance(init_name_list, list)

    # Precompute likelihoods
    # lik_d_UdK : 2D array, Ud x K
    lik_d_UdK = topics_KV[:, word_id_d_Ud].T.copy()
    log_lik_d_UdK = np.log(1e-100 + lik_d_UdK)

    best_ELBO = -np.inf
    best_N_d_K = None
    best_info = None
    for init_name in init_name_list:
        if init_name.count("_x") > 0:
            n_reps = int(init_name.split("_x")[1])
        else:
            n_reps = 1

        for rep in xrange(n_reps):
            init_P_d_K = make_initial_P_d_K(
                init_name,
                prng=prng,
                alpha_K=alpha_K,
                init_P_d_K_list=[init_pi_d_K])
            if verbose:
                pprint__N_d_K(init_P_d_K, "init")

            cur_N_d_K, cur_info = calc_N_d_K__vb_coord_ascent(
                word_ct_d_Ud=word_ct_d_Ud,
                lik_d_UdK=lik_d_UdK,
                log_lik_d_UdK=log_lik_d_UdK,
                alpha_K=alpha_K,
                init_P_d_K=init_P_d_K,
                verbose=verbose,
                do_trace_elbo=do_trace_elbo,
                **lstep_kwargs)
            cur_ELBO = calc_elbo_for_single_doc__simplified_from_N_d_K(
                word_ct_d_Ud=word_ct_d_Ud,
                log_lik_d_UdK=log_lik_d_UdK,
                alpha_K=alpha_K,
                N_d_K=cur_N_d_K)
            if verbose:
                pprint__N_d_K(cur_N_d_K, "final", cur_ELBO)

            if cur_ELBO > best_ELBO + 1e-6:
                best_ELBO = cur_ELBO
                best_N_d_K = cur_N_d_K
                best_info = cur_info
                if verbose:
                    print "best: %s" % init_name
            elif cur_ELBO > best_ELBO - 1e-6:
                if verbose:
                    print "tied: %s" % init_name
    if verbose:
        print ""
    best_info['ELBO'] = best_ELBO
    return best_N_d_K, best_info

def calc_N_d_K__vb_coord_ascent(
        word_id_d_Ud=None,
        word_ct_d_Ud=None,
        lik_d_UdK=None,
        log_lik_d_UdK=None,
        topics_KV=None,
        alpha_K=None,
        init_theta_d_K=None,
        init_N_d_K=None,
        init_P_d_K=None,
        lstep_converge_thr=0.0001,
        lstep_max_iters=100,
        do_trace_elbo=False,
        verbose=False,
        **unused_kwargs):
    """ Estimate token-assignment counts for VB approximate posterior.

    Uses one run of coordinate descent.

    Returns
    -------
    N_d_K : 1D array, size K
    info_dict : dict
    """
    if lik_d_UdK is None:
        lik_d_UdK = topics_KV[:, word_id_d_Ud].T.copy()
    if log_lik_d_UdK is None and do_trace_elbo:
        log_lik_d_UdK = np.log(1e-100 + lik_d_UdK)

    P_d_K = np.zeros_like(alpha_K)
    sumresp_U = np.zeros_like(word_ct_d_Ud)
    if init_P_d_K is not None:
        P_d_K[:] = init_P_d_K
        N_d_K = np.zeros_like(alpha_K)
        np.dot(lik_d_UdK, P_d_K, out=sumresp_U)
        np.dot(word_ct_d_Ud / sumresp_U, lik_d_UdK, out=N_d_K)
        N_d_K *= P_d_K
    elif init_theta_d_K is not None:
        N_d_K = np.maximum(init_theta_d_K - alpha_K, 1e-10)
    elif init_N_d_K is not None:
        N_d_K = init_N_d_K

    prev_N_d_K = np.zeros_like(N_d_K)
    digamma_sumtheta_d = digamma(np.sum(alpha_K) + np.sum(word_ct_d_Ud))

    if do_trace_elbo:
        elbo_list = list()
    converge_dist = np.inf
    for local_iter in range(1, 1+lstep_max_iters):
        if do_trace_elbo:
            elbo = calc_elbo_for_single_doc__simplified_from_N_d_K(
                word_ct_d_Ud=word_ct_d_Ud,
                log_lik_d_UdK=log_lik_d_UdK,
                alpha_K=alpha_K,
                N_d_K=N_d_K)
            elbo_list.append(elbo)                 
        np.add(N_d_K, alpha_K, out=P_d_K)
        digamma(P_d_K, out=P_d_K)
        np.subtract(P_d_K, digamma_sumtheta_d, out=P_d_K)
        np.exp(P_d_K, out=P_d_K)
        np.dot(lik_d_UdK, P_d_K, out=sumresp_U)
        # Update DocTopicCounts
        np.dot(word_ct_d_Ud / sumresp_U, lik_d_UdK, out=N_d_K)
        N_d_K *= P_d_K

        if verbose and local_iter % 10 == 0:
            pprint__N_d_K(N_d_K)

        if local_iter % 5 == 0:
            converge_dist = np.sum(np.abs(N_d_K - prev_N_d_K))
            if converge_dist < lstep_converge_thr:
                break
        prev_N_d_K[:] = N_d_K

    opt_info = dict(
        n_iters=local_iter,
        max_iters=lstep_max_iters,
        did_converge=converge_dist < lstep_converge_thr,
        converge_thr=lstep_converge_thr,
        converge_dist=converge_dist,
        )
    if do_trace_elbo:
        opt_info['trace_lb_logpdf_x'] = np.asarray(elbo_list)
        opt_info['trace_lb_logpdf_x_pertok'] = np.asarray(elbo_list) / np.sum(word_ct_d_Ud)
    return N_d_K, opt_info


def calc_elbo_for_single_doc__simplified_from_N_d_K(
        word_ct_d_Ud=None,
        log_lik_d_UdK=None,
        alpha_K=None,
        N_d_K=None):
    theta_d_K = N_d_K + alpha_K
    E_log_pi_d_K = digamma(theta_d_K) - digamma(np.sum(theta_d_K))
    log_resp_d_UK = log_lik_d_UdK + E_log_pi_d_K[np.newaxis,:]
    return (
        np.inner(word_ct_d_Ud, logsumexp(log_resp_d_UK, axis=1))
        + c_Dir_1D(alpha_K) - c_Dir_1D(theta_d_K)
        + np.inner(alpha_K - theta_d_K, E_log_pi_d_K)
        )


def make_initial_P_d_K(
        init_name,
        prng=np.random,
        alpha_K=None,
        init_P_d_K_list=None):
    K = alpha_K.size

    if init_name.count('warm'):
        return init_P_d_K_list.pop()
    elif init_name.count('uniform_sample'):
        return prng.dirichlet(np.ones(K))
    elif init_name.count('prior_sample'):
        return prng.dirichlet(alpha_K)
    elif init_name.count("prior_mean"):
        return alpha_K / np.sum(alpha_K) #np.zeros(K, dtype=alpha_K.dtype)
    else:
        raise ValueError("Unrecognized vb lstep_init_name: " + init_name)

def pprint__N_d_K(N_d_K, label='', elbo=None):
    if elbo:
        print(
            "%6s" % label
            + " " + ' '.join(['%7.2f' % a for a in N_d_K])
            + " %.7e" % elbo)
    else:
        print "%6s" % label, ' '.join(['%7.2f' % a for a in N_d_K])

def c_Dir_1D(alpha_K):
    return gammaln(np.sum(alpha_K)) - np.sum(gammaln(alpha_K))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--Ud', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=1.1)
    parser.add_argument(
        '--lstep_max_iters',
        type=int,
        default=100)
    parser.add_argument('--verbose', type=int, default=0)
    args = parser.parse_args()

    lstep_kwargs = dict(
        lstep_max_iters=args.lstep_max_iters,
        )
    if args.verbose:
        lstep_kwargs['verbose'] = True
        lstep_kwargs['very_verbose'] = True

    K = args.K
    Ud = args.Ud
    alpha_K = args.alpha * np.ones(K, dtype=np.float64)

    prng = np.random.RandomState(12342)
    topics_KV = prng.rand(K, Ud)
    topics_KV /= np.sum(topics_KV, axis=1)[:,np.newaxis]
    word_id_d_Ud = np.arange(Ud)
    word_ct_d_Ud = prng.randint(low=1, high=3, size=Ud)
    word_ct_d_Ud = np.asarray(word_ct_d_Ud, dtype=np.float64)
    print "Applying K=%d topics to doc with Ud=%d uniq terms" % (K, Ud)


    for (init_name, init_pi_d_K) in [
            ('prior_mean', None),
            ('prior_sample', None),
            ('warm', np.arange(K)),
            ]:
        N_d_K, info_dict = calc_N_d_K__vb_coord_ascent__many_tries(
            word_id_d_Ud=word_id_d_Ud,
            word_ct_d_Ud=word_ct_d_Ud,
            topics_KV=topics_KV,
            alpha_K=alpha_K,
            init_name=init_name,
            init_pi_d_K=init_pi_d_K,
            verbose=True,
            **lstep_kwargs)
