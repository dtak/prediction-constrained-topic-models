import argparse
import numpy as np
import os
from scipy.special import gammaln, digamma
from scipy.misc import logsumexp

from calc_N_d_K__vb_qpiDir_qzCat import (
    calc_N_d_K__vb_coord_ascent__many_tries)

def calc_elbo_for_many_docs(
        dataset=None,
        alpha=None,
        alpha_K=None,
        topics_KV=None,
        verbose=False,
        print_progress_every=-1,
        init_name_list=['prior_mean'],
        init_pi_DK=None,
        prng=None,
        seed=0,
        return_info=False,
        active_ct_thr=0.01,
        do_trace_elbo=False,
        **lstep_kwargs):

    assert dataset is not None
    assert topics_KV is not None

    K = topics_KV.shape[0]
    dtype = topics_KV.dtype
    word_ct_U = np.asarray(dataset['word_ct_U'], dtype=dtype)
    if alpha_K is None:
        alpha_K = float(alpha) * np.ones(K, dtype=dtype)
    else:
        alpha_K = np.asarray(alpha_K, dtype=dtype)

    if return_info:
        theta_DK = np.zeros((dataset['n_docs'], K))

    if init_pi_DK is not None:
        assert init_pi_DK.shape[0] == dataset['n_docs']
        assert init_pi_DK.shape[1] == K
        assert 'warm' in init_name_list
    else:
        init_P_d_K = None

    if prng is None:
        prng = np.random.RandomState(seed)

    ttl_lb_logpdf_x = 0.0
    ttl_n_tokens = 0
    ttl_n_docs = 0

    D = dataset['n_docs']
    if print_progress_every > 0:
        converged_per_doc = np.zeros(D, dtype=np.int32)
        dist_per_doc = np.zeros(D, dtype=np.float64)
        iter_per_doc = np.zeros(D, dtype=np.int32)
        n_active_per_doc = np.zeros(D, dtype=np.float64)
        start_time_sec = time.time()
    for d in range(D):
        start = dataset['doc_indptr_Dp1'][d]
        stop = dataset['doc_indptr_Dp1'][d+1]
        Ud = stop - start
        word_ct_d_Ud = word_ct_U[start:stop]
        word_id_d_Ud = dataset['word_id_U'][start:stop]

        if init_pi_DK is not None:
            init_pi_d_K = init_pi_DK[d]

        N_d_K, info_dict = \
            calc_N_d_K__vb_coord_ascent__many_tries(
                word_id_d_Ud=word_id_d_Ud,
                word_ct_d_Ud=word_ct_d_Ud,
                topics_KV=topics_KV,
                alpha_K=alpha_K,
                init_name_list=init_name_list,
                init_pi_d_K=init_pi_d_K,
                prng=prng,
                verbose=verbose,
                do_trace_elbo=do_trace_elbo,
                **lstep_kwargs)

        if return_info:
            theta_DK[d] = N_d_K + alpha_K

        # Norm constant per document
        h_x_d = gammaln(1.0 + np.sum(word_ct_d_Ud)) \
            - np.sum(gammaln(1.0 + word_ct_d_Ud))

        # Aggregate
        ttl_lb_logpdf_x += info_dict['ELBO'] + h_x_d
        ttl_n_tokens += np.sum(word_ct_d_Ud)
        ttl_n_docs += 1

        if print_progress_every > 0:
            dist_per_doc[d] = info_dict['converge_dist']
            converged_per_doc[d] = info_dict['did_converge']
            iter_per_doc[d] = info_dict['n_iters']
            n_active_per_doc[d] = np.sum(N_d_K >= active_ct_thr)
        # Do the printing of the progress
        if print_progress_every > 0 and (
                (d + 1) % print_progress_every == 0
                or (d + 1) == D
                ):
            msg = make_readable_summary_for_pi_DK_inference(
                n_docs_completed=ttl_n_docs,
                n_docs=D,
                dist_per_doc=dist_per_doc,
                iters_per_doc=iter_per_doc,
                converged_per_doc=converged_per_doc,
                n_active_per_doc=n_active_per_doc,
                elapsed_time_sec=time.time() - start_time_sec)
            msg += "\n neg_log_p(x) %.6e" % (
                ttl_neg_log_p_x / ttl_n_tokens)
            pprint(msg)

    ttl_lb_logpdf_x_per_tok = ttl_lb_logpdf_x / ttl_n_tokens
    if return_info:
        info_dict = dict(
            theta_DK=theta_DK,
            dist_per_doc=dist_per_doc,
            iters_per_doc=iter_per_doc,
            converged_per_doc=converged_per_doc,
            n_active_per_doc=n_active_per_doc,
            )
        return ttl_lb_logpdf_x, ttl_lb_logpdf_x_per_tok, info_dict
    else:
        return ttl_lb_logpdf_x, ttl_lb_logpdf_x_per_tok
