import numpy as np
import time

from pc_toolbox.model_slda.est_local_params__single_doc_map import (
    calc_nef_map_pi_d_K,
    DefaultDocTopicOptKwargs,
    )

from pc_toolbox.utils_io import (
    pprint,
    make_percentile_str)

from utils_summarize_pi_DK_estimation import (
    make_readable_summary_for_pi_DK_estimation)

def calc_nef_map_pi_DK(
        dataset=None,
        topics_KV=None,
        alpha=None,
        nef_alpha=None,
        init_pi_DK=None,
        n_seconds_between_print=-1,
        active_proba_thr=0.005,
        return_info=False,
        calc_pi_d_K=calc_nef_map_pi_d_K,
        **some_pi_estimation_kwargs):
    ''' Extract doc-topic probability features for every doc in dataset.

    Args
    ----
    dataset : dict with array fields
        'n_docs' : int, non-negative
            number of documents in dataset
        'word_id_U' : 1D array, size U, dtype=int
            vocab ids for each doc-term pair in dataset
        'word_ct_U' : 1D array, size U, dtype=float
            counts for each doc-term pair in dataset
        'doc_indptr_Dp1' : 1D array, size D+1, type=int
            indptr / fenceposts delineating where individual docs begin/end
    topics_KV : 2D array, size K x V, rows sum to one
        probability of each word v appearing under each topic k
    alpha : float, positive value
        concentration parameter of Dirichlet prior on doc-topic probas
    
    Returns
    -------
    pi_DK : 2D array, size D x K
        Each row has positive entries and sums to one.
    info_dict : dict
        Only returned if called with return_info=True
    '''
    # Parse pi estimation kwargs
    pi_estimation_kwargs = dict(**DefaultDocTopicOptKwargs)
    for key in pi_estimation_kwargs.keys():
        if key in some_pi_estimation_kwargs:
            val = DefaultDocTopicOptKwargs[key]
            if isinstance(val, float):
                pi_estimation_kwargs[key] = float(some_pi_estimation_kwargs[key])
            else:
                pi_estimation_kwargs[key] = int(some_pi_estimation_kwargs[key])

    assert topics_KV is not None
    K = int(topics_KV.shape[0])

    n_docs = dataset['n_docs']
    doc_indptr_Dp1 = dataset['doc_indptr_Dp1']
    word_id_U = dataset['word_id_U']
    word_ct_U = dataset['word_ct_U']

    pi_DK = np.zeros((n_docs, K))   
    n_docs_converged = 0
    n_docs_restarted = 0
    iters_per_doc = np.zeros(n_docs, dtype=np.int32)
    n_active_per_doc = np.zeros(n_docs, dtype=np.int32)
    restarts_per_doc = np.zeros(n_docs, dtype=np.int32)
    step_size_per_doc = np.zeros(n_docs, dtype=np.float32)
    dist_per_doc = np.zeros(n_docs, dtype=np.float32)
    loss_per_doc = np.zeros(n_docs, dtype=np.float32)

    is_time = False
    start_time_sec = time.time()
    last_print_sec = start_time_sec
    for d in xrange(n_docs):
        start_d = doc_indptr_Dp1[d]
        stop_d = doc_indptr_Dp1[d+1]

        if init_pi_DK is None:
            init_pi_d_K = None
        else:
            init_pi_d_K = init_pi_DK[d]

        # MCH: Cannot autograd when doing this kind of assignment
        pi_DK[d,:], info_dict = \
            calc_pi_d_K(
                word_id_U[start_d:stop_d],
                word_ct_U[start_d:stop_d],
                topics_KV=topics_KV,
                alpha=alpha,
                nef_alpha=nef_alpha,
                init_pi_d_K=init_pi_d_K,
                **pi_estimation_kwargs)
        if return_info or n_seconds_between_print > 0:
            n_active_per_doc[d] = \
                np.sum(pi_DK[d,:] > active_proba_thr)
            n_docs_restarted += info_dict['n_restarts'] > 0
            n_docs_converged += info_dict['did_converge']
            iters_per_doc[d] = info_dict['n_iters']
            step_size_per_doc[d] = info_dict['pi_step_size']
            try:
                dist_per_doc[d] = info_dict['cur_L1_diff']
            except KeyError:
                dist_per_doc = None
            try:
                restarts_per_doc[d] = info_dict['n_restarts']
            except KeyError:
                restarts_per_doc = None
            try:
                loss_per_doc[d] = info_dict['loss']
            except KeyError:
                pass

            cur_time_sec = time.time()
            if n_seconds_between_print > 0:
                is_time = cur_time_sec - last_print_sec > n_seconds_between_print
            is_last = (d + 1) == n_docs
            if is_last or is_time:
                msg = make_readable_summary_for_pi_DK_estimation(
                    elapsed_time_sec=cur_time_sec - start_time_sec,
                    n_docs=n_docs,
                    n_docs_completed=d+1,
                    n_docs_converged=n_docs_converged,
                    n_docs_restarted=n_docs_restarted,
                    iters_per_doc=iters_per_doc,
                    n_active_per_doc=n_active_per_doc,
                    dist_per_doc=dist_per_doc,
                    restarts_per_doc=restarts_per_doc,
                    step_size_per_doc=step_size_per_doc,
                    loss_per_doc=loss_per_doc)

                last_print_sec = cur_time_sec
                if n_seconds_between_print > 0:
                    pprint(msg)
    if return_info:
        agg_info_dict = dict(
            summary_msg=msg,
            iters_per_doc=iters_per_doc,
            n_active_per_doc=n_active_per_doc,
            dist_per_doc=dist_per_doc,
            restarts_per_doc=restarts_per_doc,
            step_size_per_doc=step_size_per_doc,
            loss_per_doc=loss_per_doc,
            loss=np.sum(loss_per_doc),
            alpha=alpha)
        return pi_DK, agg_info_dict
    else:
        return pi_DK
