"""
slda_loss__cython.py

Provides functions for computing loss function for PC sLDA objective.
Uses fast Cython-ized implementation.

Does NOT compute gradients, so cannot be used for training.
"""

import numpy as np
from scipy.special import gammaln

from sscape.utils_data import make_slice_for_step
from sscape.utils_io import pprint
from slda__base import (
    load_dataset,
    slice_dataset,
    init_param_dict,
    unflatten_to_param_dict,
    flatten_to_differentiable_param_vec,
    )

from calc_single_doc_pi_d_K__fast_nefmap import \
    calc_single_doc_pi_d_K
from calc_super_single_doc_pi_d_K__linesearch import \
    calc_super_single_doc_pi_d_K

from calc_topic_probas_for_single_doc import \
    calc_topic_probas_for_single_doc, DefaultDocTopicOptKwargs
from calc_topic_probas_for_many_docs import calc_pi_DK_for_many_docs
from trans_util_log_unit_interval import log_logistic_sigmoid
from trans_util_unit_interval import logistic_sigmoid
from util_make_readable_summary import make_readable_summary_for_pi_DK_inference

def extract_features(**kwargs):
    from lda_feat_extractor import extract_features as _extract
    return _extract(**kwargs)

def calc_neg_log_proba__slda(
        dataset=None,
        topics_KV=None,
        w_CK=None,
        param_vec=None,
        include_norm_const=True,
        rescale_divide_by_num_obs=True,
        rescale_y_to_match_x=False,
        weight_x=1.0,
        weight_y=1.0,
        alpha=None,
        nef_alpha=None,
        tau=None,
        delta=None,
        lambda_w=None,
        return_dict=False,
        pi_estimation_mode='missing_y',
        pi_estimation_weight_y=0.0,
        do_fast=True,
        LP=None,
        verbose=False,
        **kwargs):
    ''' Compute log probability of bow dataset under topic model.

    Returns
    -------
    log_proba : Total log probability of dataset under provided sLDA model.
        Scaled by number of word tokens in the dataset.
    '''
    # Parse smoothing parameter
    # Force to use nef formulation (always > 1.0 for convexity)
    if nef_alpha is not None:
        nef_alpha = float(nef_alpha)
    elif alpha is not None:
        nef_alpha = float(alpha)
    else:
        raise ValueError("Need to define alpha or nef_alpha")    
    alpha = None
    assert alpha is None
    assert isinstance(nef_alpha, float)
    if nef_alpha < 1.0:
        nef_alpha = nef_alpha + 1.0
    assert nef_alpha >= 1.0
    assert nef_alpha <  2.0

    if verbose:
        msg = ">>> fastloss calc_neg_log_proba %s  nef_alpha %.4f"
        msg += " w_y %.3f  pi_w_y %d x %.3f"
        pprint(msg % (
            pi_estimation_mode, nef_alpha, weight_y,
            (pi_estimation_mode == 'observe_y'), pi_estimation_weight_y))

    # cur_pi_DK is the pretrained pi_DK
    # pi_DK is something else (the one we fill in, if needed)
    cur_pi_DK = None
    if isinstance(LP, dict) and 'cur_pi_DK' in LP:
        cur_pi_DK = LP['cur_pi_DK']
    else:
        func_for_single_doc = calc_topic_probas_for_single_doc
        if do_fast:
            func_for_single_doc = calc_single_doc_pi_d_K

    # Process hyperparams
    # if they are unset, just mark as None and let later code complain
    try:
        delta = float(delta)
    except TypeError:
        delta = None
    try:
        tau = float(tau)
        if tau < 1.0:
            tau += 1.0
    except TypeError:
        tau = None
    try:
        lambda_w = float(lambda_w)
    except TypeError:
        lambda_w = None

    if param_vec is not None:
        param_dict = unflatten_to_param_dict(param_vec, **kwargs)
        topics_KV = param_dict['topics_KV']
        w_CK = param_dict['w_CK']

    n_docs = int(dataset['n_docs'])
    n_labels = int(dataset['n_labels'])
    if 'y_rowmask' in dataset:
        y_finite_DC = dataset['y_DC'][dataset['y_rowmask']==1]
        u_y_vals = np.unique(y_finite_DC.flatten())
    else:
        u_y_vals = np.unique(dataset['y_DC'].flatten())
    assert np.all(np.isfinite(u_y_vals))
    if u_y_vals.size <= 2 and np.union1d([0.0, 1.0], u_y_vals).size == 2:
        output_data_type = 'binary'
    else:
        output_data_type = 'real'
    K = w_CK.shape[1]

    avg_log_proba_x = 0.0
    avg_log_proba_y = 0.0
    avg_log_proba_pi = 0.0 

    if return_dict:
        pi_DK = np.zeros((n_docs, K))
        n_docs_converged = 0
        n_docs_restarted = 0
        dist_per_doc = np.zeros(n_docs, dtype=np.float64)
        step_size_per_doc = np.zeros(n_docs, dtype=np.float64)
        iters_per_doc = np.zeros(n_docs, dtype=np.int32)
        n_active_per_doc = np.zeros(n_docs, dtype=np.int32)
        restarts_per_doc = np.zeros(n_docs, dtype=np.int32)
    if return_dict and weight_y > 0:
        y_proba_DC = np.zeros((n_docs, n_labels))

    # Skip norm const
    # avg_log_proba_pi = n_docs * (
    #    gammaln(K * alpha) - K * gammaln(alpha))
    for d in xrange(n_docs):
        start_d = dataset['doc_indptr_Dp1'][d]
        stop_d = dataset['doc_indptr_Dp1'][d+1]
        word_id_d_U = dataset['word_id_U'][start_d:stop_d]
        word_ct_d_U = dataset['word_ct_U'][start_d:stop_d]

        # Semi-supervised case: some examples have unknown labels
        y_d_is_missing = \
            'y_rowmask' in dataset and dataset['y_rowmask'][d] == 0

        if cur_pi_DK is not None:
            if d == 0: pprint('using precomputed cur_pi_DK from LP')
            pi_d_K = cur_pi_DK[d]
            info_dict = dict(
                n_restarts=0,
                did_converge=0,
                pi_step_size=0.0,
                n_iters=0)
        elif pi_estimation_mode == 'missing_y' or y_d_is_missing:
            # THIS CALL IS ALWAYS UNSUPERVISED!
            pi_d_K, info_dict = \
                func_for_single_doc(
                    word_id_d_U,
                    word_ct_d_U,
                    topics_KV=topics_KV,
                    alpha=None,
                    nef_alpha=nef_alpha,
                    **kwargs)
        elif pi_estimation_mode == 'observe_y':
            # THIS CALL IS SUPERVISED!
            pi_d_K, info_dict = \
                calc_super_single_doc_pi_d_K(
                    word_id_d_U,
                    word_ct_d_U,
                    topics_KV=topics_KV,
                    alpha=None,
                    nef_alpha=nef_alpha,
                    y_d_C=np.asarray(dataset['y_DC'][d], dtype=np.int32),
                    w_CK=w_CK,
                    weight_y=float(pi_estimation_weight_y),
                    **kwargs)
        else:
            raise ValueError("Unrecognized pi_estimation_mode: %s" % (
                pi_estimation_mode))

        if return_dict:
            pi_DK[d] = pi_d_K
            n_active_per_doc[d] = np.sum(pi_d_K > 0.005)
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

        log_proba_x_d = np.inner(
            word_ct_d_U,
            np.log(np.dot(pi_d_K, topics_KV[:, word_id_d_U])))
        log_proba_x_d += \
            gammaln(1.0 + np.sum(word_ct_d_U)) - \
            np.sum(gammaln(1.0 + word_ct_d_U))
        avg_log_proba_x += weight_x * log_proba_x_d
            
        avg_log_proba_pi += np.sum(
            (nef_alpha - 1.0) * np.log(1e-9 + pi_d_K))
        if y_d_is_missing:
            y_proba_DC[d] = np.nan
            continue
        if weight_y > 0 and output_data_type == 'binary':
            if rescale_y_to_match_x:
                weight_y_d = weight_y * np.sum(word_ct_d_U)
            else:
                weight_y_d = weight_y
            y_d_C = dataset['y_DC'][d]
            sign_y_d_C = np.sign(y_d_C - 0.01)
            log_proba_y_d_C = log_logistic_sigmoid(
                sign_y_d_C * np.dot(w_CK, pi_d_K))
            log_proba_y_d = np.sum(log_proba_y_d_C)
            avg_log_proba_y += weight_y_d * log_proba_y_d

            if return_dict:
                proba_y_eq_1_d_C = logistic_sigmoid(
                    np.dot(w_CK, pi_d_K))
                y_proba_DC[d] = proba_y_eq_1_d_C
        if weight_y > 0 and output_data_type == 'real':
            y_d_C = dataset['y_DC'][d]
            y_est_d_C = np.dot(w_CK, pi_d_K)
            log_proba_y_d = -0.5 / delta * np.sum(np.square(y_est_d_C - y_d_C))
            avg_log_proba_y += weight_y * log_proba_y_d
            if return_dict:
                y_proba_DC[d] = y_est_d_C
    # topics_KV is guaranteed to be >= min_eps
    log_proba_topics = \
        (tau - 1) * np.sum(np.log(topics_KV))
    log_proba_w = \
        -1.0 * weight_y * lambda_w * np.sum(np.square(w_CK))
    scale_y = 1.0
    if rescale_divide_by_num_obs:
        scale = float(np.sum(dataset['word_ct_U']))
        _, C = dataset['y_DC'].shape
        avg_log_proba_x = avg_log_proba_x / scale
        avg_log_proba_pi = avg_log_proba_pi / scale
        log_proba_topics /= scale
        log_proba_w /= scale

        if 'y_rowmask' in dataset:
            n_y_docs = 1e-10 + float(np.sum(dataset['y_rowmask']))
        else:
            n_y_docs = float(n_docs)
        if rescale_y_to_match_x:
            scale_y = 1.0
            avg_log_proba_y = avg_log_proba_y / scale
        else:
            scale_y = float(C * n_y_docs) / scale
            avg_log_proba_y = avg_log_proba_y / (C * n_y_docs)

    if return_dict:
        ans_dict = dict(
            loss_x=-1.0 * avg_log_proba_x,
            loss_y=-1.0 * avg_log_proba_y,
            loss_pi=-1.0 * avg_log_proba_pi,
            loss_topics=-1.0 * log_proba_topics,
            loss_w=-1.0 * log_proba_w,
            scale=scale,
            scale_y=scale_y,
            pi_DK=pi_DK,
            output_data_type=output_data_type,
            summary_msg=make_readable_summary_for_pi_DK_inference(
                n_docs=n_docs,
                n_docs_completed=n_docs,
                n_docs_converged=n_docs_converged,
                n_docs_restarted=n_docs_restarted,
                iters_per_doc=iters_per_doc,
                n_active_per_doc=n_active_per_doc,
                dist_per_doc=dist_per_doc,
                restarts_per_doc=restarts_per_doc,
                step_size_per_doc=step_size_per_doc,
                ),
            iters_per_doc=iters_per_doc,
            n_active_per_doc=n_active_per_doc,
            dist_per_doc=dist_per_doc,
            restarts_per_doc=restarts_per_doc,
            step_size_per_doc=step_size_per_doc,
            )
        ans_dict['loss_all'] = 0.0
        for key in ans_dict:
            if key.startswith('loss'):
                skey = 'scale_' + key.replace('loss_', '')
                if skey in ans_dict:
                    ans_dict['loss_all'] += ans_dict[skey] * ans_dict[key]
                else:
                    ans_dict['loss_all'] += ans_dict[key]                    
        if return_dict and weight_y > 0:
            ans_dict['y_proba_DC'] = y_proba_DC
        return ans_dict
    else:
        return -1.0 * (
            avg_log_proba_x + avg_log_proba_pi 
            + scale_y * avg_log_proba_y
            + log_proba_topics
            + log_proba_w
            )
