"""
slda_loss__cython.py

Provides functions for computing loss function for PC sLDA objective.
Uses fast Cython-ized implementation of the local (per-document) step.

Does NOT compute gradients (not autodiff-able), so cannot be used for training.
"""

import numpy as np
from scipy.special import gammaln
import time

from pc_toolbox.utils_diffable_transforms import (
    log_logistic_sigmoid,
    logistic_sigmoid,
    )
from est_local_params__single_doc_map import (
    calc_nef_map_pi_d_K,
    make_convex_alpha_minus_1,
    DefaultDocTopicOptKwargs)
from est_local_params__many_doc_map import (
    make_readable_summary_for_pi_DK_estimation,
    )

def calc_loss__slda(
        dataset=None,
        topics_KV=None,
        w_CK=None,
        alpha=None,
        nef_alpha=1.1,
        tau=1.1,
        delta=0.1,
        lambda_w=0.001,
        weight_x=1.0,
        weight_y=1.0,
        weight_pi=1.0,
        return_dict=False,
        rescale_total_loss_by_n_tokens=True,
        pi_estimation_mode='missing_y',
        active_proba_thr=0.005,
        fixed_pi_DK=None,
        **pi_opt_kwargs):
    ''' Compute loss of provided dataset under sLDA topic model.

    Returns
    -------
    loss : float
        Total loss (-1 * log proba.) of dataset under provided sLDA model.
        By default, rescaled by number of word tokens in the dataset.
    '''
    # Parse hyperparams
    delta = float(delta)
    tau = float(tau)
    lambda_w = float(lambda_w)
    convex_alpha_minus_1 = make_convex_alpha_minus_1(alpha=alpha, nef_alpha=nef_alpha)
    assert convex_alpha_minus_1 >= 0.0
    assert convex_alpha_minus_1 < 1.0

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


    if return_dict:
        start_time_sec = time.time()
        pi_DK = np.zeros((n_docs, K))
        n_docs_converged = 0
        n_docs_restarted = 0
        iters_per_doc = np.zeros(n_docs, dtype=np.int32)
        dist_per_doc = np.zeros(n_docs, dtype=np.float32)
        step_size_per_doc = np.zeros(n_docs, dtype=np.float32)
        n_active_per_doc = np.zeros(n_docs, dtype=np.int32)
        restarts_per_doc = np.zeros(n_docs, dtype=np.int32)
        y_proba_DC = np.zeros((n_docs, n_labels))

    # Aggregators for different loss terms
    loss_x = 0.0
    loss_y = 0.0
    loss_pi = 0.0 
    for d in xrange(n_docs):
        start_d = dataset['doc_indptr_Dp1'][d]
        stop_d = dataset['doc_indptr_Dp1'][d+1]
        word_id_d_U = dataset['word_id_U'][start_d:stop_d]
        word_ct_d_U = dataset['word_ct_U'][start_d:stop_d]

        # Semi-supervised case: some examples have unknown labels
        y_d_is_missing = \
            'y_rowmask' in dataset and dataset['y_rowmask'][d] == 0

        if fixed_pi_DK is not None:
            if d == 0: pprint('using precomputed cur_pi_DK from LP')
            pi_d_K = cur_pi_DK[d]
            info_d = dict(
                n_restarts=0,
                did_converge=0,
                cur_L1_diff=0.0,
                pi_step_size=0.0,
                n_iters=0)
        elif pi_estimation_mode == 'missing_y' or y_d_is_missing:
            # THIS CALL IS ALWAYS UNSUPERVISED!
            pi_d_K, info_d = \
                calc_nef_map_pi_d_K(
                    word_id_d_U,
                    word_ct_d_U,
                    topics_KV=topics_KV,
                    convex_alpha_minus_1=convex_alpha_minus_1,
                    method='cython',
                    **pi_opt_kwargs)
        elif pi_estimation_mode == 'observe_y':
            # THIS CALL IS SUPERVISED!
            pi_d_K, info_d = \
                calc_super_nef_map_pi_d_K__cython(
                    word_id_d_U,
                    word_ct_d_U,
                    topics_KV=topics_KV,
                    convex_alpha_minus_1=convex_alpha_minus_1,
                    y_d_C=np.asarray(dataset['y_DC'][d], dtype=np.int32),
                    w_CK=w_CK,
                    weight_y=float(pi_estimation_weight_y),
                    **pi_opt_kwargs)
        else:
            raise ValueError(
                "Unrecognized pi_estimation_mode: %s" % (
                    pi_estimation_mode))

        if return_dict:
            pi_DK[d] = pi_d_K
            n_active_per_doc[d] = np.sum(pi_d_K > active_proba_thr)
            n_docs_restarted += info_d['n_restarts'] > 0
            n_docs_converged += info_d['did_converge']
            iters_per_doc[d] = info_d['n_iters']
            step_size_per_doc[d] = info_d['pi_step_size']
            dist_per_doc[d] = info_d.get('cur_L1_diff', -1.0)
            restarts_per_doc[d] = info_d.get('n_restarts', -1)
        
        # LOSS due to x
        logpdf_x_d = np.inner(
            word_ct_d_U,
            np.log(np.dot(pi_d_K, topics_KV[:, word_id_d_U])))
        logpdf_x_d += \
            gammaln(1.0 + np.sum(word_ct_d_U)) - \
            np.sum(gammaln(1.0 + word_ct_d_U))
        loss_x -= weight_x * logpdf_x_d

        # LOSS due to pi
        loss_pi -= weight_pi * np.sum(
            (convex_alpha_minus_1) * np.log(1e-9 + pi_d_K))
            

        # Semi-supervised case: skip examples with unknown labels
        if y_d_is_missing:
            y_proba_DC[d] = np.nan
            continue
        if weight_y > 0 and output_data_type == 'binary':
            y_d_C = dataset['y_DC'][d]
            sign_y_d_C = np.sign(y_d_C - 0.01)
            logpdf_y_d = np.sum(log_logistic_sigmoid(
                sign_y_d_C * np.dot(w_CK, pi_d_K)))
            loss_y -= weight_y * logpdf_y_d
            if return_dict:
                proba_y_eq_1_d_C = logistic_sigmoid(
                    np.dot(w_CK, pi_d_K))
                y_proba_DC[d] = proba_y_eq_1_d_C
        if weight_y > 0 and output_data_type == 'real':
            y_d_C = dataset['y_DC'][d]
            y_est_d_C = np.dot(w_CK, pi_d_K)
            logpdf_y_d = -0.5 / delta * np.sum(
                np.square(y_est_d_C - y_d_C))
            loss_y -= weight_y * logpdf_y_d
            if return_dict:
                y_proba_DC[d] = y_est_d_C
    # ... end loop over docs

    # GLOBAL PARAM REGULARIZATION TERMS
    # Loss for topic-word params
    loss_topics = \
        -1.0 * (tau - 1) * np.sum(np.log(topics_KV))

    # Loss for regression weights
    # Needs to scale with weight_y so lambda_w doesnt grow as weight_y grows
    loss_w = \
        float(weight_y) * lambda_w * np.sum(np.square(w_CK))

    # RESCALING LOSS TERMS
    if rescale_total_loss_by_n_tokens:
        scale_ttl = float(np.sum(dataset['word_ct_U']))
    else:
        scale_ttl = 1.0
    loss_x /= scale_ttl
    loss_pi /= scale_ttl
    loss_topics /= scale_ttl
    loss_w /= scale_ttl
    loss_y /= scale_ttl
    # TOTAL LOSS
    loss_ttl = loss_x + loss_y + loss_pi + loss_topics + loss_w

        
    if return_dict:
        # Compute unweighted loss
        uw_x = np.maximum(weight_x, 1.0)
        uloss_x__pertok = loss_x * scale_ttl / float(
            uw_x * np.sum(dataset['word_ct_U']))

        uw_y = np.maximum(weight_y, 1.0)
        n_y_docs, C = dataset['y_DC'].shape
        n_y_docs = 1e-10 + float(n_y_docs)
        if 'y_rowmask' in dataset:
            n_y_docs = 1e-10 + float(np.sum(dataset['y_rowmask']))
        uloss_y__perdoc = loss_y * scale_ttl / float(
            uw_y * C * n_y_docs)

        ans_dict = dict(
            loss_ttl=loss_ttl,
            loss_x=loss_x,
            loss_y=loss_y,
            loss_pi=loss_pi,
            loss_topics=loss_topics,
            loss_w=loss_w,
            rescale_total_loss_by_n_tokens=rescale_total_loss_by_n_tokens,
            uloss_x__pertok=uloss_x__pertok,
            uloss_y__perdoc=uloss_y__perdoc,
            output_data_type=output_data_type,
            pi_DK=pi_DK,
            y_proba_DC=y_proba_DC,
            n_docs_converged=n_docs_converged,
            n_docs_restarted=n_docs_restarted,
            iters_per_doc=iters_per_doc,
            dist_per_doc=dist_per_doc,
            step_size_per_doc=step_size_per_doc,
            n_active_per_doc=n_active_per_doc,
            summary_msg=make_readable_summary_for_pi_DK_estimation(
                elapsed_time_sec=time.time() - start_time_sec,
                n_docs=n_docs,
                n_docs_converged=n_docs_converged,
                n_docs_restarted=n_docs_restarted,
                iters_per_doc=iters_per_doc,
                dist_per_doc=dist_per_doc,
                step_size_per_doc=step_size_per_doc,
                restarts_per_doc=restarts_per_doc,
                n_active_per_doc=n_active_per_doc,
                **pi_opt_kwargs),
            )
        return ans_dict
    else:
        return loss_ttl


if __name__ == '__main__':
    import os
    from sklearn.externals import joblib
    from slda_utils__dataset_manager import load_dataset

    # Simplest possible test
    # Load the toy bars dataset
    # Load "true" bars topics
    # Compute the loss
    dataset_path = os.path.expandvars("$PC_REPO_DIR/datasets/toy_bars_3x3/")
    dataset = load_dataset(dataset_path, split_name='train')

    # Load "true" 4 bars
    GP = joblib.load(
        os.path.join(dataset_path, "good_loss_x_K4_param_dict.dump"))
    topics_KV = GP['topics_KV']
    w_CK = GP['w_CK']

    loss_dict = calc_loss__slda(
        dataset=dataset,
        topics_KV=topics_KV,
        w_CK=w_CK,
        nef_alpha=1.1,
        tau=1.1,
        lambda_w=0.001,
        return_dict=True)
    print(loss_dict['summary_msg'])