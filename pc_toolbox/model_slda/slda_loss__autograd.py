"""
slda_loss__autograd.py

Provides functions for computing loss and gradient for PC training.
"""

import autograd
import autograd.numpy as np
from autograd.scipy.special import gammaln

from pc_toolbox.utils_data import make_slice_for_step
from pc_toolbox.utils_diffable_transforms import (
    log_logistic_sigmoid,
    logistic_sigmoid,
    )

from slda_utils__dataset_manager import (
    load_dataset,
    slice_dataset,
    )
#from slda_utils__param_manager import (
#    init_param_dict,
#    unflatten_to_param_dict,
#    flatten_to_differentiable_param_vec,
#    )
from est_local_params__single_doc_map import (
    calc_nef_map_pi_d_K__numpy,
    DefaultDocTopicOptKwargs)
#from est_local_params__many_doc_map import (
#    make_readable_summary_for_pi_DK_inference)

def make_loss_func_and_grad_func(
        dataset=None,
        n_batches=1,
        seed=42,
        n_laps=None,
        alg_state_kwargs=None,
        **kwargs):
    '''
    '''
    def _make_slice(step_id):
        return make_slice_for_step(
            step_id=step_id,
            seed=seed,
            n_total=dataset['n_docs'],
            n_batches=n_batches)
    def loss_func(
            param_vec=None,
            step_id=0,
            dataset=dataset,
            cur_slice=None,
            **input_kwargs):
        if alg_state_kwargs is not None and isinstance(step_id, int):
            step_id = alg_state_kwargs['cur_step']
        for key in kwargs:
            if key not in input_kwargs:
                input_kwargs[key] = kwargs[key]
        if cur_slice is None:
            if isinstance(step_id, int):
                cur_slice = _make_slice(step_id)
            else:
                cur_slice = None
        cur_dataset = slice_dataset(
            dataset=dataset,
            cur_slice=cur_slice)
        if n_laps is None or step_id is None or n_laps == 0:
            frac_progress = 1.0
        else:
            frac_progress = float(step_id) / float(n_laps * n_batches)
        return calc_neg_logpdf__slda(
            param_vec=param_vec,
            dataset=cur_dataset,
            frac_progress=frac_progress,
            **input_kwargs)
    _grad_func = autograd.grad(loss_func)
    def grad_func(*args, **input_kwargs):
        for key in kwargs:
            if key not in input_kwargs:
                input_kwargs[key] = kwargs[key]
        try:
            return_dict = input_kwargs['return_dict']
            del input_kwargs['return_dict']
        except KeyError:
            return_dict = False
        grad_vec = _grad_func(
            *args, **input_kwargs)
        if return_dict:
            input_kwargs['return_dict'] = True
            extra_dict = loss_func(
                *args, **input_kwargs)
            input_kwargs['return_dict'] = False
            return grad_vec, extra_dict
        else:
            return grad_vec
    return loss_func, grad_func

def calc_neg_logpdf__slda(
        param_vec=None,
        dataset=None,
        topics_KV=None,
        w_CK=None,
        weight_x=1.0,
        weight_y=1.0,
        weight_pi=1.0,
        alpha=None,
        nef_alpha=None,
        tau=1.0,
        delta=0.1,
        lambda_w=0.000001,
        return_dict=False,
        rescale_total_loss_by_n_tokens=True,
        pi_estimation_mode='missing_y',
        frac_progress=1.0,
        quality_of_init=1.0,
        pi_min_iters=DefaultDocTopicOptKwargs['min_pi_max_iters'],
        pi_max_iters=DefaultDocTopicOptKwargs['max_iters'],
        **kwargs):
    ''' Compute log probability of provided dataset under sLDA model.

    Returns
    -------
    logpdf : Total log probability of dataset under provided sLDA model.
        Scaled by number of word tokens in the dataset.
    '''
    delta = float(delta)
    tau = float(tau)
    lambda_w = float(lambda_w)
    convex_alpha_minus_1 = make_convex_alpha(alpha=alpha, nef_alpha=nef_alpha)
    assert convex_alpha_minus_1 >= 0.0
    assert convex_alpha_minus_1 < 1.0
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

    if return_dict:
        pi_DK = np.zeros((n_docs, K))
        n_docs_converged = 0
        n_docs_restarted = 0
        step_size_per_doc = np.zeros(n_docs, dtype=np.int32)
        iters_per_doc = np.zeros(n_docs, dtype=np.int32)
    if return_dict and weight_y > 0:
        y_proba_DC = np.zeros((n_docs, n_labels))

    # Establish pi_opt_kwargs
    half_frac_progress = np.minimum(1.0, 2*frac_progress)
    pi_min_iters = int(pi_min_iters + np.ceil(
        quality_of_init * (pi_max_iters - pi_min_iters)))
    cur_pi_max_iters = int(pi_min_iters + np.ceil(
        half_frac_progress * (pi_max_iters - pi_min_iters)))
    pi_opt_kwargs = dict(**DefaultDocTopicOptKwargs)
    pi_opt_kwargs['max_iters'] = cur_pi_max_iters

    # Aggregators for different loss terms
    logpdf_x = 0.0
    logpdf_y = 0.0
    logpdf_pi = 0.0 
    assert pi_estimation_mode == 'missing_y'
    for d in xrange(n_docs):
        start_d = dataset['doc_indptr_Dp1'][d]
        stop_d = dataset['doc_indptr_Dp1'][d+1]
        word_id_d_U = dataset['word_id_U'][start_d:stop_d]
        word_ct_d_U = dataset['word_ct_U'][start_d:stop_d]

        pi_d_K, info_dict = \
            calc_nef_map_pi_d_K__numpy(
                word_id_d_U,
                word_ct_d_U,
                topics_KV=topics_KV,
                convex_alpha_minus_1=convex_alpha_minus_1,
                **pi_opt_kwargs)

        if return_dict:
            pi_DK[d] = pi_d_K
            n_docs_restarted += info_dict['n_restarts'] > 0
            n_docs_converged += info_dict['did_converge']
            step_size_per_doc[d] = info_dict['pi_step_size']
            iters_per_doc[d] = info_dict['n_iters']
        if weight_x > 0:
            logpdf_x_d = np.inner(
                word_ct_d_U,
                np.log(np.dot(pi_d_K, topics_KV[:, word_id_d_U])))
            logpdf_x_d += \
                gammaln(1.0 + np.sum(word_ct_d_U)) - \
                np.sum(gammaln(1.0 + word_ct_d_U))
            logpdf_x += weight_x * logpdf_x_d
            
        if weight_pi > 0:
            logpdf_pi += weight_pi * np.sum(
                (alpha_pdf - 1.0) * np.log(1e-9 + pi_d_K))

        # Semi-supervised case: skip examples with unknown labels
        if 'y_rowmask' in dataset and dataset['y_rowmask'][d] == 0:
            continue
        if weight_y > 0 and output_data_type == 'binary':
            y_d_C = dataset['y_DC'][d]
            sign_y_d_C = np.sign(y_d_C - 0.01)
            logpdf_y_d = np.sum(log_logistic_sigmoid(
                sign_y_d_C * np.dot(w_CK, pi_d_K)))
            logpdf_y += weight_y * logpdf_y_d
            if return_dict:
                proba_y_eq_1_d_C = logistic_sigmoid(
                    np.dot(w_CK, pi_d_K))
                y_proba_DC[d] = proba_y_eq_1_d_C
        if weight_y > 0 and output_data_type == 'real':
            y_d_C = dataset['y_DC'][d]
            y_est_d_C = np.dot(w_CK, pi_d_K)
            logpdf_y_d = -0.5 / delta * np.sum(
                np.square(y_est_d_C - y_d_C))
            logpdf_y += weight_y * logpdf_y_d
            if return_dict:
                y_proba_DC[d] = y_est_d_C

    # GLOBAL PARAM REGULARIZATION TERMS
    # Loss for topic-word params
    logpdf_topics = \
        (tau - 1) * np.sum(np.log(topics_KV))
    # Loss for regression weights
    logpdf_w = \
        -1.0 * lambda_w * np.sum(np.square(w_CK))
    weight_w = float(weight_y)

    # RESCALING LOSS TERMS
    if rescale_total_loss_by_n_tokens:
        scale_ttl = float(np.sum(dataset['word_ct_U']))
        logpdf_x /= scale_ttl
        logpdf_pi /= scale_ttl
        logpdf_topics /= scale_ttl
        logpdf_w /= scale_ttl

        n_y_docs, C = dataset['y_DC'].shape
        n_y_docs = float(n_y_docs) + 1e-10
        if 'y_rowmask' in dataset:
            n_y_docs = 1e-10 + float(np.sum(dataset['y_rowmask']))
            logpdf_y__perdoc_perlabel = logpdf_y / float(C * n_y_docs)
        logpdf_y = logpdf_y / scale_ttl

    loss_ttl = -1.0 * (
        logpdf_x
        + logpdf_pi 
        + logpdf_y
        + weight_w * logpdf_w
        + logpdf_topics
        )
    if return_dict:
        ans_dict = dict(
            loss_ttl=loss_ttl,
            loss_x=-1.0 * logpdf_x,
            loss_y=-1.0 * logpdf_y,
            loss_pi=-1.0 * logpdf_pi,
            loss_topics=-1.0 * logpdf_topics,
            loss_w=-1.0 * logpdf_w,
            weight_w=weight_w,
            loss_y__perdoc_perlabel=-1.0 * logpdf_y__perdoc_perlabel,
            pi_DK=pi_DK,
            n_docs_converged=n_docs_converged,
            n_docs_restarted=n_docs_restarted,
            iters_per_doc=iters_per_doc,
            step_size_per_doc=step_size_per_doc,
            summary_msg=make_readable_summary_for_pi_DK_inference(
                n_docs=n_docs,
                n_docs_converged=n_docs_converged,
                n_docs_restarted=n_docs_restarted,
                iters_per_doc=iters_per_doc),
            output_data_type=output_data_type,
            )
        if return_dict and weight_y > 0:
            ans_dict['y_proba_DC'] = y_proba_DC
        return ans_dict
    else:
        return loss_ttl

