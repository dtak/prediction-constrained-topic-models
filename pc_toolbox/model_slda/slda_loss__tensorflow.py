"""
slda_loss__tensorflow.py

Provides functions for computing loss and gradient for PC training,

Uses Tensorflow implementation under the hood.
"""

import tensorflow as tf

import numpy as np
from scipy.special import gammaln

from slda_utils__diffable_param_manager__tensorflow import (
    unflatten_to_common_param_dict)
from est_local_params__single_doc_map import (
    calc_nef_map_pi_d_K__tensorflow,
    make_convex_alpha_minus_1,
    )

def make_loss_func_and_grad_func_wrt_paramvec_and_step(
        dataset=None,
        n_batches=1,
        data_seed=42,
        dim_P=None,
        model_hyper_P=None,
        pi_frac_max_iters_first_train_lap=1.0,
        max_train_laps=None,
        **unused_kwargs):
    ''' Create and return two callable functions: one for loss, one for gradient

    Returns
    -------
    loss_func : func of two args (param_vec, step_id)
        Input:
            param_vec : numpy array, 1D
            step_id : int or None
        Output:
            float
    grad_func : func of two args (param_vec, step_id)
        Input:
            param_vec : numpy array, 1D
            step_id : int or None
        Output:
            grad_vec : numpy array, 1D
    '''
    K = int(dim_P['n_states'])
    V = int(dim_P['n_vocabs'])
    C = int(dim_P['n_labels'])
    S = K * (V - 1) + C * K  # total num params in param vec
    _param_vec = tf.Variable(
        tf.zeros([S], dtype=tf.float64))
    _n_docs = tf.placeholder(shape=[], dtype=tf.int32)
    _word_id_U = tf.placeholder(shape=[None], dtype=tf.int32)
    _word_ct_U = tf.placeholder(shape=[None], dtype=tf.float64)
    _doc_indptr_Dp1 = tf.placeholder(shape=[None], dtype=tf.int32)
    _y_DC = tf.placeholder(shape=[None, C], dtype=tf.float64)
    _y_rowmask = tf.placeholder(shape=[None], dtype=tf.int32)
    _frac_train_laps_completed = tf.placeholder(shape=(), dtype=tf.float64)
    (_loss_x, _loss_y, _loss_pi, _loss_t, _loss_w,
        _pi_DK, _y_proba_DC) = calc_loss__slda__tensorflow_graph(
        param_vec=_param_vec,
        dataset=dict(
            n_docs=_n_docs,
            n_labels=C,
            word_id_U=_word_id_U,
            word_ct_U=_word_ct_U,
            doc_indptr_Dp1=_doc_indptr_Dp1,
            y_DC=_y_DC,
            y_rowmask=_y_rowmask,
            ),
        frac_train_laps_completed=_frac_train_laps_completed,
        dim_P=dim_P,
        convex_alpha_minus_1=make_convex_alpha_minus_1(alpha=model_hyper_P['alpha']),
        tau=model_hyper_P['tau'],
        lambda_w=model_hyper_P['lambda_w'],
        weight_x=model_hyper_P['weight_x'],
        weight_y=model_hyper_P['weight_y'],
        )
    _loss_ttl = _loss_x + _loss_y + _loss_pi + _loss_t + _loss_w
    _grad_vec = tf.gradients(_loss_ttl, [_param_vec])[0]
    sess = tf.Session()
    ## BEGIN LOSS FUNC DEFN
    def loss_func(
            param_vec=None,
            step_id=None,
            **unused_kwargs):
        """ Compute loss at provided flat parameter vec

        Returns
        -------
        loss_val : float
        """
        if step_id is None or step_id < 0:
            cur_dataset = dataset
            frac_train_laps_completed = 1.0
        else:
            cur_slice = make_slice_for_step(
                step_id=step_id,
                seed=data_seed,
                n_total=dataset['n_docs'],
                n_batches=n_batches)
            cur_dataset = slda_utils__dataset_manager.slice_dataset(
                dataset=dataset,
                cur_slice=cur_slice)
            frac_train_laps_completed = np.minimum(
                1.0,
                float(step_id) / float(max_train_laps * n_batches))
        loss_ttl = sess.run([_loss_ttl],
            feed_dict={
                _param_vec:param_vec,
                _n_docs:cur_dataset['n_docs'],
                _word_id_U:cur_dataset['word_id_U'],
                _word_ct_U:cur_dataset['word_ct_U'],
                _doc_indptr_Dp1:cur_dataset['doc_indptr_Dp1'],
                _y_DC:cur_dataset['y_DC'],
                _y_rowmask:cur_dataset.get(
                    'y_rowmask',
                    np.ones(cur_dataset['n_docs'], dtype=np.int32)),
                _frac_train_laps_completed:frac_train_laps_completed,
                })
        return loss_ttl
    ## END LOSS FUNC DEFN

    ## BEGIN GRAD FUNC DEFN
    def grad_func(
            param_vec=None,
            step_id=None,
            **unused_kwargs):
        """ Compute loss at provided flat parameter vec

        Returns
        -------
        loss_val : float
        """
        if step_id is None or step_id < 0:
            cur_dataset = dataset
            frac_train_laps_completed = 1.0
        else:
            cur_slice = make_slice_for_step(
                step_id=step_id,
                seed=data_seed,
                n_total=dataset['n_docs'],
                n_batches=n_batches)
            cur_dataset = slda_utils__dataset_manager.slice_dataset(
                dataset=dataset,
                cur_slice=cur_slice)
            frac_train_laps_completed = np.minimum(
                1.0,
                float(step_id) / float(max_train_laps * n_batches))
        grad_vec = sess.run([_grad_vec],
            feed_dict={
                _param_vec:param_vec,
                _n_docs:cur_dataset['n_docs'],
                _word_id_U:cur_dataset['word_id_U'],
                _word_ct_U:cur_dataset['word_ct_U'],
                _doc_indptr_Dp1:cur_dataset['doc_indptr_Dp1'],
                _y_DC:cur_dataset['y_DC'],
                _y_rowmask:cur_dataset.get(
                    'y_rowmask',
                    np.ones(cur_dataset['n_docs'], dtype=np.int32)),
                _frac_train_laps_completed:frac_train_laps_completed,
                })
        return grad_vec
    ## END GRAD FUNC DEFN
    return loss_func, grad_func

def calc_loss__slda__tensorflow_graph(
        param_vec=None,
        dim_P=None,
        dataset=None,
        convex_alpha_minus_1=None,
        tau=1.1,
        delta=0.1,
        lambda_w=0.001,
        weight_x=1.0,
        weight_y=1.0,
        weight_pi=1.0,
        return_dict=False,
        rescale_total_loss_by_n_tokens=True,
        frac_train_laps_completed=1.0,
        pi_frac_max_iters_first_train_lap=1.0,
        pi_min_iters=DefaultDocTopicOptKwargs['pi_min_iters'],
        pi_max_iters=DefaultDocTopicOptKwargs['pi_max_iters'],
        active_proba_thr=0.005,
        **unused_kwargs):
    ''' Compute log probability of bow dataset under topic model.

    Returns
    -------
    log_proba : avg. log probability of dataset under provided LDA model.
        Scaled by number of docs in the dataset.
    '''
    # Unpack dataset
    doc_indptr_Dp1 = dataset['doc_indptr_Dp1']
    word_id_U = dataset['word_id_U']
    word_ct_U = dataset['word_ct_U']
    n_docs = dataset['n_docs']
    y_DC = dataset['y_DC']
    y_rowmask = dataset['y_rowmask']
    
    ## Unpack params
    assert param_vec is not None
    param_dict = unflatten_to_common_param_dict(param_vec, **dim_P)
    topics_KV = param_dict['topics_KV']
    w_CK = param_dict['w_CK']
    K, _ = topics_KV.get_shape().as_list()
    C, _ = w_CK.get_shape().as_list()

    ## Establish pi_opt_kwargs
    half_frac_progress = tf.minimum(1.0, 2 * frac_train_laps_completed)
    pi_min_iters = int(pi_min_iters + np.ceil(
        pi_frac_max_iters_first_train_lap * (pi_max_iters - pi_min_iters)))
    cur_pi_max_iters = tf.cast(
        pi_min_iters + tf.ceil(
            half_frac_progress * (pi_max_iters - pi_min_iters)),
        tf.int32)
    pi_opt_kwargs = dict(**DefaultDocTopicOptKwargs)
    pi_opt_kwargs['pi_max_iters'] = cur_pi_max_iters

    def has_docs_left(
            d, avg_log_proba_x, avg_log_proba_y,
            avg_log_proba_pi, pi_arr, y_arr):
        return d < n_docs
    def update_doc(
            d, avg_log_proba_x, avg_log_proba_y,
            avg_log_proba_pi, pi_arr, y_arr):
        start_d = doc_indptr_Dp1[d]
        stop_d = doc_indptr_Dp1[d+1]
        word_id_d_Ud = word_id_U[start_d:stop_d]
        word_ct_d_Ud = word_ct_U[start_d:stop_d]
        pi_d_K, topics_KUd, _, _ = \
            _calc_nef_map_pi_d_K__tensorflow_graph(
                _word_id_d_Ud=word_id_d_Ud,
                _word_ct_d_Ud=word_ct_d_Ud,
                _topics_KV=topics_KV,
                convex_alpha_minus_1=convex_alpha_minus_1,
                **pi_opt_kwargs)
        pi_arr = pi_arr.write(d, pi_d_K)
        avg_log_proba_pi_d = weight_pi * tf.reduce_sum(
            (alpha - 1.0) * tf.log(1e-9 + pi_d_K))
        avg_log_proba_x_d = tf.reduce_sum(
            word_ct_d_U * 
            tf.log(tf.matmul(tf.reshape(pi_d_K, (1,K)), topics_KUd)))
        avg_log_proba_x_d += (
            tf.lgamma(1.0 + tf.reduce_sum(word_ct_d_U))
            - tf.reduce_sum(tf.lgamma(1.0 + word_ct_d_U)))

        log_proba_y_d_C = tf.reduce_sum(
            w_CK * tf.reshape(pi_d_K, (1,K)), axis=1)
        avg_log_proba_y_d = tf.cond(
            y_rowmask[d] > 0,
            lambda: -1.0 * tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(log_proba_y_d_C, y_DC[d])),
            lambda: tf.constant(0.0, dtype=tf.float64))
        y_arr = y_arr.write(d, tf.sigmoid(log_proba_y_d_C))
        return (
            d+1,
            avg_log_proba_x + weight_x * avg_log_proba_x_d,
            avg_log_proba_y + weight_y * avg_log_proba_y_d,
            avg_log_proba_pi + avg_log_proba_pi_d,
            pi_arr,
            y_arr)

    _avg_log_proba_x = tf.constant(0.0, dtype=tf.float64)
    _avg_log_proba_y = tf.constant(0.0, dtype=tf.float64)
    _avg_log_proba_pi = tf.constant(0.0, dtype=tf.float64)
    _K = tf.cast(K, tf.float64)
    _convex_alpha_minus_1 = tf.cast(convex_alpha_minus_1, tf.float64)
    _d = 0
    _pi_arr = tf.TensorArray(dtype=tf.float64, size=n_docs) 
    _y_arr = tf.TensorArray(dtype=tf.float64, size=n_docs) 
    (_d, _avg_log_proba_x, _avg_log_proba_y, _avg_log_proba_pi,
        _pi_arr, _y_arr) = tf.while_loop(
            has_docs_left,
            update_doc,
            loop_vars=[
                _d, _avg_log_proba_x, _avg_log_proba_y, 
                _avg_log_proba_pi, _pi_arr, _y_arr])
    _pi_DK = tf.reshape(_pi_arr.concat(), (n_docs, K))
    _y_proba_DC = tf.reshape(_y_arr.concat(), (n_docs, C))

    _avg_log_proba_topics = (tau - 1.0) * tf.reduce_sum(tf.log(topics_KV))
    _avg_log_proba_w = -1.0 * (
        weight_y * lambda_w * tf.reduce_sum(tf.square(w_CK)))

    scale_ttl = tf.reduce_sum(word_ct_U)
    _avg_log_proba_x /= scale_ttl
    _avg_log_proba_pi /= scale_ttl
    _avg_log_proba_y /= scale_ttl
    _avg_log_proba_topics /= scale_ttl
    _avg_log_proba_w /= scale_ttl

    return (
        -1.0 * _avg_log_proba_x,
        -1.0 * _avg_log_proba_y,
        -1.0 * _avg_log_proba_pi,
        -1.0 * _avg_log_proba_topics,
        -1.0 * _avg_log_proba_w,
        _pi_DK,
        _y_proba_DC)
