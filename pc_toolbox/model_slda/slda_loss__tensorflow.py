"""
slda_loss__tensorflow.py

Provides functions for computing loss and gradient for PC training,
using Tensorflow implementation.
"""

import numpy as np
import tensorflow as tf


def make_loss_func_and_grad_func(
        dataset=None,
        weight_x=1.0,
        weight_y=1.0,
        alg_state_kwargs=None,
        n_batches=1,
        n_laps=None,
        **kwargs):
    '''
    '''
    K = int(kwargs['n_states'])
    V = int(kwargs['n_vocabs'])
    C = int(kwargs['n_labels'])
    S = K * (V - 1) + C * K
    _param_vec = tf.Variable(
        tf.zeros([S], dtype=tf.float64))
    _n_docs = tf.placeholder(shape=[], dtype=tf.int32)
    _word_id_U = tf.placeholder(shape=[None], dtype=tf.int32)
    _word_ct_U = tf.placeholder(shape=[None], dtype=tf.float64)
    _doc_indptr_Dp1 = tf.placeholder(shape=[None], dtype=tf.int32)
    _y_DC = tf.placeholder(shape=[None, C], dtype=tf.float64)
    _y_rowmask = tf.placeholder(shape=[None], dtype=tf.int32)
    _weight_x = tf.placeholder(shape=(), dtype=tf.float64)
    _weight_y = tf.placeholder(shape=(), dtype=tf.float64)
    _frac_progress = tf.placeholder(shape=(), dtype=tf.float32)
    (_loss_x, _loss_y, _scale_y, _loss_pi, 
        _loss_t, _loss_w, _pi_DK, _y_proba_DC) = calc_neg_log_proba__slda(
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
        weight_x=_weight_x,
        weight_y=_weight_y,
        frac_progress=_frac_progress,
        return_extra=True,
        **kwargs)
    _loss_all = (
        _loss_x 
        + (_scale_y * _loss_y)
        + _loss_pi
        + _loss_t
        + tf.sign(_weight_y) * _loss_w
        )
    _grad_of_loss = tf.gradients(_loss_all, [_param_vec])[0]
    sess = tf.Session()

    def _make_slice(step_id=0):
        return sscape.utils_data.make_slice_for_step(
            step_id=step_id,
            n_total=dataset['n_docs'],
            n_batches=n_batches,
            **kwargs)
    def loss_func(
            param_vec=None,
            step_id=None,
            dataset=dataset,
            cur_slice=None,
            return_dict=False,
            weight_x=weight_x,
            weight_y=weight_y,
            **loss_kwargs):
        if alg_state_kwargs is not None and isinstance(step_id, int):
            step_id = alg_state_kwargs['cur_step']
        if n_laps is None or step_id is None or n_laps == 0:
            frac_progress = 1.0
        else:
            frac_progress = float(step_id) / float(n_laps * n_batches)
        
        if param_vec is None:
            param_vec = flatten_to_differentiable_param_vec(**loss_kwargs)
        if cur_slice is None and isinstance(step_id, int):
            cur_slice = _make_slice(step_id)
        cur_dataset = slice_dataset(
            dataset=dataset,
            cur_slice=cur_slice)
        (loss_all, loss_x, loss_y, scale_y, loss_pi, loss_t, loss_w, 
            pi_DK, y_proba_DC) = sess.run(
            [_loss_all, _loss_x, _loss_y, _scale_y,
             _loss_pi, _loss_t, _loss_w, _pi_DK, _y_proba_DC],
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
                _weight_x:weight_x,
                _weight_y:weight_y,
                _frac_progress:frac_progress,
                })
        if return_dict:
            return dict(
                loss_all=loss_all,
                loss_x=loss_x,
                loss_y=loss_y,
                scale_y=scale_y,
                loss_pi=loss_pi,
                loss_topics=loss_t,
                loss_w=loss_w,
                pi_DK=pi_DK,
                y_proba_DC=y_proba_DC)
        else:
            return loss_all

    #_, second_grad_func = make_secondary_loss_func_and_grad_func(
    #    dataset=dataset, **kwargs)
    def grad_func(
            param_vec=None,
            step_id=None,
            dataset=dataset,
            cur_slice=None,
            weight_x=weight_x,
            weight_y=weight_y,
            return_dict=False,
            **loss_kwargs):
        if alg_state_kwargs is not None and isinstance(step_id, int):
            step_id = alg_state_kwargs['cur_step']
        if n_laps is None or step_id is None or n_laps == 0:
            frac_progress = 1.0
        else:
            frac_progress = float(step_id) / float(n_laps * n_batches)
        if cur_slice is None and isinstance(step_id, int):
            cur_slice = _make_slice(step_id)
        cur_dataset = slice_dataset(
            dataset=dataset,
            cur_slice=cur_slice)
        grad_vec, pi_DK = sess.run(
            [_grad_of_loss, _pi_DK],
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
                _weight_x:weight_x,
                _weight_y:weight_y,
                _frac_progress:frac_progress
                })
        '''
        do_secondary = do_secondary_update(
            weight_x=weight_x,
            weight_y=weight_y)
        if do_secondary:
            assert cur_dataset['n_docs'] == pi_DK.shape[0]
            second_grad_vec = second_grad_func(
                param_vec,
                dataset=cur_dataset,
                step_id=None,
                cur_slice=None,
                pi_DK=pi_DK,
                weight_x=weight_x,
                weight_y=weight_y,
                **loss_kwargs)
            grad_vec += second_grad_vec
        '''
        if return_dict:
            return grad_vec, dict(pi_DK=pi_DK)
        else:
            return grad_vec
    return loss_func, grad_func

def calc_loss__slda(
        dataset=None,
        param_vec=None,
        topics_KV=None,
        w_CK=None,
        weight_x=1.0,
        weight_y=1.0,
        alpha=1.0,
        tau=1.0,
        lambda_w=0.000001,
        quality_of_init=1.0,
        frac_progress=1.0,
        pi_min_iters=DefaultDocTopicOptKwargs['min_pi_max_iters'],
        pi_max_iters=DefaultDocTopicOptKwargs['max_iters'],
        **kwargs):
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
    
    # Unpack params
    if param_vec is not None:
        param_dict = unflatten_to_param_dict__tf(param_vec, **kwargs)
        topics_KV = param_dict['topics_KV']
        w_CK = param_dict['w_CK']
    K, _ = topics_KV.get_shape().as_list()
    C, _ = w_CK.get_shape().as_list()
    # Establish pi_opt_kwargs
    pi_min_iters = int(pi_min_iters + np.ceil(
        quality_of_init * (pi_max_iters - pi_min_iters)))
    pi_opt_kwargs = dict(**DefaultDocTopicOptKwargs)
    half_frac_progress = tf.minimum(1.0, 2 * frac_progress)
    cur_pi_max_iters = tf.cast(
        pi_min_iters + tf.ceil(
            half_frac_progress * (pi_max_iters - pi_min_iters)),
        tf.int32)
    pi_opt_kwargs['max_iters'] = cur_pi_max_iters

    def has_docs_left(
            d, avg_log_proba_x, avg_log_proba_y,
            avg_log_proba_pi, pi_arr, y_arr):
        return d < n_docs
    def update_doc(
            d, avg_log_proba_x, avg_log_proba_y,
            avg_log_proba_pi, pi_arr, y_arr):
        start_d = doc_indptr_Dp1[d]
        stop_d = doc_indptr_Dp1[d+1]
        word_id_d_U = word_id_U[start_d:stop_d]
        word_ct_d_U = word_ct_U[start_d:stop_d]
        
        pi_d_K, topics_KUd, info_dict = \
            calc_topic_probas_for_single_doc(
                word_id_d_U,
                word_ct_d_U,
                topics_KV=topics_KV,
                alpha=alpha,
                **pi_opt_kwargs)
        pi_arr = pi_arr.write(d, pi_d_K)
        avg_log_proba_pi_d = tf.reduce_sum(
            (alpha - 1.0) * tf.log(1e-9 + pi_d_K))
        avg_log_proba_x_d = tf.reduce_sum(
            word_ct_d_U * 
            tf.log(tf.matmul(tf.reshape(pi_d_K, (1,K)), topics_KUd)))
        if include_norm_const:
            avg_log_proba_x_d += \
                tf.lgamma(1.0 + tf.reduce_sum(word_ct_d_U)) - \
                tf.reduce_sum(tf.lgamma(1.0 + word_ct_d_U))

        log_proba_y_d_C = tf.reduce_sum(
            w_CK * tf.reshape(pi_d_K, (1,K)), axis=1)

        avg_log_proba_y_d = tf.cond(
            y_rowmask[d] > 0,
            lambda: -1.0 * tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(log_proba_y_d_C, y_DC[d])),
            lambda: tf.constant(0.0, dtype=tf.float64))
        #avg_log_proba_y_d = \
        #    -1.0 * tf.cast(y_rowmask[d], tf.float64) * tf.reduce_sum(
        #        tf.nn.sigmoid_cross_entropy_with_logits(
        #            log_proba_y_d_C, y_DC[d]))
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
    _K = tf.cast(K, tf.float64)
    _alpha = tf.cast(alpha, tf.float64)
    _avg_log_proba_pi = tf.constant(0.0, dtype=tf.float64)
    #_avg_log_proba_pi = tf.cast(n_docs, tf.float64) * (
    #        tf.lgamma(_K * _alpha) - _K * tf.lgamma(_alpha))
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

    if rescale_divide_by_num_obs:
        scale = tf.reduce_sum(word_ct_U)
        n_tr_docs = 1e-10 + tf.cast(tf.reduce_sum(y_rowmask), tf.float64)
        _avg_log_proba_x /= scale
        _avg_log_proba_pi /= scale
        _avg_log_proba_y /= (C * n_tr_docs)
        _avg_log_proba_topics /= scale
        _avg_log_proba_w /= scale
        scale_y = (C * n_tr_docs) / scale
    else:
        scale_y = 1.0 / scale
    return (
        -1.0 * _avg_log_proba_x,
        -1.0 * _avg_log_proba_y,
        scale_y,
        -1.0 * _avg_log_proba_pi,
        -1.0 * _avg_log_proba_topics,
        -1.0 * _avg_log_proba_w,
        _pi_DK, _y_proba_DC)
