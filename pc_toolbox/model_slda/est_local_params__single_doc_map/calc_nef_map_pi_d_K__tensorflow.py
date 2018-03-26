import tensorflow as tf
import numpy as np
from calc_nef_map_pi_d_K__defaults import DefaultDocTopicOptKwargs

def calc_nef_map_pi_d_K__tensorflow(
        word_id_d_Ud=None,
        word_ct_d_Ud=None,
        topics_KV=None,
        topics_KUd=None,
        convex_alpha_minus_1=None,
        **pi_opt_kwargs):
    ''' Find MAP estimate of the K-dim. proba vector for specific document.

    Uses Natural-parameter Exponential Family (NEF) formulation,
    so the optimization problem is always convex.

    Finds solution via iterative exponentiated gradient steps.

    Requires tensorflow.

    Returns
    -------
    pi_d_K : 1D array, size K
        Contains non-negative entries that sum to one.
    info_dict : dict
    '''
    Ud = word_ct_d_Ud.size

    ## Form the graph
    if topics_KV is not None:
        K, V = topics_KV.shape
        _topics_KV = tf.placeholder(shape=[K, V], dtype=tf.float64)
        _word_id_d_Ud = tf.placeholder(shape=[Ud], dtype=tf.int32)
        _word_ct_d_Ud = tf.placeholder(shape=[Ud], dtype=tf.float64)
        _pi_d_K, _info_dict = _calc_nef_map_pi_d_K__tensorflow_graph(
            _word_id_d_Ud=_word_id_d_Ud,
            _word_ct_d_Ud=_word_ct_d_Ud,
            _topics_KV=_topics_KV,
            convex_alpha_minus_1=convex_alpha_minus_1,
            **pi_opt_kwargs) 
        feed_dict={
            _word_id_d_Ud:word_id_d_Ud,
            _word_ct_d_Ud:word_ct_d_Ud,
            _topics_KV:topics_KV,
            } 
    else:
        K, Ud = topics_KUd.shape
        _topics_KUd = tf.placeholder(shape=[K, Ud], dtype=tf.float64)
        _word_ct_d_Ud = tf.placeholder(shape=[Ud], dtype=tf.float64)
        _pi_d_K, _info_dict, const_dict = _calc_nef_map_pi_d_K__tensorflow_graph(
            _word_ct_d_Ud=_word_ct_d_Ud,
            _topics_KUd=_topics_KUd,
            convex_alpha_minus_1=convex_alpha_minus_1,
            )
        feed_dict={
            _word_ct_d_Ud:word_ct_d_Ud,
            _topics_KUd:topics_KUd,
            }
    sess = tf.Session()
    pi_d_K, info_dict = sess.run([_pi_d_K, _info_dict],
        feed_dict=feed_dict)
    info_dict.update(const_dict)
    return pi_d_K, info_dict 


def _calc_nef_map_pi_d_K__tensorflow_graph(
        _word_id_d_Ud=None,
        _word_ct_d_Ud=None,
        _topics_KUd=None,
        _topics_KV=None,
        convex_alpha_minus_1=None,
        pi_max_iters=DefaultDocTopicOptKwargs['pi_max_iters'],
        pi_converge_thr=DefaultDocTopicOptKwargs['pi_converge_thr'],
        pi_step_size=DefaultDocTopicOptKwargs['pi_step_size'],
        pi_min_step_size=DefaultDocTopicOptKwargs['pi_min_step_size'],
        pi_step_decay_rate=DefaultDocTopicOptKwargs['pi_step_decay_rate'],
        pi_min_mass_preserved_to_trust_step=(
            DefaultDocTopicOptKwargs['pi_min_mass_preserved_to_trust_step']),
        **kwargs):
    ''' Find MAP estimate of the K-dim. proba vector for specific document.

    Uses Natural-parameter Exponential Family (NEF) formulation,
    so the optimization problem is always convex.

    Finds solution via iterative exponentiated gradient steps.

    Requires tensorflow.

    Returns
    -------
    pi_d_K : 1D array, size K
        Contains non-negative entries that sum to one.
    info_dict : dict
    '''
    if _topics_KUd is None:
        # Fancy indexing
        # RHS equal to topics_KV[:, word_id_d_Ud]
        _topics_KUd = tf.transpose(
            tf.gather(tf.transpose(_topics_KV), _word_id_d_Ud))
    K, Ud = _topics_KUd.get_shape().as_list()
    _ct_topics_KUd = _topics_KUd * _word_ct_d_Ud

    ## Define part 1/2 of while loop: How to update
    def update_pi_d_K_func(
            iterid, pi_d_K, pi_step_size, cur_l1_diff, did_converge, n_restarts):
        _denom_Ud = 1.0 / tf.matmul(tf.reshape(pi_d_K, (1,K)), _topics_KUd)
        grad_K = pi_step_size * (
            tf.matmul(_ct_topics_KUd, tf.transpose(_denom_Ud))[:,0]
            + tf.cast(convex_alpha_minus_1, tf.float64) / (1e-9 + pi_d_K))
        grad_K = grad_K - tf.reduce_max(grad_K)
        new_pi_d_K = pi_d_K * tf.exp(grad_K)
        new_pi_d_K_sum = tf.reduce_sum(new_pi_d_K)
        new_pi_d_K = new_pi_d_K / new_pi_d_K_sum
        iterid += 1

        #did_converge = tf.logical_and(
        #    tf.equal(tf.mod(iterid, 5), 0),
        #    tf.less(tf.reduce_sum(tf.abs(new_pi_d_K - pi_d_K)), pi_converge_thr)
        #    )
        new_l1_diff = tf.cond(
            tf.equal(tf.mod(iterid, 5), 0),
            lambda: (tf.reduce_sum(tf.abs(new_pi_d_K - pi_d_K))),
            lambda: (cur_l1_diff))
        did_converge = tf.less(new_l1_diff, pi_converge_thr)

        def make_reset_state():
            return (
                iterid - 1,
                1.0 * pi_d_K,
                pi_step_size * pi_step_decay_rate,
                cur_l1_diff,
                tf.constant(False),
                n_restarts + 1,
                )
        return tf.cond(
            new_pi_d_K_sum <= pi_min_mass_preserved_to_trust_step,
            make_reset_state,
            lambda: (
                iterid,
                new_pi_d_K,
                pi_step_size,
                new_l1_diff,
                did_converge,
                n_restarts))
    ## Define PART 2/2 of while loop: func that returns True/False to continue
    def is_not_converged_func(
            iterid, pi_d_K, pi_step_size, cur_l1_diff, did_converge, n_restarts):
        return tf.cond(
            tf.logical_and(
                pi_step_size >= pi_min_step_size,
                tf.logical_and(iterid < pi_max_iters, tf.logical_not(did_converge))),
            lambda: tf.constant(True),
            lambda: tf.constant(False))

    # Initialize as uniform vector over K simplex
    _init_pi_d_K = 1.0 / K * tf.ones(K, dtype=tf.float64)
    _pi_d_K = 1.0 * _init_pi_d_K

    # Initialize alg state variables as proper tf constants
    _pi_step_size = tf.constant(pi_step_size, dtype=tf.float64)
    _iterid = tf.constant(0, dtype=tf.int32)
    _n_restarts = tf.constant(0, dtype=tf.int32)
    _cur_L1_diff = tf.constant(1.0, dtype=tf.float64)
    _did_converge = tf.constant(False, dtype=tf.bool)
    (_iterid, _pi_d_K, _pi_step_size,
            _cur_L1_diff, _did_converge, _n_restarts) = tf.while_loop(
        is_not_converged_func,
        update_pi_d_K_func,
        loop_vars=[
            _iterid, _pi_d_K, _pi_step_size,
            _cur_L1_diff, _did_converge, _n_restarts],
        )
    _info_dict = dict(
        n_iters=_iterid,
        n_restarts=_n_restarts,
        did_converge=_did_converge,
        cur_L1_diff=_cur_L1_diff,
        pi_step_size=_pi_step_size,
        )
    const_dict = dict(
        pi_max_iters=pi_max_iters,
        pi_converge_thr=pi_converge_thr,
        pi_min_step_size=pi_min_step_size,
        convex_alpha_minus_1=convex_alpha_minus_1)
    return _pi_d_K, _topics_KUd, _info_dict, const_dict
