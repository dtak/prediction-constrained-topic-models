import tensorflow as tf

from pc_toolbox.utils_diffable_transforms import tfm__2D_rows_sum_to_one

def _unflatten_to_common_param_dict__tf_graph(
        param_vec,
        n_states=0,
        n_vocabs=0,
        n_labels=0,
        min_eps=tfm__2D_rows_sum_to_one.MIN_EPS,
        **unused_kwargs):
    K = int(n_states)
    V = int(n_vocabs)
    C = int(n_labels)
    F_topics = K * (V-1)
    log_topics_KVm1 = tf.reshape(param_vec[:F_topics], (K, V-1))
    log_topics_KV = tf.concat([
        log_topics_KVm1,
        tf.zeros([K, 1], dtype=tf.float64)],
        axis=1)
    topics_KV = min_eps + tf.exp(
        log_topics_KV
        - tf.reduce_logsumexp(
            log_topics_KV,
            reduction_indices=[1],
            keepdims=True)
        + tf.log1p(tf.cast(-V * min_eps, dtype=tf.float64)))
    w_CK = tf.reshape(param_vec[F_topics:], (C, K))
    return dict(topics_KV=topics_KV, w_CK=w_CK)
    
    
def unflatten_to_common_param_dict__tf(
        param_vec=None,
        n_states=1,
        n_labels=1,
        n_vocabs=1,
        **dim_kwargs):
    K = int(n_states)
    V = int(n_vocabs)
    C = int(n_labels) 
    S = K * (V-1) + K * C
    _param_vec = tf.placeholder(shape=[S], dtype=tf.float64)
    _param_dict = _unflatten_to_common_param_dict__tf_graph(
        _param_vec,
        n_states=n_states,
        n_labels=n_labels,
        n_vocabs=n_vocabs,
        )
    sess = tf.Session()
    param_dict = sess.run([_param_dict], feed_dict={_param_vec:param_vec})[0]
    return param_dict
