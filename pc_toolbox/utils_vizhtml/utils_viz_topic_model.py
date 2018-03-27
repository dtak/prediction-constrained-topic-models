import numpy as np

from utils_top_words_html import make_top_words_html_from_topics

def show_topics_and_weights(
        param_dict=None,
        topics_KV=None,
        w_CK=None,
        uids_K=None,
        sort_topics_by=None,
        vocab_list=None,
        max_topics_to_display=200,
        n_words_per_topic=10,
        n_chars_per_word=30,
        rank_words_by='proba_word_given_topic',
        y_ind=0,
        vmax=0.05,
        vmin=0.00,
        add_bias_term_to_w_CK=0.0,
        **viz_kwargs):
    """ Show topics and weights for specific sLDA param dict

    Returns
    -------
    html_str : list of lines of html
    """
    if param_dict is not None:
        topics_KV = param_dict['topics_KV']
        w_CK = param_dict['w_CK']
    assert topics_KV is not None
    assert w_CK is not None
    # Make local temp copies
    # so we can re-sort at will
    topics_KV = topics_KV.copy()
    w_c_K = w_CK[y_ind].copy()
    assert w_c_K.ndim == 1
    K = w_c_K.size

    if uids_K is None:
        uids_K = np.arange(K)

    if rank_words_by == 'proba_word_given_topic':
        topics_KV /= topics_KV.sum(axis=1)[:,np.newaxis] 
    elif rank_words_by == 'proba_topic_given_word':
        topics_KV /= topics_KV.sum(axis=0)
    else:
        raise ValueError("Unrecognized rank_words_by: %s" % rank_words_by)

    ## Sort params if needed
    if sort_topics_by is not None:
        if sort_topics_by.count('w'):
            sort_ids = np.argsort(w_c_K)
            w_c_K = w_c_K[sort_ids]
            topics_KV = topics_KV[sort_ids]
            uids_K = uids_K[sort_ids]

    ## Prepare xlabels
    xlabels = ['% .1f' % a for a in w_c_K]

    ## Make plots
    if vocab_list is None:
        raise NotImplementedError("TODO make bars viz")
    else:
        return make_top_words_html_from_topics(
                topics_KV,
                vocab_list=vocab_list,
                xlabels=xlabels,
                uids_K=uids_K,
                n_words_per_topic=n_words_per_topic,
                n_chars_per_word=n_chars_per_word,
                max_topics_to_display=max_topics_to_display,
                **viz_kwargs)
