import numpy as np
import bnpy.viz

from sklearn.externals import joblib
from utils_snapshots import (
    load_param_dict_at_specific_snapshot,
    )

def show_topics_and_weights(
        param_dict=None,
        snapshot_path=None,
        task_path=None,
        lap=None,
        snapshot_filename=None,
        sort_by=None,
        vocab_list=None,
        return_param_dict=False,
        Kmax=200,
        n_top_words=10,
        show_enriched_words=False,
        y_ind=0,
        vmax=0.05,
        vmin=0.00,
        do_html=False,
        download_if_necessary=False,
        rsync_path=None,
        local_path=None,
        remote_path=None,
        add_bias_term_to_w_CK=0.0,
        **kwargs):
    """ Show topics and weights

    Returns
    -------
    html_str : list of lines of html
    """
    ## Load param dict
    load_param_kwargs = dict(
        add_bias_term_to_w_CK=add_bias_term_to_w_CK,
        download_if_necessary=download_if_necessary,
        rsync_path=rsync_path,
        local_path=local_path,
        remote_path=remote_path,
        )
    if param_dict is not None:
        P = param_dict
    elif snapshot_path is not None:
        P = load_param_dict_at_specific_snapshot(
            snapshot_path=snapshot_path,
            **load_param_kwargs)
    elif snapshot_filename is not None:
        if snapshot_filename.endswith('snapshot/'):
            snapshot_path = os.path.join(task_path, snapshot_filename)
        elif snapshot_filename.endswith('.dump'):
            snapshot_path = os.path.join(task_path, snapshot_filename)            
        else:
            snapshot_path = os.path.join(task_path, '%s_param_dict.dump' % (
                snapshot_filename))
        P = load_param_dict_at_specific_snapshot(
            snapshot_path=snapshot_path,
            **load_param_kwargs)
    elif lap is not None:
        P = load_param_dict_at_specific_snapshot(
            task_path=task_path,
            lap=lap,
            **load_param_kwargs)
    else:
        raise ValueError("No valid snapshot specified.")

    ## Sort param dict internally if needed
    # Here we have loaded parameter dict as P
    uids_K = None
    if sort_by is not None:
        if sort_by.count('w') and 'w_CK' in P and P['w_CK'] is not None:
            sort_ids = np.argsort(P['w_CK'][y_ind])
            P['w_CK'][y_ind, :] = P['w_CK'][y_ind, sort_ids]
            P['topics_KV'] = P['topics_KV'][sort_ids]
            uids_K = sort_ids

    ## Prepare xlabels
    if 'w_CK' in P and P['w_CK'] is not None:
        xlabels = ['% .1f' % a for a in P['w_CK'][y_ind].flatten()]
    else:
        xlabels = None

    ## Make plots
    if vocab_list is None:
        return bnpy.viz.BarsViz.show_square_images(
            P['topics_KV'],
            vmin=vmin,
            vmax=vmax,
            xlabels=xlabels,
            max_n_images=Kmax,
            **kwargs);
    else:
        if show_enriched_words:
            topics_KV = P['topics_KV'].copy() / P['topics_KV'].sum(axis=0) 
        else:
            topics_KV = P['topics_KV']
        if do_html:
            return bnpy.viz.PrintTopics.htmlTopWordsFromTopics(
                topics_KV,
                vocabList=vocab_list,
                label_per_topic=xlabels,
                Ktop=n_top_words,
                maxKToDisplay=Kmax,
                uids_K=uids_K,
                **kwargs)
        else:
            bnpy.viz.PrintTopics.plotCompsFromWordCounts(
                topics_KV=topics_KV,
                vocabList=vocab_list,
                xlabels=xlabels,
                Ktop=n_top_words,
                Kmax=Kmax,
                **kwargs)
    if return_param_dict:
        return P