"""
Compute evaluation metrics for model snapshot


"""

import numpy as np
import os
import pandas as pd
import psutil
import time

from sklearn.metrics import roc_auc_score
from distutils.dir_util import mkpath
from collections import OrderedDict

import slda_loss__cython

from pc_toolbox.utils_io import (
    pprint,
    append_to_txtfile,
    update_symbolic_link,
    start_timer_segment,
    stop_timer_segment,
    pprint_timer_segments,
    )
from pc_toolbox.topic_quality_metrics import (
    calc_coherence_metrics as coh
    )
def calc_perf_metrics_for_snapshot_param_dict(
        param_dict=None,
        topics_KV=None,
        w_CK=None,
        datasets_by_split=None,
        model_hyper_P=None,
        dim_P=None,
        alg_state_kwargs=None,
        output_path=None,
        cur_lap=0.0,
        cur_step=None,
        elapsed_time_sec=0.0,
        losstrain_ttl=None,
        verbose_timings=False,
        disable_output=False,
        do_force_update_w_CK=0,
        perf_metrics_pi_optim_kwargs=None,
        **unused_kwargs):
    ''' Compute performance metrics at provided topic model param dict.

    Returns
    -------
    info_dict : dict
        Contains all perf. metric information.

    Post Condition
    --------------
    Row appended to CSV files in output_path/
        * snapshot_perf_metrics_train.csv
        * snapshot_perf_metrics_valid.csv
        * snapshot_perf_metrics_test.csv
    '''
    if perf_metrics_pi_optim_kwargs is None:
        perf_metrics_pi_optim_kwargs = dict()

    etimes = OrderedDict()
    etimes = start_timer_segment(etimes, 'total')

    # Unpack parameters
    if param_dict is not None:
        topics_KV = param_dict['topics_KV']
        w_CK = param_dict['w_CK']
    if topics_KV is None:
        raise ValueError("topics_KV should not None")
    if not np.all(np.isfinite(topics_KV)):
        raise ValueError("topics_KV should not be NaN or Inf")
    if w_CK is None:
        raise ValueError("w_CK should not None")
    if not np.all(np.isfinite(w_CK)):
        raise ValueError("w_CK should not be NaN or Inf")
    # Track norms of params (crude debugging tool)
    l1_norm_logtopics = np.mean(np.abs(np.log(topics_KV.flatten())))
    l1_norm_w = np.mean(np.abs(w_CK.flatten()))

    # Unpack hyperparams
    alpha = model_hyper_P['alpha']
    tau = model_hyper_P['tau']
    lambda_w = model_hyper_P['lambda_w']
    weight_y = model_hyper_P['weight_y']

    # Unpack state kwargs
    if alg_state_kwargs is not None:
        output_path = alg_state_kwargs['output_path']
        cur_lap = alg_state_kwargs['cur_lap']
        cur_step = alg_state_kwargs['cur_step']
        elapsed_time_sec = alg_state_kwargs['elapsed_time_sec']

    # TODO check if dataset is semisupervised
    y_DC = datasets_by_split['train']['y_DC']
    n_labels = y_DC.shape[1]
    u_y_vals = np.unique(y_DC.flatten())
    if u_y_vals.size <= 2 and np.union1d([0.0, 1.0], u_y_vals).size == 2:
        output_data_type = 'binary'
    else:
        output_data_type = 'real'

    # Count number of docs for which at least one pair of each vocab word occurs
    _, ndocs_csc_VV = coh.calc_pairwise_cooccurance_counts(
        dataset=datasets_by_split['train'])

    split_names = ['train', 'valid', 'test']
    for split_name in split_names:
        etimes = start_timer_segment(etimes, '%s_calc_lossmap' % split_name)
        ans_dict = slda_loss__cython.calc_loss__slda(
            dataset=datasets_by_split[split_name],
            topics_KV=topics_KV,
            w_CK=w_CK,
            LP=None,
            weight_x=1.0,
            weight_y=1.0,
            alpha=alpha,
            tau=tau,
            lambda_w=lambda_w,
            pi_estimation_mode='missing_y',
            pi_estimation_weight_y=0.0,
            return_dict=True,
            **perf_metrics_pi_optim_kwargs)
        etimes = stop_timer_segment(etimes, '%s_calc_lossmap' % split_name)
        assert 'summary_msg' in ans_dict

        # Extract doc-topic features
        assert 'pi_DK' in ans_dict
        pi_DK = ans_dict.pop('pi_DK')

        info_dict = OrderedDict([
            ('step', float(cur_step)),
            ('lap', float(cur_lap)),
            ('elapsed_time_sec', float(elapsed_time_sec)),
            ('logpdf_x_pertok', -1*ans_dict['uloss_x__pertok']),
            ('logpdf_y_perdoc', -1*ans_dict['uloss_y__perdoc']),
            ('lossmap_ttl_pertok', ans_dict['loss_ttl']),
            ('lossmap_x_pertok', ans_dict['loss_x']),
            ('lossmap_y_pertok', ans_dict['loss_y']),
            ('lossmap_pi_pertok', ans_dict['loss_pi']),
            ('lossmap_topic_pertok', ans_dict['loss_topics']),
            ('lossmap_w_pertok', ans_dict['loss_w']),
            ])
        if losstrain_ttl is not None:
            info_dict['losstrain_ttl'] = float(losstrain_ttl)

        ## Compute y metrics
        # Case 1/2: binary
        etimes = start_timer_segment(etimes, '%s_calc_y_metrics' % split_name)
        assert 'y_proba_DC' in ans_dict
        if output_data_type.count('binary'):
            y_proba_DC = ans_dict.pop('y_proba_DC')
            C = y_proba_DC.shape[1]
            assert np.nanmin(y_proba_DC) >= 0.0
            assert np.nanmax(y_proba_DC) <= 1.0
            for c in xrange(n_labels):
                ytrue_c_D = datasets_by_split[split_name]['y_DC'][:,c]
                yproba_c_D = y_proba_DC[:, c]
                # Keep only finite values
                rowmask = np.logical_and(
                    np.isfinite(yproba_c_D),
                    np.isfinite(ytrue_c_D))
                ytrue_c_D = ytrue_c_D[rowmask]
                yproba_c_D = yproba_c_D[rowmask]
                if ytrue_c_D.size == 0:
                    raise ValueError("Label id c=%d has no observed y values" % c)

                yhat_c_D = np.asarray(
                    yproba_c_D > 0.5,
                    dtype=ytrue_c_D.dtype)

                # Error rate
                error_rate_y__c = np.sum(np.logical_xor(ytrue_c_D, yhat_c_D))
                error_rate_y__c /= float(ytrue_c_D.size)
                info_dict['y_%d_error_rate' % c] = error_rate_y__c

                # Area under ROC curve
                try:
                    roc_auc_y__c = roc_auc_score(ytrue_c_D, yproba_c_D)
                except ValueError as e:
                    # Error occurs when not enough examples of each label
                    roc_auc_y__c = 0.0
                info_dict['y_%d_roc_auc' % c] = roc_auc_y__c

        # Case 2/2: real values
        elif output_data_type.count('real'):
            # Remember, y_proba_DC is really estimated mean of y_DC
            y_est_DC = ans_dict.pop('y_proba_DC')
            for c in xrange(n_labels):
                y_true_c_D = datasets_by_split['split']['y_DC'][:, c]
                y_est_c_D = y_est_DC[:, c]
                # Keep only finite values
                rowmask = np.logical_and(
                    np.isfinite(y_true_c_D),
                    np.isfinite(y_est_c_D))
                y_true_c_D = y_true_c_D[rowmask]
                y_est_c_D = y_est_c_D[rowmask]
                if y_true_c_D.size == 0:
                    raise ValueError("Label id c=%d has no observed y values" % c)
                # Compute RMSE
                rmse = np.sqrt(np.mean(np.square(y_true_c_D - y_est_c_D)))
                info_dict['y_%d_rmse' % c] = rmse
        etimes = stop_timer_segment(etimes, '%s_calc_y_metrics' % split_name)

        ## COHERENCE
        etimes = start_timer_segment(etimes, '%s_calc_coher_metrics' % split_name)
        K = topics_KV.shape[0]
        npmi_K = np.zeros(K)
        for k in range(K):
            # Select at most 20 vocab words per topic
            # But if fewer than that take up 90% of the mass, take only those
            top_vocab_ids = np.argsort(-1*topics_KV[k])[:20]
            cumsum_mass = np.cumsum(topics_KV[k, top_vocab_ids])
            m = np.searchsorted(cumsum_mass, 0.9)
            top_vocab_ids = top_vocab_ids[:(m+1)]
            npmi_K[k], _ = coh.calc_npmi_and_pmi_coherence_for_top_ranked_terms_in_topic(
                ndocs_csc_VV=ndocs_csc_VV,
                top_vocab_ids=top_vocab_ids,
                pair_smooth_eps=0.1)
        if K < 10:
            perc_list = [0, 50, 100]
        else:
            perc_list = [0, 10, 50, 90, 100]
        for perc in perc_list:
            pstr = '%06.2f' % perc
            info_dict['topic_npmi_p' + pstr] = np.percentile(npmi_K, perc)

        etimes = stop_timer_segment(etimes, '%s_calc_coher_metrics' % split_name)


        info_dict['losstrain_weight_y'] = weight_y
        info_dict['alpha'] = alpha
        info_dict['tau'] = tau
        info_dict['lambda_w'] = lambda_w

        info_dict['n_states'] = float(topics_KV.shape[0])
        info_dict['l1norm_w'] = float(l1_norm_w)
        info_dict['l1norm_logtopics'] = float(l1_norm_logtopics)

        info_df = pd.DataFrame([info_dict])
        col_order = info_dict.keys()
        ppinfo_str = info_df.to_csv(
            None,
            float_format='% 20.12g',
            na_rep='%20s' % 'nan',
            index=False,
            header=False,
            columns=col_order) # relying on an ordered dict here
        info_str = info_df.to_csv(
            None,
            float_format='% .12g',
            na_rep='nan',
            index=False,
            header=False,
            columns=col_order) # relying on an ordered dict here
        assert np.max(map(len, col_order)) <= 20
        if not disable_output:
            csv_fpath = os.path.join(output_path, 'snapshot_perf_metrics_%s.csv' % split_name)
            ppcsv_fpath = os.path.join(output_path, 'pretty_snapshot_perf_metrics_%s.csv' % split_name)

            if int(cur_step) == 0:
                with open(csv_fpath, 'w') as f:
                    header_str = ','.join(['%s' % s for s in col_order])
                    f.write(header_str + "\n")
                with open(ppcsv_fpath, 'w') as f:
                    header_str = ','.join(['%20s' % s for s in col_order])
                    f.write(header_str + "\n")
            with open(csv_fpath, 'a') as f:
                f.write(info_str)
            with open(ppcsv_fpath, 'a') as f:
                f.write(ppinfo_str)

            pi_summary_txt_fpath = os.path.join(
                output_path, 'perf_metrics_pi_optim_summaries_%s.txt' % split_name)
            lap_prefix = 'lap %011.3f  ' % cur_lap
            with open(pi_summary_txt_fpath, 'a') as f:
                f.write(
                    lap_prefix + ans_dict['summary_msg'] + "\n")


    # Write timings to txt file for comparison
    msg = pprint_timer_segments(etimes, prefix='lap%011.3f' % (cur_lap))
    if verbose_timings:
        pprint(msg)
    if not disable_output:
        timings_txt = os.path.join(output_path, 'timings_for_perf_metrics.txt')
        with open(timings_txt, 'a') as f:
            f.write(msg)
    return info_dict
