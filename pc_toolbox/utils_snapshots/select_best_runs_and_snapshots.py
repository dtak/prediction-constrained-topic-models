# -*- coding: utf-8 -*-
"""

Usage
-----
python select_best_runs_and_snapshots.py \
    --output_path [path] \
    --results_path_patterns [patternA,patternB,patternC] \
    [other args]

"""

from __future__ import unicode_literals

import argparse
import os
import sys
import glob
import numpy as np
import pandas as pd
import json

from collections import OrderedDict
from distutils.dir_util import mkpath

from pc_toolbox.utils_io import (
    pprint,
    config_pprint_logging,
    load_list_of_unicode_from_txt)
from utils_snapshots import (
    make_snapshot_path_for_lap,
    download_snapshot,
    load_param_dict_at_specific_snapshot)

SPLIT_NAMES = ["TRAIN", "VALID", "TEST"]
COLUMN_NAMES = [
    "JOB_PATH",
    "TASKID",
    ]

def fetch_default_score(ranking_func_name):
    if ranking_func_name.count('argmax'):
        return -1.0 * np.inf
    elif ranking_func_name.count('argmin'):
        return +1.0 * np.inf
    else:
        raise ValueError("Unrecognized ranking_func_name %s" % ranking_func_name)

def load_default_column_name_dict(output_data_type='binary'):
    if output_data_type == 'binary':
        json_fpath = os.path.expandvars(
            "$PC_REPO_DIR/pc_toolbox/utils_snapshots/snapshot_perf_metrics__binary_outcomes.json")
        with open(json_fpath, 'r') as fd:
            j = json.load(fd)
            kv_pairs = [
                (a['name'], a['type'])
                    for a in j['resources'][0]['schema']['fields']]
            return OrderedDict(kv_pairs)

def get_score_ranking_function_for_colname(score_name):
    if score_name.count("AUC"):
        return np.argmax
    elif score_name.count("LOSS"):
        return np.argmin
    else:
        return np.argmin

def select_best_from_many_runs(
        legend_name=None,
        results_path_patterns=None,
        output_path=None,
        txt_src_path=None,
        target_y_name=None,
        all_y_names=None,
        col_names_to_use_at_selection=['N_STATES'],
        col_names_to_keep="",
        col_names_to_keep_per_split="",
        min_lap_to_use_at_selection=10,
        split_name_to_use_at_selection='VALID',
        selection_score_colname='LOSS_X',
        selection_score_ranking_func='argmin',
        unk_list=None,
        **kwargs):
    """
    """
    provided_arg_dict = dict(**locals())

    # Create output_path on disk
    if output_path.count("$"):
        for key, val in locals().items():
            if output_path.count('$' + key):
                output_path = output_path.replace("$" + key, str(val))
    if not os.path.exists(output_path):
        mkpath(output_path)

    # Setup logging
    suffix = "__select_best__y_target=%s_score=%s_legend=%s" % (
            target_y_name,
            selection_score_colname,
            legend_name)
    config_pprint_logging(
        output_path,
        txtfile='stdout%s.txt' % suffix)

    # Write parsed args to plain-text file
    # so we can exactly reproduce later
    this_script_prefix = '[select_best.py says:]'
    pprint("%s Parsing args ..." % this_script_prefix)
    with open(os.path.join(output_path, 'settings%s.txt' % suffix), 'w') as f:
        for key, val in provided_arg_dict.items():
            f.write(key + ' = ' + str(val) + '\n')
            pprint(key + ' = ' + str(val))
    with open(os.path.join(output_path, 'args%s.txt' % suffix), 'w') as f:
        for key, val in provided_arg_dict.items():
            f.write('--' + key + ' ' + str(val) + '\n')
    pprint('')

    # Parse unknown args
    if unk_list is not None and len(unk_list) > 0:
        pprint("UNKNOWN ARGS (ignored)")
        for key in unk_list:
            pprint(key)
        del unk_list


    # Parse target y names
    target_y_name = unicode(target_y_name)
    if not isinstance(all_y_names, list):
        if os.path.exists(all_y_names):
            all_y_names = load_list_of_unicode_from_txt(all_y_names)
        else:
            all_y_names = map(unicode, all_y_names.split(","))
    def force_list_of_strings(val):
        if not isinstance(val, list):
            val = map(str, val.split(","))
        return val
    results_path_patterns = force_list_of_strings(results_path_patterns)
    col_names_to_use_at_selection = force_list_of_strings(
        col_names_to_use_at_selection)
    col_names_to_keep = force_list_of_strings(
        col_names_to_keep)
    col_names_to_keep_per_split = force_list_of_strings(
        col_names_to_keep_per_split)

    # Load df for all runs that match the query
    all_matching_runs_df = load_df_from_all_folders_matching_list_of_patterns(
        list_of_path_patterns=results_path_patterns,
        legend_name=legend_name,
        y_ind=all_y_names.index(target_y_name),
        column_names=COLUMN_NAMES,
        task_ids=range(1, 10),
        )
    all_matching_runs_df['TARGET_LABEL_NAME'] = target_y_name

    if selection_score_colname.startswith("="):
        formula = selection_score_colname.lstrip("=")
        all_matching_runs_df[selection_score_colname] = 0.0
        add_ops = formula.split("+")
        for op in add_ops:
            coef, colname = op.lstrip('(').rstrip(')').split("*")
            coef = float(coef)
            all_matching_runs_df[selection_score_colname] += coef * all_matching_runs_df[colname].values

    if selection_score_ranking_func is None:
        selection_score_ranking_func = get_score_ranking_function_for_colname(
            selection_score_colname)
    elif selection_score_ranking_func == 'argmax':
        selection_score_ranking_func = np.argmax
    else:
        selection_score_ranking_func = np.argmin

    ## Create dataframe with only the best task at each legend name
    best_df = select_best_df_at_each_value_of_specific_vars(
        all_matching_runs_df,
        legend_name=legend_name,
        keys=col_names_to_use_at_selection,
        query_min_lap=min_lap_to_use_at_selection,
        score_colname=selection_score_colname,
        score_ranking_func=selection_score_ranking_func,
        target_splitname=split_name_to_use_at_selection,
        )
    row_dict_list = list()
    # Write the legend names to output path
    for cur_legend_name in np.unique(best_df['LEGEND_NAME_ASCII'].values):

        ## Make symlink to best run's task_path directory
        cur_query_str = (
            "LEGEND_NAME_ASCII == '%s' and IS_BEST_SNAPSHOT > 0"
            % (cur_legend_name)
            )
        # Prepare existing path
        best_snapshot_df = best_df.query(cur_query_str)
        assert best_snapshot_df.shape[0] == len(SPLIT_NAMES)
        best_task_path = best_snapshot_df['TASK_PATH_AT_BEST_SNAPSHOT'].values[0]
        best_task_path = best_task_path.rstrip(os.path.sep)
        assert os.path.exists(best_task_path)
        # Prepare symlink path
        job_path = "best_snapshot_run-legend_name=%s" % (
            cur_legend_name.replace(" ", "_"))
        cur_symlink_output_job_path = os.path.join(output_path, job_path)
        mkpath(cur_symlink_output_job_path)
        cur_symlink_output_task_path = os.path.join(output_path, job_path, 'best_task')
        # Remove any old version
        if os.path.islink(cur_symlink_output_task_path):
            os.unlink(cur_symlink_output_task_path)
        # Finally, make the symlink happen
        os.symlink(best_task_path, cur_symlink_output_task_path)
        pprint("\nLEGEND_NAME %s" % cur_legend_name)
        pprint("NEW BEST TASK PATH:\n%s" % cur_symlink_output_task_path)

        ## Make symlink to best snapshot directory

        # Prepare existing snapshot path (download content if necessary)
        snapshot_path = make_snapshot_path_for_lap(
            task_path=best_snapshot_df['TASK_PATH_AT_BEST_SNAPSHOT'].values[0],
            lap=best_snapshot_df['LAP_AT_BEST_SNAPSHOT'].values[0],
            )
        if not os.path.exists(snapshot_path):
            download_snapshot(snapshot_path)
        # Prepare new symlink path
        cur_symlink_snapshot_path = os.path.join(
            cur_symlink_output_job_path, 'best_snapshot')
        # Remove any old version
        if os.path.islink(cur_symlink_snapshot_path):
            os.unlink(cur_symlink_snapshot_path)
        # Finally, make the symlink happen
        os.symlink(snapshot_path, cur_symlink_snapshot_path)
        pprint("NEW BEST SNAPSHOT PATH:\n%s" % cur_symlink_snapshot_path)

        ## If needed, make brand new snapshot with only target y column
        if len(all_y_names) > 1 and target_y_name != 'avg':
           GP = load_param_dict_at_specific_snapshot(
               snapshot_path=snapshot_path)
           new_GP = dict(**GP)
           new_GP['w_CK'] = GP['w_CK'][all_y_names.index(target_y_name),:][np.newaxis,:]
           save_topic_model_snapshot(
               output_path=cur_symlink_output_job_path,
               prefix='targety=%s' % (target_y_name),
               **new_GP)

        ## Append to .csv file
        row_dict = OrderedDict()
        row_dict['LEGEND_NAME'] = legend_name
        for key in col_names_to_use_at_selection:
            row_dict[key] = best_snapshot_df[key].values[0]
        for key in col_names_to_keep:
            row_dict[key] = best_snapshot_df[key].values[0]            

        for split_name in SPLIT_NAMES:
            best_split_df = best_snapshot_df.query("SPLIT_NAME == '%s'" % split_name)
            assert best_split_df.shape[0] == 1
            assert isinstance(col_names_to_keep_per_split, list)
            for key in col_names_to_keep_per_split:
                split_key = "%s_%s" % (split_name.upper(), key)
                row_dict[split_key] = best_split_df[key].values[0]
        row_dict['LAP'] = best_snapshot_df['LAP'].values[0]
        row_dict['LABEL_NAME'] = best_snapshot_df['TARGET_LABEL_NAME'].values[0]
        row_dict['SNAPSHOT_SRCFILE'] = cur_symlink_snapshot_path
        row_dict['TXTSRCFILES_PATH'] = txt_src_path
        row_dict_list.append(row_dict)


    pprint("\nWriting csv file documenting all best snapshots for legend %s" % (
        legend_name))
    my_df = pd.DataFrame(row_dict_list)
    basename = "best_snapshots_%s.csv" % legend_name
    csv_fpath = os.path.join(output_path, basename)
    my_df.to_csv(
        csv_fpath,
        columns=row_dict_list[0].keys(),
        index=False)
    pprint("WROTE CSV FILE:\n%s" % csv_fpath)


def load_df_from_all_folders_matching_list_of_patterns(
        list_of_path_patterns=None,
        legend_name=None,
        y_ind=0,
        column_names=None,
        query_str=None,
        task_ids=None,
        **kwargs):
    pprint(">>> BEGIN load_df_from_all_folders_that_match_pattern")
    list_of_match_df = list()
    for path_pattern in list_of_path_patterns:
        cur_alg_df = load_df_from_all_folders_that_match_pattern(
            path_pattern,
            y_ind=y_ind,
            task_ids=task_ids,
            column_names=column_names)
        if query_str is not None:
            cur_alg_df = cur_alg_df.query(query_str).copy()

        # Append to list of all matching dataframes
        list_of_match_df.append(cur_alg_df)
    # Create all matching DataFrame
    all_matching_runs_df = pd.concat(list_of_match_df)
    pprint("<<< END   load_df_from_all_folders_that_match_pattern")
    return all_matching_runs_df


######################
## Funcs that select best df

def select_best_df_at_each_value_of_specific_vars(
        df,
        legend_name='Gibbs_LDA',
        keys=['N_STATES'],
        disp_keys=None,
        no_legend_keys=[],
        query="SPLIT_NAME == '$target_splitname' and LAP >= $query_min_lap",
        query_min_lap=5,
        target_splitname='VALID',
        score_colname='LOSS_X',
        score_ranking_func=np.argmin,
        **kwargs):
    ''' Produce dataframe of best runs at each value of specific variables.

    Args
    ----
    df : pandas DataFrame
        Each row represents a snapshot during training.
    legend_name : string
        Nickname of all runs provided.
    keys : list of strings
        Column names of specified variables used for best run selection.

    Returns
    -------
    best_df : pandas DataFrame
    '''
    if disp_keys is None:
        disp_keys = ['LAP_AT_BEST_SNAPSHOT', 'TASKID'] + keys
    query = query.replace("$query_min_lap", str(query_min_lap))
    query = query.replace("$target_splitname", str(target_splitname))
    pprint("Finding snapshots with %s of %s" % (
        score_ranking_func.__name__, score_colname))
    pprint("Among snapshots satisfying query: %s" % query)
    def expand_query_str_list(cur_list, new_vals):
        new_list = list()
        if len(cur_list) == 0:
            for new_q_str in new_vals:
                new_list.append(new_q_str)
        else:
            for q_str in cur_list:
                for new_q_str in new_vals:
                    new_list.append(q_str + " and " + new_q_str)
        return new_list
    
    query_str_list = list()
    pprint("Finding best task for each possible combo of these legend keys:")
    for key in keys:
        is_finite_mask = np.logical_not(pd.isnull(df[key].values))
        if np.sum(is_finite_mask) > 0:
            u_vals = np.unique(df[key].values[is_finite_mask]).tolist()
        else:
            u_vals = []
        if not np.all(is_finite_mask):
            u_vals += [np.nan]
        new_queries = list()
        for u_val in u_vals:
            if isinstance(u_val, str):
                new_query_str = "%s == '%s'" % (key, u_val)
            elif np.isfinite(u_val):
                new_query_str = "%s == %s" % (key, u_val)
            else:
                new_query_str = "%s != %s" % (key, key)
            new_queries.append(new_query_str)
        if len(new_queries) == 1:
            if len(query_str_list) < 1:
                query_str_list.extend(new_queries)
            continue
        pprint("    %s: %s" % (key, ','.join(map(str,u_vals))))
        query_str_list = expand_query_str_list(query_str_list, new_queries)
    
    best_df_list = list()
    for query_str in query_str_list:
        best_job_df = make_best_job_df(
            df.query(query_str),
            target_query=query,
            score_colname=score_colname,
            score_ranking_func=score_ranking_func,
            target_splitname=target_splitname,
            **kwargs)

        if best_job_df is None:
            pprint("NO BEST TASK AVAILABLE FOR %s + %s" % (legend_name, query_str))
            continue

        # _UNIQUE_LEGEND_NAME distinctly identifies each "best job"
        # like 'Gibbs_LDA K == 5'
        # LEGEND_NAME may be simpler with duplicates
        # like 'Gibbs_LDA', for each of K in [5,10, 20]
            
        cur_queries = [s for s in query_str.split('and')]
        cur_legend_name = legend_name
        cur_ulegend_name = legend_name
        for cur_query_str in cur_queries:
            is_bad = False
            for no_leg_key in no_legend_keys:
                if cur_query_str.count(no_leg_key) > 0:
                    is_bad = True
            cur_ulegend_name += " " + cur_query_str.strip()
            if not is_bad:
                cur_legend_name += " " + cur_query_str.strip()
        best_job_df['_UNIQUE_LEGEND_NAME'] = cur_ulegend_name
        best_job_df['LEGEND_NAME'] = cur_legend_name
        best_df_list.append(best_job_df)
    best_df = pd.concat(best_df_list)

    pprint("ON SPLIT %s:" % (target_splitname))
    q_df = best_df.query(
        "IS_BEST_SNAPSHOT > 0 and SPLIT_NAME == '%s'"
            % target_splitname)
    disp_df = q_df[[score_colname] + disp_keys]
    disp_df = disp_df.apply(pd.to_numeric, errors='ignore')
    pprint(disp_df.to_string(
        index=False,
        header=True,
        float_format=lambda x: ' %.3f' % float(x)))

    best_df.reset_index(inplace=True)
    best_df = simplify_best_df_and_make_unicode_friendly(best_df)
    best_df.reset_index(inplace=True)
    return best_df


def make_best_job_df(
        df,
        target_query="SPLIT_NAME == 'VALID' and LAP > 50",
        target_splitname='VALID',
        score_colname='Y_ERROR_RATE',
        score_ranking_func=np.argmin,
        verbose=False):
    ''' Find single best task among all jobs in provided df.
    
    Returns
    -------
    best_job_df : dataframe of best single task
    '''
    default_score = fetch_default_score(score_ranking_func.__name__)
    job_paths = np.unique(df['JOB_PATH'].values)

    best_task_idstr_list = ['' for a in range(len(job_paths))]
    best_score_idx = np.zeros_like(job_paths, dtype=np.int32)
    best_score = default_score * np.ones_like(job_paths, dtype=np.float64)
    best_lap_idx = np.zeros_like(job_paths, dtype=np.float64)
    for jj, job_path in enumerate(job_paths):
        if job_path is None:
            continue

        cur_job_best_df = make_best_task_df(
            df.query("JOB_PATH == '%s'" % job_path),
            target_query=target_query,
            score_colname=score_colname,
            score_ranking_func=score_ranking_func,
            default_score=default_score,
            verbose=verbose)

        # Narrow down to ___ split, after __ laps
        cur_job_best_df = cur_job_best_df.query(target_query)
        if verbose:
            pprint(job_path.split(os.path.sep)[-1])

        if cur_job_best_df.shape[0] < 1:
            if verbose:
                pprint('    skipped. Too small to satisfy query.')
            continue

        split_name_chk = np.unique(cur_job_best_df['SPLIT_NAME'].values)
        assert len(split_name_chk) == 1
        assert split_name_chk[0].lower() == target_splitname.lower()
    
        best_task_idstr_list[jj] = str(cur_job_best_df['TASKID'].values[0])
        best_score_idx[jj] = score_ranking_func(
            cur_job_best_df[score_colname].values)
        best_score[jj] = cur_job_best_df[score_colname].values[
            best_score_idx[jj]]
        best_lap_idx[jj] = cur_job_best_df['LAP'].values[
            best_score_idx[jj]]
        if verbose:
            print("    best %s = %.4f at lap %9.3f of task %s" % (
                score_colname, best_score[jj],
                best_lap_idx[jj], best_task_idstr_list[jj]))

    # No tasks/jobs exist that satisfy target_query
    # This can happen when runs havent gone long enough yet
    if np.allclose(best_score, default_score):
        return None

    best_job_idx = score_ranking_func(best_score)
    best_job_df = df.query("JOB_PATH == '%s' and TASKID == '%s'" % (
            job_paths[best_job_idx],
            best_task_idstr_list[best_job_idx])).copy()
    best_job_df['SCORE_AT_BEST_SNAPSHOT'] = best_score[best_job_idx]
    best_job_df['LAP_AT_BEST_SNAPSHOT'] = best_lap_idx[best_job_idx]
    best_job_df['IS_BEFORE_BEST_SNAPSHOT'] = np.asarray(
        best_job_df['LAP'].values.copy() <= best_lap_idx[best_job_idx],
        dtype=np.int32)
    best_job_df['TASK_PATH_AT_BEST_SNAPSHOT'] = os.path.join(
        job_paths[best_job_idx],
        best_task_idstr_list[best_job_idx])
    best_job_df['IS_BEST_SNAPSHOT'] = np.asarray(
        best_job_df['LAP'].values.copy() == best_lap_idx[best_job_idx],
        dtype=np.int32)
    best_job_df['FRAC_PROGRESS'] = \
        1.0 * best_job_df['LAP'].values.copy() \
        / np.max(best_job_df['LAP'].values)
    return best_job_df

def make_best_task_df(
        df,
        target_query="SPLIT_NAME == 'VALID' and LAP > 50",
        score_colname='Y_ERROR_RATE',
        score_ranking_func=np.argmin,
        default_score=None,
        verbose=False):
    ''' Find best task for each unique job in provided df.

    Returns
    -------
    best_df : dataframe of best tasks for each unique job
    '''
    if default_score is None:
        default_score = fetch_default_score(score_ranking_func.__name__)
    best_task_df_list = list()
    job_paths = np.unique(df['JOB_PATH'].values)
    for job_path in job_paths:
        if job_path is None:
            continue
        job_df = df.query("JOB_PATH == '%s'" % job_path)
        taskids = np.unique(job_df['TASKID'].values)
        best_score_idx = np.zeros_like(taskids, dtype=np.int32)
        best_score = default_score * np.ones_like(taskids, dtype=np.float64)
        for tt, taskidstr in enumerate(taskids):
            task_df = job_df.query(target_query + " and TASKID == '%s'" % taskidstr)
            if task_df.shape[0] < 1:
                continue
            if not np.all(np.isfinite(task_df[score_colname].values)):
                pprint(task_df[score_colname].values)
            best_score_idx[tt] = score_ranking_func(task_df[score_colname].values)
            best_score[tt] = task_df[score_colname].values[best_score_idx[tt]]
        best_task_idx = score_ranking_func(best_score)
        best_task_df = job_df.query("TASKID == '%s'" % taskids[best_task_idx])
        best_task_df_list.append(best_task_df)
        if verbose:
            pprint(job_path)
            pprint("best task: %s" % best_task_idx)
    return pd.concat(best_task_df_list)


def load_df_from_all_folders_that_match_pattern(
        src_path_pattern='',
        task_ids='1',
        when_task_path_does_not_exist='continue',
        when_split_csv_does_not_exist='raise_error',
        y_ind=0,
        column_names=None,
        output_data_type='binary',
        engine=None,
        csv_pattern='snapshot_perf_metrics_%s.csv'):
    ''' Load results from many folders that match a pattern into data frame.

    Aggregates results from many pipelines.

    Returns
    -------
    df : pandas DataFrame
    '''
    src_path_list = [s for s in sorted(glob.glob(src_path_pattern))]
    mega_df = None
    df_list = list()
    column_names = load_default_column_name_dict(
        output_data_type=output_data_type)
    for src_path in src_path_list:
        df = load_df_from_training_results_folder(
            src_path=src_path, 
            task_ids=task_ids,
            when_task_path_does_not_exist=when_task_path_does_not_exist,
            when_split_csv_does_not_exist=when_split_csv_does_not_exist,
            column_names=column_names,
            engine=engine,
            csv_pattern=csv_pattern,
            y_ind=y_ind)
        df_list.append(df)
    mega_df = pd.concat(df_list)
    return mega_df


def load_df_from_training_results_folder(
        src_path='',
        task_ids='1',
        when_task_path_does_not_exist='raise_error',
        when_split_csv_does_not_exist='raise_error',
        column_names=COLUMN_NAMES,
        y_ind=0,
        engine=None,
        csv_pattern='callback_%s_info.csv',
        ):
    ''' Load results from plain-text files into data frame

    Returns
    -------
    df : pandas DataFrame
    '''
    if isinstance(task_ids, str) or isinstance(task_ids, int):
        task_ids = [str(task_ids)]
    else:
        task_ids = [str(tid) for tid in task_ids]

    df_dict = dict()

    # Parse fields from folder name
    # which will usually look like
    # fromscratch-model=lda-n_topics=10-...
    path_parts = src_path.rstrip(os.path.sep).split(os.path.sep)
    basename = path_parts[-1]
    fields = basename.split('-')
    mfields = list()
    fid = 0

    while fid <= len(fields) - 1:
        # deal with minus signs in numeric values
        # like "value=1.0e-3"
        if fields[fid].endswith('e') and fields[fid+1][0].isdigit():
            mfields.append(fields[fid] + '-' + fields[fid+1])
            fid += 2
        else:
            mfields.append(fields[fid])
            fid += 1

    fields = mfields
    for field in fields:
        if field.count('=') == 0:
            continue
        name, value = field.split('=')
        if name.upper() in column_names:
            try:
                df_dict[name.upper()] = float(value)
            except ValueError:
                df_dict[name.upper()] = str(value)

    mega_row_dict_list = list()
    for task_id in task_ids:
        assert isinstance(task_id, str) or isinstance(task_id, unicode)
        task_path = os.path.join(src_path, task_id)
        if not os.path.exists(task_path):
            if when_task_path_does_not_exist.count('error'):
                raise IOError("Task path does not exist:\n" + task_path)
            else:
                continue
        df_dict['JOB_PATH'] = src_path
        df_dict['TASK_PATH'] = task_path
        df_dict['TASKID'] = task_id

        df_by_split = dict()
        n_rows = np.inf
        for split_name in SPLIT_NAMES:
            cur_split_csv_path = os.path.join(
                task_path, csv_pattern % split_name)
            try:
                df_by_split[split_name] = pd.read_csv(
                    cur_split_csv_path, engine=engine)
                n_rows = np.minimum(n_rows, df_by_split[split_name].shape[0])  
            except IOError as e:
                if when_split_csv_does_not_exist.count('error'):
                    raise e
                else:
                    df_by_split[split_name] = None
        assert np.isfinite(n_rows)
        n_rows = int(n_rows)
        row_dict_list = list()
        n_used_splits = 0
        for split_name in SPLIT_NAMES:
            split_df = df_by_split.get(split_name, None)

            if split_df is None:
                continue
            n_used_splits += 1
            ## Truncate all splits to same num rows
            # Can happen when we're in middle of training
            # and the _train.csv file is written to before _test.csv
            split_df = split_df[:n_rows]

            for _, row_df in split_df.iterrows():
                row_dict = df_dict.copy()
                row_dict['SPLIT_NAME'] = split_name.upper()
                for key, val in row_df.to_dict().items():
                    key = key.strip().upper()
                    if key.startswith("Y_"):
                        target_ind_prefix = "Y_%d_" % y_ind
                        if not key.startswith(target_ind_prefix):
                            # Skip Y_%d values that arent the target ind
                            continue
                        key = key.replace(target_ind_prefix, 'Y_')
                    row_dict[key] = val
                row_dict_list.append(row_dict)
        assert len(row_dict_list) == n_rows * n_used_splits
        mega_row_dict_list.extend(row_dict_list)

    # Aggregate into giant df and return
    df = pd.DataFrame(mega_row_dict_list)
    for column_name in column_names:
        if column_name not in df.columns:
            df[column_name] = np.nan * np.ones(df.shape[0])
    for column in df.columns:
        if column not in column_names:
            del df[column]
    for column_name in column_names:
        col_type_str = column_names[column_name]
        if col_type_str == 'integer':
            df[column_name] = df[column_name].astype(np.int32)
        elif col_type_str == 'number':
            df[column_name] = df[column_name].astype(np.float64)

    return df


def simplify_best_df_and_make_unicode_friendly(
        best_df,
        replacements={'WEIGHT_Y':'λ', '==':'=', "'":""},
        replacements_ascii={'λ':'WEIGHT_Y', '=':''},
        at_best_keys=[
            'LOGPDF_X_PERTOK_AT_BEST_SNAPSHOT',
            'LOGPDF_Y_PERDOC_AT_BEST_SNAPSHOT'],
        ):
    ''' Update legend names to be shorter/unicode friendly.

    Also adds _AT_BEST_SNAPSHOT fields
    '''
    legcolid = best_df.columns.tolist().index('LEGEND_NAME')
    best_df["LEGEND_NAME"] = best_df["LEGEND_NAME"].apply(lambda x: unicode(x))
    best_df["LEGEND_NAME_ASCII"] = best_df["LEGEND_NAME"].apply(lambda x: unicode(x))
    for row_id in range(best_df.shape[0]):
        leg_str = best_df.iloc[row_id, legcolid]
        for before, after in replacements.items():
            leg_str = leg_str.replace(before, after)
        leg_str = ' '.join(leg_str.split())
        best_df.iloc[row_id, legcolid] = leg_str

        # Now make ascii-safe version of each name
        leg_str_ascii = leg_str
        for before, after in replacements_ascii.items():
            leg_str_ascii = leg_str_ascii.replace(before, after)
        best_df.loc[row_id, 'LEGEND_NAME_ASCII'] = (
            ' '.join(leg_str_ascii.decode('ascii').split())).replace(' ', '_')
        
    at_best_row_mask = best_df.IS_BEST_SNAPSHOT.values > 0
    for leg in np.unique(best_df['_UNIQUE_LEGEND_NAME'].values):
        for split in np.unique(best_df['SPLIT_NAME'].values):
            leg_split_row_mask = np.logical_and(
                best_df._UNIQUE_LEGEND_NAME.values == leg,
                best_df.SPLIT_NAME.values == split)
            best_leg_split_row_mask = np.logical_and(
                at_best_row_mask, leg_split_row_mask)

            assert np.sum(best_leg_split_row_mask) == 1
            assert np.sum(best_leg_split_row_mask) < np.sum(leg_split_row_mask)
            for at_best_key in at_best_keys:
                target_key = at_best_key.replace('_AT_BEST_SNAPSHOT', '')
                best_leg_split_row_id = np.flatnonzero(best_leg_split_row_mask)[0]
                val_at_best = best_df[target_key].values[best_leg_split_row_id]
                best_df.loc[leg_split_row_mask, at_best_key] = val_at_best

    # Verify all row indices are unique
    assert best_df.shape[0] == np.unique(best_df.index.values).size

    return best_df


def read_args_from_stdin_and_run():
    ''' Main executable function to select best runs and snapshots
    '''
    if not sys.stdin.isatty():
        for line in sys.stdin.readlines():
            line = line.strip()
            sys.argv.extend(line.split(' '))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results_path_patterns',
        default='/tmp/',
        type=str,
        help="Comma-separated list of strings to be given to glob")
    parser.add_argument(
        '--txt_src_path',
        default='/tmp/',
        type=str,
        help="Path to folder containing X_colnames.txt and Y_colnames.txt")
    parser.add_argument(
        '--output_path',
        default='/tmp/',
        type=str,
        help="Path to folder to hold output")
    parser.add_argument(
        '--legend_name',
        default='Gibbs_LDA',
        help='Short name')
    parser.add_argument(
        '--target_y_name', 
        default='',
        type=str,
        help='Name of target outcome variable')
    parser.add_argument(
        '--all_y_names', 
        default='',
        type=str,
        help="Comma-separated list of all possible outcome variables")
    parser.add_argument(
        '--selection_score_colname',
        default='Y_ROC_AUC',
        type=str,
        help="Name of csv column used to select best runs",
        )
    parser.add_argument(
        '--selection_score_ranking_func',
        default=None,
        type=str,
        choices=['argmax', 'argmin', None],
        help="Name of ranking function used to select best runs",
        )    
    parser.add_argument(
        '--col_names_to_use_at_selection',
        default='N_STATES,FRAC_LABELS',
        type=str,
        help="Name of csv columns for which unique values each get separate best",
        )
    parser.add_argument(
        '--col_names_to_keep_per_split',
        default='',
        type=str,
        help=("Name of csv columns to keep in resulting best_snapshots.csv"
            + " Will be queried at each possible split"),
        )
    parser.add_argument(
        '--col_names_to_keep',
        default='',
        type=str,
        help="Name of csv columns to keep in resulting best_snapshots.csv",
        )
    # Parse the input args
    args, unk_list = parser.parse_known_args()
    arg_dict = vars(args)

    # Run the main function with parsed args
    select_best_from_many_runs(
        unk_list=unk_list,
        **arg_dict)

if __name__ == '__main__':
    read_args_from_stdin_and_run()
