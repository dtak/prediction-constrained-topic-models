'''
eval_pretrained_sklearn_binary_classifier.py

Usage
-----
$ python eval_pretrained_sklearn_binary_classifier.py \
    --dataset_path [path] \
    --pretrained_clf_path [path] \
    [optional args]

Optional arguments
------------------
--dataset_path DATASET_PATH
                        Path to folder containing:
                            *.npy files: X_train, y_train, P_train
                            *.txt files: X_colnames.txt, y_colnames.txt
--pretrained_clf_path OUTPUT_PATH
                        Path to folder holding output from this evaluator.
                        Includes:
                            * clf_<id>_.dump : loadable clf object
                            * clf_<id>_callback_train.csv : perf metrics
'''

from __future__ import print_function

import numpy as np
import pandas as pd
import datetime
import sys
import os
import argparse
import itertools
import time
import scipy.sparse
import glob

from collections import OrderedDict
from distutils.dir_util import mkpath

from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer, Binarizer
from sklearn.pipeline import Pipeline
from pc_toolbox.utils_io import (
    load_csr_matrix, pprint, config_pprint_logging,
    load_list_of_strings_from_txt,
    load_list_of_unicode_from_txt,
    )

from train_and_eval_sklearn_binary_classifier import (
    make_constructor_and_evaluator_funcs,
    ThresholdClassifier,
    )

from calc_roc_auc_via_bootstrap import calc_binary_clf_metric_with_ci_via_bootstrap

def read_args_from_stdin_and_run():
    ''' Main executable function to train and evaluate classifier.

    Post Condition
    --------------
    AUC and other eval info printed to stdout.
    Trained classifier saved ???.
    '''
    if not sys.stdin.isatty():
        for line in sys.stdin.readlines():
            line = line.strip()
            sys.argv.extend(line.split(' '))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path',
        default='/tmp/',
        type=str,
        help="Path to folder containing:" +
        " *.npy files: X_train, y_train, P_train"
        " *.txt files: X_colnames.txt and y_colnames.txt")
    parser.add_argument(
        '--pretrained_clf_path',
        default='/tmp/',
        type=str,
        help="Path to folder to hold output from classifier. Includes:" +
        " perf_metric*.txt files: auc_train.txt & auc_test.txt" +
        " settings.txt: description of all settings to reproduce.")
    parser.add_argument(
        '--output_path',
        default=None,
        type=str,
        help='Place to write output files. Defaults to pretrained_clf_path if not specified.')
    parser.add_argument(
        '--split_names',
        default='test')
    parser.add_argument(
        '--split_nicknames',
        default=None)
    parser.add_argument(
        '--features_path',
        default='/tmp/',
        type=str,
        help="Path to folder with SSAMfeat*.npy files")
    parser.add_argument(
        '--target_arr_name',
        default='Y',
        type=str,
        )
    parser.add_argument(
        '--target_names',
        default='all',
        type=str,
        help='Name of response/intervention to test.' +
        ' To try specific interventions, write names separated by commas.' +
        ' To try all interventions, use special name "all"')
    parser.add_argument(
        '--top_k',
        default=3, type=int,
        help='Num of predictions to take to compute hitrate')
    parser.add_argument(
        '--oracle_dataset_path',
        default=None, type=str,
        help='Path to train arrays used as oracle knowledge for predictions')
    parser.add_argument(
        '--seed_bootstrap',
        default=42,
        type=int,
        help='Seed for bootstrap')
    parser.add_argument(
        '--n_bootstraps',
        default=5000,
        type=int,
        help='Number of samples for bootstrap conf. intervals')
    parser.add_argument(
        '--bootstrap_stratify_pos_and_neg',
        default=True,
        type=int,
        help='Whether to stratify examples or not')

    args, unk_list = parser.parse_known_args()
    arg_dict = vars(args)

    dataset_path = arg_dict['dataset_path']
    if arg_dict['output_path'] is None:
        output_path = arg_dict['pretrained_clf_path']
    else:
        output_path = arg_dict['output_path']
        if not os.path.exists(output_path):
            mkpath(output_path)

    if arg_dict['oracle_dataset_path'] is None:
        arg_dict['oracle_dataset_path'] = dataset_path

    # Set default seed for numpy
    # Might be used by cross-validation, etc.
    np.random.seed(8675309)

    clf_opt_list = list()
    # Read parsed args from plain-text file
    # so we can exactly reproduce pretrained classifier
    try:
        with open(os.path.join(arg_dict['pretrained_clf_path'], 'settings.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip()
                clf_opt_list.append(line.split(' = '))
        clf_opts = dict(clf_opt_list)
    except IOError as e:
        clf_opts = dict()
        clf_opts['feature_arr_names'] = 'X' # DUMMY!
        clf_opts['classifier_name'] = arg_dict['pretrained_clf_path']

    feat_path_list = [
        arg_dict['dataset_path'],
        arg_dict['features_path']]

    pprint('[run_classifier says:] Loading dataset ...')
    start_time = time.time()
    feature_arr_names = clf_opts['feature_arr_names'].split(',')
    pprint('feature_arr_names:')
    feat_colnames_by_arr = OrderedDict()
    for feat_arr_name in feature_arr_names:
        pprint(feat_arr_name)
        cur_feat_colnames = None
        for feat_path in feat_path_list:
            colname_fpath = os.path.join(
                feat_path,
                feat_arr_name + '_colnames.txt')
            if os.path.exists(colname_fpath):
                cur_feat_colnames = \
                    [unicode(feat_arr_name + ":") + s
                        for s in load_list_of_unicode_from_txt(colname_fpath)]
                break
        feat_colnames_by_arr[feat_arr_name] = cur_feat_colnames

    target_arr_name = arg_dict['target_arr_name']
    all_target_names = load_list_of_strings_from_txt(os.path.join(
        arg_dict['dataset_path'],
        target_arr_name + '_colnames.txt'))
    target_names = arg_dict['target_names']
    if target_names == 'all':
        target_names = all_target_names
        target_cols = np.arange(len(all_target_names)).tolist()
    else:
        target_names = target_names.split(',')
        target_cols = list()
        for name in target_names:
            assert name in all_target_names
            target_cols.append(all_target_names.index(name))

    datasets_by_split = OrderedDict()
    split_names = arg_dict['split_names'].split(',')
    if arg_dict['split_nicknames'] is None:
        split_nicknames = split_names
    else:
        split_nicknames = arg_dict['split_nicknames'].split(',')

    alias_by_split = OrderedDict()
    for nickname, split_name in zip(split_nicknames,split_names):
        datasets_by_split[nickname] = dict()
        split_dataset = datasets_by_split[nickname]
        alias_by_split[nickname] = split_name

        # Load Y
        dense_fpath = os.path.join(
            dataset_path,
            target_arr_name + "_%s.npy" % split_name)
        y = np.asarray(np.load(dense_fpath), order='C', dtype=np.int32)
        if y.ndim < 2:
            y = y[:,np.newaxis]
        assert y.ndim == 2
        assert y.shape[1] == len(all_target_names)
        split_dataset['y'] = y[:, target_cols]
        assert split_dataset['y'].shape[1] == len(target_cols)

        # Load X
        x_list = list()      
        for feat_arr_name in feature_arr_names:
            x_cur = None

            def fpath_generator():
                for feat_path in feat_path_list:
                    for sname in [nickname, split_name]:
                        dense_fpath = os.path.join(
                            feat_path, feat_arr_name + "_" + sname + ".npy")
                        sparse_fpath = os.path.join(
                            feat_path, feat_arr_name + "_csr_" + sname + ".npz")
                        yield dense_fpath, sparse_fpath
            ds_path_list = [pair for pair in fpath_generator()] 
            for ii, (dense_fpath, sparse_fpath) in enumerate(ds_path_list):
                try:
                    if os.path.exists(sparse_fpath):
                        x_cur = load_csr_matrix(sparse_fpath)
                        assert np.all(np.isfinite(x_cur.data))
                        break
                    else:
                        x_cur = np.asarray(
                            np.load(dense_fpath),
                            order='C', dtype=np.float64)
                        if x_cur.ndim < 2:
                            x_cur = np.atleast_2d(x_cur).T
                        assert np.all(np.isfinite(x_cur))
                        break
                except IOError as e:
                    if ii == len(ds_path_list) - 1:
                        # Couldn't find desired file in any feat_path
                        raise e
                    else:
                        # Try the next feat_path in the list
                        pass
            if x_cur is not None:
                if feat_colnames_by_arr[feat_arr_name] is not None:
                    feat_dim = len(feat_colnames_by_arr[feat_arr_name])
                    assert x_cur.shape[1] == feat_dim
                else:
                    # Add dummy colnames
                    feat_dim = x_cur.shape[1]
                    n_sig_digits = np.maximum(
                        3, int(np.ceil(np.log10(feat_dim))))
                    fmt_str = "%s_%0" + str(n_sig_digits) + "d"
                    feat_colnames_by_arr[feat_arr_name] = [
                        fmt_str % (feat_arr_name, fid)
                            for fid in range(feat_dim)]
                x_list.append(x_cur)

        if isinstance(x_list[0], np.ndarray):
            split_dataset['x'] = np.hstack(x_list)
        else:
            split_dataset['x'] = scipy.sparse.hstack(x_list, format='csr')

        assert split_dataset['x'].ndim == 2
        assert split_dataset['x'].shape[0] == split_dataset['y'].shape[0]
        assert (
            isinstance(split_dataset['x'], np.ndarray)
            or isinstance(split_dataset['x'], scipy.sparse.csr_matrix)
            )

        if split_name == split_names[0]:
            # Flatten feat colnames into single list
            feat_colnames = sum(feat_colnames_by_arr.values(), [])
            assert isinstance(feat_colnames, list)
            assert len(feat_colnames) == split_dataset['x'].shape[1]

            print('y colnames: %s' % ' '.join(target_names))
            if len(feat_colnames) > 10:
                print('x colnames: %s ... %s' % (' '.join(feat_colnames[:5]), ' '.join(feat_colnames[-5:])))
            else:
                print('x colnames: %s' % ' '.join(feat_colnames))

        print('---- %5s dataset summary' % split_name)
        print('%9d total examples' % y.shape[0])
        print('y : %d x %d targets' % split_dataset['y'].shape)
        print('x : %d x %d features' % split_dataset['x'].shape)

        for c in xrange(len(target_names)):
            y_c = split_dataset['y'][:,c]
            print('target %s : frac pos %.3f' % (target_names[c], np.mean(y_c)))
            print('    %6d pos examples' % np.sum(y_c == 1))
            print('    %6d neg examples' % np.sum(y_c == 0))

    elapsed_time = time.time() - start_time
    print('[run_classifier says:] dataset loaded after %.2f sec.' % elapsed_time)

    print('[eval_pretrained_classifier says:] eval multilabel')
    eval_pretrained_avg_clf(
        classifier_name=clf_opts['classifier_name'],
        classifier_path=arg_dict['pretrained_clf_path'],
        datasets_by_split=datasets_by_split,
        alias_by_split=alias_by_split,
        dataset_path=arg_dict['dataset_path'],
        oracle_dataset_path=arg_dict['oracle_dataset_path'],
        feat_colnames=feat_colnames,
        output_path=output_path,
        target_arr_name=target_arr_name,
        label_names=all_target_names,
        feature_arr_names=feature_arr_names,
        seed_bootstrap=arg_dict['seed_bootstrap'],
        n_bootstraps=arg_dict['n_bootstraps'],
        bootstrap_stratify_pos_and_neg=arg_dict['bootstrap_stratify_pos_and_neg'],
        )
    elapsed_time = time.time() - start_time
    print('[eval_pretrained_classifier says:] completed after %.2f sec' % (elapsed_time))


def eval_pretrained_avg_clf(
        classifier_path='/tmp/',
        classifier_name='logistic_regression',
        target_arr_name='Y',
        label_names=None,
        feature_arr_names=[],
        dataset_path=None,
        oracle_dataset_path=None,
        datasets_by_split=None,
        alias_by_split=None,
        verbose=True,
        feat_colnames=None,
        output_path='/tmp/',
        top_k=3,
        n_bootstraps=1000,
        seed_bootstrap=42,
        bootstrap_stratify_pos_and_neg=False,
        ):
    start_time = time.time()
    feature_arr_name = ','.join(feature_arr_names)
    # Read classifier obj from disk
    some_key = datasets_by_split.keys()[0]
    n_label = datasets_by_split[some_key]['y'].shape[1]
    if os.path.exists(classifier_path):
        clf_path_pattern = os.path.join(
                classifier_path,
                'clf_*_object.dump')
        clf_path_files = glob.glob(clf_path_pattern)
        assert n_label > 1
        if len(clf_path_files) == 0:
            raise ValueError("Bad clf_path_pattern: %s" % (
                clf_path_files))

        elif len(clf_path_files) == n_label:
            clf_list = list()
            for yid in range(n_label):
                clf_path = os.path.join(
                    classifier_path,
                    'clf_%d_object.dump' % (yid))
                assert clf_path in clf_path_files
                clf = joblib.load(clf_path)
                clf_list.append(clf)
            def clf_avg_func(x_ND, split_name=None, row_ids=None):
                N = x_ND.shape[0]
                yhat_NC = np.zeros((N, n_label), dtype=np.float64)
                for c in range(n_label):
                    yhat_NC[:,c] = clf_list[c].predict_proba(x_ND)[:,1]
                return yhat_NC
        else:
            raise ValueError("SHOULDNT BE HERE")

    # Evaluate
    for ss, split in enumerate(datasets_by_split.keys()):
        y_hat_NC = clf_avg_func(datasets_by_split[split]['x'])
        y_true_NC = datasets_by_split[split]['y']

        def calc_avg_AUC(y_true_NC, y_hat_NC):
            C = y_true_NC.shape[1]
            auc_C = np.zeros(C)
            for c in range(C):
                auc_C[c] = roc_auc_score(y_true_NC[:,c], y_hat_NC[:,c])
            #print(' '.join(['%.3f' % x for x in auc_C]))
            return np.mean(auc_C)

        ci_tuples = [(2.5,97.5), (10,90)]
        avg_auc, auc_intervals = calc_binary_clf_metric_with_ci_via_bootstrap(
            y_pred=y_hat_NC,
            y_true=y_true_NC,
            metric_func=calc_avg_AUC,
            ci_tuples=ci_tuples,
            stratify_pos_and_neg=bootstrap_stratify_pos_and_neg,
            n_bootstraps=n_bootstraps,
            seed=seed_bootstrap)
        row_dict = OrderedDict([
            ('Y_COL_NAME', 'avg'),
            ('SPLIT_NAME', split.upper()),
            ('Y_ROC_AUC', avg_auc),
            ('Y_ERROR_RATE', np.nan),
            ('Y_F1_SCORE', np.nan),
            ('LEGEND_NAME', classifier_name),
            ('IS_BEST_SNAPSHOT', 1),
            ])
        for ci_tuple, auc_tuple in zip(ci_tuples, auc_intervals):
            for perc, val in zip(ci_tuple, auc_tuple):
                row_dict['Y_ROC_AUC_CI_%05.2f%%' % perc] = val

        for ci_tuple, f1_tuple in zip(ci_tuples, auc_intervals):
            for perc, val in zip(ci_tuple, f1_tuple):
                row_dict['Y_F1_SCORE_CI_%05.2f%%' % perc] = np.nan

        csv_df = pd.DataFrame([row_dict], columns=row_dict.keys())
        csv_fpath = os.path.join(
            output_path,
            'clf_avg_callback_%s.csv' % (split))
        csv_df.to_csv(
                csv_fpath,
                index=False)
        if verbose:
            elapsed_time = time.time() - start_time
            print("eval on %5s split done after %11.2f sec" % (split, elapsed_time))
            print("wrote csv file: " + csv_fpath)
 
if __name__ == '__main__':
    read_args_from_stdin_and_run()
