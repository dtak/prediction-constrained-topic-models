import os
import numpy as np

import scipy.sparse
from distutils.dir_util import mkpath
from sklearn.externals import joblib

from pc_toolbox.utils_io import update_symbolic_link

def save_topic_model_param_dict(
        param_dict,
        output_path=None,
        param_output_fmt='dump',
        disable_output=False,
        **alg_state_kwargs):
    """ Save snapshot of topic model parameters to disk

    Returns
    -------
    snapshot_path : path to where results were saved.
    """
    snapshot_path = None
    if output_path is not None and (not disable_output):
        cur_lap = alg_state_kwargs['cur_lap']
        if param_output_fmt.count('dump'):
            best_filepath = os.path.join(
                output_path, 'best_param_dict.dump')
            cur_filepath = os.path.join(
                output_path, 'lap%011.3f_param_dict.dump' % (cur_lap))
            joblib.dump(param_dict, cur_filepath, compress=1)
            update_symbolic_link(cur_filepath, best_filepath)

        if param_output_fmt.count('topic_model_snapshot'):
            prefix = 'lap%011.3f' % cur_lap
            snapshot_path = save_topic_model_params_as_txt_files(
                output_path,
                prefix, 
                **param_dict)
            best_path = snapshot_path.replace(prefix, 'best')
            if best_path.count('best') > 0:
                update_symbolic_link(snapshot_path, best_path)
            else:
                raise ValueError("Bad path: " + snapshot_path)
    return snapshot_path

def load_topic_model_param_dict(
        snapshot_path=None,
        task_path=None,
        prefix='best',
        lap=None,
        w_txt_basename='w_CK.txt',
        add_bias_term_to_w_CK=0.0,
        **kwargs):
    ''' Load topic model parameters from disk.

    Supports either dump file or folder of txt files

    Returns
    -------
    param_dict : dict with fields
        * topics_KV : 2D array, K x V
        * w_CK : 2D array, C x K
    '''
    if snapshot_path is None:
        if lap is not None:
            prefix = 'lap%011.3f' % float(lap)
        assert prefix is not None

        for pprefix in [prefix, prefix + "_param_dict.dump"]:
            try:
                dump_path = os.path.join(task_path, pprefix)
                param_dict = joblib.load(dump_path)
                return param_dict
            except IOError as e:
                pass
        snapshot_path = os.path.join(
            task_path,
            prefix + "_topic_model_snapshot")
    try:
        param_dict = joblib.load(snapshot_path)
        return param_dict
    except IOError:
        pass

    try:
        tau = float(np.loadtxt(os.path.join(snapshot_path, 'tau.txt')))
    except IOError:
        if 'tau' in kwargs:
            tau = float(kwargs['tau'])
        else:
            tau = None
    try:
        alpha = float(np.loadtxt(os.path.join(snapshot_path, 'alpha.txt')))
    except IOError:
        if 'alpha' in kwargs:
            alpha = float(kwargs['alpha'])
        else:
            alpha = None
    try:
        lambda_w = float(np.loadtxt(os.path.join(snapshot_path, 'lambda_w.txt')))
    except IOError:
        if 'lambda_w' in kwargs:
            lambda_w = float(kwargs['lambda_w'])
        else:
            lambda_w = None

    try:
        topics_KV = np.loadtxt(
            os.path.join(snapshot_path, 'topics_KV.txt'))
    except IOError:
        csr_prefix = 'topic_word_count_csr'
        Q = dict()
        for suffix in ['data', 'indices', 'indptr', 'shape']:
            csr_fpath = '%s_%s.txt' % (csr_prefix, suffix)
            Q[suffix] = np.loadtxt(os.path.join(snapshot_path, csr_fpath))
        topic_count_KV = scipy.sparse.csr_matrix(
            (Q['data'], Q['indices'], Q['indptr']),
            shape=Q['shape'])
        topics_KV = topic_count_KV.toarray().copy()
        del Q
        topics_KV += tau
        topics_KV /= topics_KV.sum(axis=1)[:,np.newaxis]

    try:
        w_txt_fpath = os.path.join(snapshot_path, w_txt_basename)
        if w_txt_basename != 'w_CK.txt':
            if os.path.exists(w_txt_fpath):
                print "  USING w_txt_basename:", w_txt_basename
            else:
                print "  FALLING BACK TO w_CK.txt"
                w_txt_fpath = os.path.join(snapshot_path, 'w_CK.txt')
        w_CK = np.loadtxt(w_txt_fpath)
        if w_CK.ndim == 1:
            w_CK = w_CK[np.newaxis,:].copy()

        if add_bias_term_to_w_CK != 0.0:
            K = w_CK.shape[1]
            w_CK = w_CK - add_bias_term_to_w_CK
    except IOError:
        w_CK = None
    return dict(
        topics_KV=topics_KV,
        w_CK=w_CK,
        tau=tau,
        alpha=alpha,
        lambda_w=lambda_w)

def save_topic_model_params_as_txt_files(
        output_path=None,
        prefix='',
        topics_KV=None,
        w_CK=None,
        pi_DK=None,
        **kwargs):
    snapshot_path = os.path.join(
        output_path, 
        prefix + "_topic_model_snapshot")
    mkpath(snapshot_path)
    np.savetxt(
        os.path.join(snapshot_path, 'topics_KV.txt'),
        topics_KV,
        fmt='%.11f',
        delimiter=' ')
    if w_CK is not None:
        np.savetxt(
            os.path.join(snapshot_path, 'w_CK.txt'),
            w_CK,
            fmt='%.9f',
            delimiter=' ')
    if pi_DK is not None:
        np.savetxt(
            os.path.join(snapshot_path, 'pi_DK.txt'),
            pi_DK,
            fmt='%.6f',
            delimiter=' ')
    for key in kwargs:
        if key.endswith('_param_dict'):
            fpath = os.path.join(snapshot_path, key + ".dump")
            joblib.dump(kwargs[key], fpath, compress=1)

    return snapshot_path
