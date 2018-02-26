import os
import scipy.sparse
import numpy as np

from pc_toolbox.utils_io import (
    save_csr_matrix,
    load_list_of_unicode_from_txt,
    make_percentile_str,
    )

def load_dataset(
        dataset_path=None,
        split_name='train',
        frac_labels_train=1.0,
        data_seed=42,
        return_info=False,
        **kwargs):
    """ Load sLDA dataset from disk.

    Returns
    -------
    dataset : dict, with keys
        x_csr_DV : sparse 2D array, size D x V (n_docs x n_vocabs)
        y_DC : dense 2D array, size D x C (n_docs x n_labels)
    info_dict : dict
        Contains extra information about dataset (column names, etc)
    """
    # Load X data
    x_csr_DV = load_x_csr(dataset_path, split_name)
    dataset_dict = dict(
        n_docs=x_csr_DV.shape[0],
        n_vocabs=x_csr_DV.shape[1],
        x_csr_DV=x_csr_DV,
        word_id_U=x_csr_DV.indices,
        word_ct_U=x_csr_DV.data,
        doc_indptr_Dp1=x_csr_DV.indptr)

    # Load Y data
    y_path = os.path.join(dataset_path, 'Y_%s.npy' % split_name)
    if os.path.exists(y_path):
        y_DC = np.load(y_path).astype(np.float64)
        if y_DC.ndim < 2:
            y_DC = y_DC[:,np.newaxis]
        dataset_dict['y_DC'] = y_DC    
        dataset_dict['n_labels'] = y_DC.shape[1]
        n_labels = y_DC.shape[1]

        # Detect truly missing rows
        y_finite_rowmask = np.all(np.isfinite(y_DC), axis=1).astype(np.int32)
        if not np.all(y_finite_rowmask):
            dataset_dict['y_rowmask'] = y_finite_rowmask
    else:
        n_labels = None
    
    if split_name == 'train' and 'y_DC' in dataset_dict:
        frac_labels_train = float(frac_labels_train)
        if frac_labels_train >= 1.0 or frac_labels_train <= 0.0:
            pass
        else:
            data_prng = np.random.RandomState(int(data_seed))
            n_total_rows = dataset_dict['y_DC'].shape[0]
            if 'y_rowmask' in dataset_dict:
                # Case where some rows are already truly missing
                # and we want to artificially mask even more
                eligible_rows = np.flatnonzero(dataset_dict['y_rowmask'] == 1)
            else:
                # Case where no rows are already missing
                # so all rows are eligible
                eligible_rows = np.arange(n_total_rows)
            n_eligible_rows = eligible_rows.size
            shuffled_rows = data_prng.permutation(eligible_rows)
            n_visible = int(np.ceil(frac_labels_train*n_eligible_rows))
            visible_rows = shuffled_rows[:n_visible]
            rowmask = np.zeros(n_total_rows, dtype=np.int32)
            rowmask[visible_rows] = 1
            assert np.allclose(
                frac_labels_train * (n_eligible_rows / float(n_total_rows)),
                np.sum(rowmask == 1) / float(rowmask.size),
                atol=0.01)
            dataset_dict['y_rowmask'] = rowmask

    if return_info:
        n_vocabs = x_csr_DV.shape[1]
        info_dict = dict(
            dataset_path=dataset_path,
            slice_dataset=slice_dataset,
            n_vocabs=n_vocabs,
            n_labels=n_labels)
        x_txt_path = os.path.join(dataset_path, 'X_colnames.txt')
        try:
            vocab_list = load_list_of_unicode_from_txt(x_txt_path)
            assert len(vocab_list) == n_vocabs
            info_dict['vocab_list'] = vocab_list
        except:
            pass
        y_txt_path = os.path.join(dataset_path, 'Y_colnames.txt')
        try:
            label_list = load_list_of_unicode_from_txt(y_txt_path)
            assert len(label_list) == n_labels
            info_dict['label_list'] = label_list
        except:
            pass
        # Wrap up and return
        return dataset_dict, info_dict
    else:
        return dataset_dict


def save_dataset(
        dataset=None,
        output_path=None,
        split_name='train',
        y_arr_prefix='Y',
        x_arr_prefix='X_csr',
        ):
    x_arr_path = os.path.join(
        output_path,
        '%s_%s.npz' % (x_arr_prefix, split_name))
    y_arr_path = os.path.join(
        output_path,
        '%s_%s.npy' % (y_arr_prefix, split_name))
    np.save(y_arr_path, dataset['y_DC'])
    save_csr_matrix(x_arr_path, dataset['x_csr_DV'])
    return x_arr_path, y_arr_path


def slice_dataset(
        cur_slice=None, dataset=None, 
        max_n_examples_per_slice=np.inf,
        include_rowmask=True,
        **kwargs):
    ''' Create slice of provided dataset.

    Returns
    -------
    slice_dict : dict of subset of data
    '''
    if cur_slice is None:
        cur_slice = slice(0, dataset['n_docs'])
    if cur_slice.stop - cur_slice.start > max_n_examples_per_slice:
        cur_slice = slice(
            cur_slice.start,
            cur_slice.start + max_n_examples_per_slice)
    
    n_vocabs = dataset['n_vocabs']
    doc_indptr_Dp1 = dataset['doc_indptr_Dp1']
    word_id_U = dataset['word_id_U']
    word_ct_U = dataset['word_ct_U']
    u_start = doc_indptr_Dp1[cur_slice.start]
    u_stop = doc_indptr_Dp1[cur_slice.stop]
    d_start = cur_slice.start
    d_stop = cur_slice.stop

    n_docs = cur_slice.stop - cur_slice.start
    word_id_U = word_id_U[u_start:u_stop]
    word_ct_U = word_ct_U[u_start:u_stop]
    doc_indptr_Dp1 = doc_indptr_Dp1[d_start:(d_stop + 1)].copy()
    doc_indptr_Dp1 = doc_indptr_Dp1 - doc_indptr_Dp1[0]

    slice_dict = dict(
        n_docs=n_docs,
        n_vocabs=n_vocabs,
        word_id_U=word_id_U,
        word_ct_U=word_ct_U,
        doc_indptr_Dp1=doc_indptr_Dp1,
        x_csr_DV=scipy.sparse.csr_matrix(
            (word_ct_U, word_id_U, doc_indptr_Dp1),
            shape=(n_docs, n_vocabs)),
        )
    if 'y_DC' in dataset:
        slice_dict['y_DC'] = dataset['y_DC'][cur_slice]
        slice_dict['n_labels'] = slice_dict['y_DC'].shape[1]
    if include_rowmask and 'y_rowmask' in dataset:
        slice_dict['y_rowmask'] = dataset['y_rowmask'][cur_slice]
    elif 'y_rowmask' in dataset:
        # handle case where some things are real nan values
        y_rowmask = np.all(np.isfinite(dataset['y_DC'][cur_slice]), axis=1).astype(np.int32)
        if np.sum(y_rowmask) < y_rowmask.size:
            slice_dict['y_rowmask'] = y_rowmask
    return slice_dict

def make_dataset_subset(doc_ids=None, dataset=None, **kwargs):
    ''' Make dataset using subset of the examples in a larger dataset.

    Returns
    -------
    subset : dict
    '''
    doc_ids = np.asarray(doc_ids, dtype=np.int32)

    n_vocabs = dataset['n_vocabs']
    doc_indptr_Dp1 = dataset['doc_indptr_Dp1']
    word_id_U = dataset['word_id_U']
    word_ct_U = dataset['word_ct_U']

    subset_wid_list = list()
    subset_wct_list = list()
    doclen_list = list()
    for d in doc_ids:
        start_d = doc_indptr_Dp1[d]
        stop_d = doc_indptr_Dp1[d+1]
        subset_wid_list.append(word_id_U[start_d:stop_d])
        subset_wct_list.append(word_ct_U[start_d:stop_d])
        doclen_list.append(stop_d - start_d)
    subset = dict(
        word_id_U=np.hstack(subset_wid_list),
        word_ct_U=np.hstack(subset_wct_list),
        doc_indptr_Dp1=np.hstack([0,np.cumsum(doclen_list)]),
        n_docs=len(doc_ids),
        n_vocabs=n_vocabs)
    subset['x_csr_DV'] = scipy.sparse.csr_matrix(
        (subset['word_ct_U'],
         subset['word_id_U'],
         subset['doc_indptr_Dp1']),
        shape=(subset['n_docs'], subset['n_vocabs']))
    if 'y_DC' in dataset:
        subset['y_DC'] = dataset['y_DC'][doc_ids]
        subset['n_labels'] = dataset['n_labels']
    if 'y_rowmask' in dataset:
        subset['y_rowmask'] = dataset['y_rowmask'][doc_ids]
    return subset

def load_x_csr(dataset_path=None, split_name='train'):
    ''' Load X array from disk

    Returns
    -------
    x_csr : sparse csr matrix
    '''
    csr_path = os.path.join(dataset_path, 'X_csr_%s.npz' % split_name)
    if os.path.exists(csr_path):
        Q = np.load(csr_path)
        x_csr_DV = scipy.sparse.csr_matrix(
            (Q['data'], Q['indices'], Q['indptr']),
            shape=Q['shape'])
    else:
        X_train = np.load(os.path.join(
            dataset_path, 'X_%s.npy' % split_name))
        x_csr_DV = scipy.sparse.csr_matrix(X_train)
    return x_csr_DV

def concat_bow_dataset(
        iterable_of_bow_datasets,
        **kwargs):
    x_DV_list = list()
    y_DC_list = list()
    for cur_data in iterable_of_bow_datasets:
        x_DV_list.append(cur_data['x_csr_DV'].toarray())
        y_DC_list.append(cur_data['y_DC'])
    y_DC = np.vstack(y_DC_list)
    x_DV = np.vstack(x_DV_list)
    x_csr_DV = scipy.sparse.csr_matrix(x_DV)
    return dict(
        n_docs=x_csr_DV.shape[0],
        n_vocabs=x_csr_DV.shape[1],
        word_id_U=x_csr_DV.indices,
        word_ct_U=x_csr_DV.data,
        doc_indptr_Dp1=x_csr_DV.indptr,
        x_csr_DV=x_csr_DV,
        y_DC=y_DC)

def describe_bow_dataset(
        dataset=None,
        dataset_name='Unknown dataset',
        percentiles=[0, 1, 10, 50, 90, 99, 100],
        label_list=None,
        **kwargs):
    n_docs, n_vocabs = dataset['x_csr_DV'].shape
    n_utokens_D = np.diff(dataset['doc_indptr_Dp1'])
    token_ct_D = np.squeeze(
        np.asarray(np.sum(dataset['x_csr_DV'], axis=1)))
    if dataset_name is not None:
        msg = "%s" % dataset_name
    else:
        msg = ""
    msg += "\n%d docs" % (n_docs)
    msg += "\n%d vocab words" % (n_vocabs)
    msg += "\nunique tokens per doc %s" % (
        make_percentile_str(n_utokens_D, percentiles, fmt_str='%7d'))
    msg += "\n total tokens per doc %s" % (
        make_percentile_str(token_ct_D, percentiles, fmt_str='%7d'))
    if 'y_DC' in dataset:
        msg += "\n%d labels" % dataset['n_labels']
        for c in xrange(dataset['n_labels']):
            y_c_D = dataset['y_DC'][:,c]
            try:
                rowmask = dataset['y_rowmask']
                y_c_D = y_c_D[rowmask == 1]
                assert np.all(np.isfinite(y_c_D))
            except KeyError:
                pass
            n_y_docs = y_c_D.size
            if c == 0:
                msg += "\n%.3f (%d/%d) docs are labeled" % (
                    float(n_y_docs)/n_docs, n_y_docs, n_docs)
            n_y_pos = np.sum(y_c_D == 1)
            if label_list:
                fmt_str = '%%-%ds' % (np.max(map(len, label_list)))
                label_c = (fmt_str + ' (%2d/%d)') % (
                    label_list[c], c+1, dataset['n_labels']) 
            else:
                label_c = 'label %2d/%d' % (c+1, dataset['n_labels']) 
            msg += '\n %s frac positive %.3f (%6d/%d)' % (
                label_c,
                float(n_y_pos) / float(n_y_docs), n_y_pos, n_y_docs)
            #msg += '\n frac negative %.3f (%6d/%d)' % (
            #    n_y_neg / float(n_y_docs), n_y_neg, n_y_docs)            
    return msg + "\n"


def make_dataset_from_x_DV(x_DV):
    ''' Create bow dataset from doc-term count array
    '''
    x_csr_DV = scipy.sparse.csr_matrix(x_DV)
    return dict(
        word_id_U=x_csr_DV.indices,
        word_ct_U=x_csr_DV.data,
        doc_indptr_Dp1=x_csr_DV.indptr,
        n_docs=x_csr_DV.shape[0],
        n_vocabs=x_csr_DV.shape[1],
        x_csr_DV=x_csr_DV)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',
        default=os.path.expandvars('$SSCAPEROOT/sscape/datasets/bow_pang_movie_reviews/v20170929_split_80_10_10/'),
        type=str)
    args = parser.parse_args()

    for split_name in ['train', 'valid', 'test']:
        data_info = load_dataset(
            dataset_path=args.dataset_path,
            split_name=split_name)
        print describe_bow_dataset(
            dataset=data_info['dataset'],
            label_list=data_info['label_list'])
