import argparse
import os
import numpy as np
import scipy.sparse
from distutils.dir_util import mkpath
from sklearn.externals import joblib

vocab_list = np.asarray([
    ['needle', 'finance',    'tech'],
    ['river',     'bank',  'stream'],
    ['mineral',   'gold', 'silicon'],
    ]).flatten().tolist()

tA = np.asarray([
    [.00, .00, .00],
    [.16, .16, .16],
    [.16, .16, .16],
    ])
tB = np.asarray([
    [.00, .16, .16],
    [.00, .16, .16],
    [.00, .16, .16],
    ])
tC = np.asarray([
    [.00, .00, .00],
    [.33, .33, .33],
    [.00, .00, .00],
    ])
tD = np.asarray([
    [.00, .00, .00],
    [.00, .00, .00],
    [.33, .33, .33],
    ])
tE = np.asarray([
    [.00, .33, .00],
    [.00, .33, .00],
    [.00, .33, .00],
    ])
tF = np.asarray([
    [.00, .00, .33],
    [.00, .00, .33],
    [.00, .00, .33],
    ])
tG = np.asarray([
    [.00, .33, .00],
    [.33, .33, .33],
    [.00, .33, .00],
    ])
tH = np.asarray([
    [.00, .00, .33],
    [.00, .00, .33],
    [.33, .33, .33],
    ])
proba_list = [.38, .38, .08, .08, .02, .02, .02, .02]
topic_list = [tA, tB, tC, tD, tE, tF, tG, tH]
for t in topic_list:
    t /= t.sum()


def draw_random_doc(
        topic_list,
        proba_list,
        min_n_words_per_doc=45,
        max_n_words_per_doc=60,
        do_return_square=True,
        proba_positive_label=0.2,
        d=0):
    prng = np.random.RandomState(d)
    V = topic_list[0].size

    # Pick which template
    k = prng.choice(len(proba_list), p=proba_list)
    n_words = prng.randint(low=min_n_words_per_doc, high=max_n_words_per_doc)
    words = prng.choice(
        V,
        p=topic_list[k].flatten(),
        replace=True,
        size=n_words)
    x_V = np.bincount(words, minlength=V)
    if prng.rand() < proba_positive_label:
        y_C = 1.0
        x_V[0] += 1
    else:
        y_C = 0.0
    return x_V, y_C


def save_csr_matrix(filename, array):
    np.savez(
        filename,
        data=array.data,
        indices=array.indices,
        indptr=array.indptr,
        shape=array.shape)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default=os.path.abspath('.'), type=str)
    parser.add_argument("--n_docs_train", default=500, type=int)
    parser.add_argument("--n_docs_test", default=500, type=int)
    parser.add_argument("--n_docs_valid", default=500, type=int)

    args = parser.parse_args()
    dataset_path = os.path.abspath(args.dataset_path)

    x_list = list()
    y_list = list()
    n_docs = args.n_docs_train + args.n_docs_valid + args.n_docs_test
    for d in range(n_docs):
        x_V, y_C = draw_random_doc(
            topic_list,
            proba_list,
            do_return_square=False,
            d=d,
            )
        x_list.append(x_V)
        y_list.append(y_C)
        if (d+1) % 100 == 0 or (d == n_docs -1):
            print "generated doc %d/%d" % (d+1, n_docs)

    # stack into array format
    x_DV = np.vstack(x_list)
    x_csr_DV = scipy.sparse.csr_matrix(x_DV)
    y_DC = np.vstack(y_list)
    if y_DC.ndim == 1:
        y_DC = y_DC[:,np.newaxis]

    train_doc_ids = np.arange(args.n_docs_train)
    valid_doc_ids = np.arange(
        args.n_docs_train,
        args.n_docs_train + args.n_docs_valid)
    test_doc_ids = np.arange(
        args.n_docs_train + args.n_docs_valid,
        x_DV.shape[0])

    np.save(os.path.join(dataset_path, "Y_train.npy"), y_DC[train_doc_ids])
    np.save(os.path.join(dataset_path, "Y_valid.npy"), y_DC[valid_doc_ids])
    np.save(os.path.join(dataset_path, "Y_test.npy"), y_DC[test_doc_ids])

    save_csr_matrix(os.path.join(dataset_path, "X_csr_train.npz"), x_csr_DV[train_doc_ids])
    save_csr_matrix(os.path.join(dataset_path, "X_csr_valid.npz"), x_csr_DV[valid_doc_ids])
    save_csr_matrix(os.path.join(dataset_path, "X_csr_test.npz"), x_csr_DV[test_doc_ids])


    # Write necessary txt files
    V = x_DV.shape[1]
    with open(os.path.join(dataset_path, 'X_colnames.txt'), 'w') as f:
        for vocab_term in vocab_list:
            f.write('%s\n' % vocab_term)
    with open(os.path.join(dataset_path, 'Y_colnames.txt'), 'w') as f:
        f.write('has_needle\n')
