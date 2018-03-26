# List of Provided Datasets

* [toy_bars_3x3/](https://github.com/dtak/prediction-constrained-topic-models/tree/master/datasets/toy_bars_3x3/)

Toy dataset first described in Hughes et al. AISTATS 2018.

* [movie_reviews_pang_lee/](https://github.com/dtak/prediction-constrained-topic-models/tree/master/datasets/movie_reviews_pang_lee/)

Dataset of movie reviews, where prediction task is take a careful bag-of-words representation of plain-text reviews from professional critics, and predict a binary label of movie quality (1 = movie received more than 2-out-of-4 stars, 0 = otherwise).


# Dataset format for supervised topic modeling

In this task, we have for each example (aka document, indexed by 'd') some input data x_d, and some outcome labels y_d:
```
* x_d_V : 1D array, size V
    bag-of-words count vector
    x_d_V[v] is a non-negative integer in {0, 1, 2, ...}

* y_d_C : 1D array, size C
    outcome vector
    y_d_C[c] is a scalar
    If all binary, this is a multivariate-outcome binary classification task
    If all real-valued, this is a multivariate-outcome regression task.
```

Dataset size variables:
```
* D : int 
    n_docs
    number of documents in current data subset
* V : int
    n_vocabs
    number of vocabulary words
* C : int
    n_labels
    number of outcome
* U : int
    n_unique_tokens
    number of non-zero (doc_id, term_id) pairs in sparse matrix
```

## Overview

Python code for saving/loading labeled datasets for topic modeling can be found in:

[`$PC_REPO_DIR/pc_toolbox/model_slda/slda_utils__dataset_manager.py`](https://github.com/dtak/prediction-constrained-topic-models/tree/master/pc_toolbox/model_slda/slda_utils__dataset_manager.py)

Example usage:
```
>>> from slda_utils__dataset_manager import load_dataset
>>> tr_dataset = load_dataset("$PC_REPO_DIR/datasets/toy_bars_3x3", split_name='train')

# Show y labels for first 5 documents
>>> tr_dataset['y_DC'][:5]

# Show dense array repr of x data of first 5 documents
>>> tr_dataset['x_csr_DV'][:5].toarray()

```


## In-memory format

We represent datasets as dictionary objections, with keys:
```
* x_csr_DV : 2D scipy.csr_matrix, shape D x V (n_docs x n_vocabs)
    Each row is sparse representation of x_d's count data.

* y_DC : 2D numpy array, shape D x C (n_docs x n_labels)
    Each row gives outcomes for doc d

* n_docs : int
    Total number of documents in this dataset
```

## X_csr Disk Format : .npz file

The [.npz file format](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html) is a standard numpy way to save/load multiple related arrays to/from a single file.

To make on-disk storage compact, we store the csr_matrix formated X for a single dataset split (train/valid/test) as a single file named "X_csr_train.npz".

```
X_csr_$SPLIT.npz : .npz file contains
    shape : 1D array, shape 2
        Encodes shape of the array (n_docs, n_vocabs)
    data : 1D array, shape U
        Contains count values for *all* non-zero (doc_id, vocab_id) entries.
    indices : 1D array, shape U
        Contains vocab ids for *all* non-zero (doc_id, vocab_id) entries.
    indptr : 1D array, shape D+1
        Defines (start,stop) slices for each document within data and indices
```

To obtain the relevant arrays for a given dense array, just do:
```
>>> x_arr = np.eye(10)
>>> x_csr = scipy.csr_matrix(x_arr)
>>> npz_dict = dict(
...     shape=x_csr.shape,
...     data=x_csr.data,
...     indices=x_csr.indices,
...     indptr=x_csr.indptr)
```
## Y_split.npy Disk Format : .npu file

The [.npy file format](https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html) is a standard provided by numpy for saving/loading single arrays.

We save the y outcomes from each dataset split (train/valid/test) as a single .npy file.


# Dataset format for binary classification

TODO