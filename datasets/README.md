Quick Links:

* [Example datasets](#example-datasets)
* [In-memory format](#in-memory-format)
* [On-disk format](#on-disk-format)

# Background: Datasets for supervised bag-of-words tasks

We consider supervised bag-of-words tasks, where we have as observed data many examples (aka 'documents'), indexed by 'd', which consist of pairs $x_d, y_d$, where:

* x_d represents the input count data
* y_d represents some target outcome labels of interest (binary movie rating, real-valued document score, etc)

In Python, we could represent these values using Numpy arrays:
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

Dataset size variables and abbreviations:
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
    number of non-zero (doc_id, vocab_id) pairs in sparse matrix
```


# Example datasets

This repo comes with two example datasets, provided in our standard [on-disk format](#on-disk-format):

* [toy_bars_3x3/](https://github.com/dtak/prediction-constrained-topic-models/tree/master/datasets/toy_bars_3x3/)

> Small toy dataset of 9 vocab words arranged in 3x3 grid. Useful for visualing inspecting learned topic structure, which look like bars on the 3x3 grid.

* [movie_reviews_pang_lee/](https://github.com/dtak/prediction-constrained-topic-models/tree/master/datasets/movie_reviews_pang_lee/)

> Dataset of movie reviews, where prediction task is take a careful bag-of-words representation of plain-text reviews from professional critics, and predict a binary label of movie quality (1 = movie received more than 2-out-of-4 stars, 0 = otherwise). Originally from Pang & Lee ACL 2005.


# In-memory format

For PC toolbox code, we represent one entire dataset (e.g. the train set or the test set) as one **Python dictionary** ('dict') object.

This dictionary has at least the following key,value entries:
```
* x_csr_DV : 2D scipy.sparse.csr_matrix, shape D x V (n_docs x n_vocabs)
    Each row is sparse representation of x_d's count data.

* y_DC : 2D numpy array, shape D x C (n_docs x n_labels)
    Each row gives outcomes for doc d

* n_docs : int
    Total number of documents in this dataset
    
* n_vocabs : int
    Total number of possible vocabulary words in this dataset.
```

## Python code for saving/loading

A dataset's dictionary representation can be loaded/saved to disk via some useful functions defined in [`$PC_REPO_DIR/pc_toolbox/model_slda/slda_utils__dataset_manager.py`](https://github.com/dtak/prediction-constrained-topic-models/tree/master/pc_toolbox/model_slda/slda_utils__dataset_manager.py)

Example usage:
```
>>> from slda_utils__dataset_manager import load_dataset
>>> tr_dataset = load_dataset("$PC_REPO_DIR/datasets/toy_bars_3x3/", split_name='train')

# Show y labels for first 5 documents
>>> tr_dataset['y_DC'][:5]

# Show dense array repr of x data of first 5 documents
>>> tr_dataset['x_csr_DV'][:5].toarray()

```


# On-disk format

Each dataset is located in its own folder on disk, such as [datasets/movie_reviews_pang_lee/](https://github.com/dtak/prediction-constrained-topic-models/tree/master/datasets/movie_reviews_pang_lee)

Inside the folder, the dataset is represented by several files contents that must match the following file names:

```
* X_colnames.txt : utf-8 formatted text file
    V lines (one line per vocab term)
    Each line contains the string name of its corresponding vocab term

* Y_colnames.txt : utf-8 formatted text file
    C lines (one line per outcome)
    Each line contains the string name of its corresponding outcome

* X_csr_train.npz : npz file for a scipy.sparse.csr_matrix
* X_csr_valid.npz : npz file for a scipy.sparse.csr_matrix
* X_csr_test.npz : npz file for a scipy.sparse.csr_matrix

* Y_train.npy : npy file
* Y_valid.npy : npy file
* Y_test.npy : npy file
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
>>> import scipy.sparse
>>> x_arr = np.eye(10)
>>> x_csr = scipy.sparse.csr_matrix(x_arr)
>>> npz_dict = dict(
...     shape=x_csr.shape,
...     data=x_csr.data,
...     indices=x_csr.indices,
...     indptr=x_csr.indptr)
```
## Y_split.npy Disk Format : .npy file

The [.npy file format](https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html) is a standard provided by numpy for saving/loading single arrays.

We save the y outcomes from each dataset split (train/valid/test) as a single .npy file.



## TODO describe how missing values work
