import numpy as np

def toCArray(X, dtype=np.float64):
    """ Convert input into numpy array of C-contiguous order.

    Ensures returned array is aligned and owns its own data,
    not a view of another array.

    Returns
    -------
    X : ND array

    Examples
    -------
    >>> Q = np.zeros(10, dtype=np.int32, order='F')
    >>> toCArray(Q).flags.c_contiguous
    True
    >>> toCArray(Q).dtype.byteorder
    '='
    """
    X = np.asarray_chkfinite(X, dtype=dtype, order='C')
    if X.dtype.byteorder != '=':
        X = X.newbyteorder('=').copy()
    if not X.flags.owndata or X.flags.aligned:
        X = X.copy()
    assert X.flags.owndata
    assert X.flags.aligned
    return X

def as1D(x):
    """ Convert input into to 1D numpy array.

    Returns
    -------
    x : 1D array

    Examples
    -------
    >>> as1D(5)
    array([5])
    >>> as1D([1,2,3])
    array([1, 2, 3])
    >>> as1D([[3,4,5,6]])
    array([3, 4, 5, 6])
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray_chkfinite(x)
    if x.ndim < 1:
        x = np.asarray_chkfinite([x])
    elif x.ndim > 1:
        x = np.squeeze(x)
    return x


def as2D(x):
    """ Convert input into to 2D numpy array.


    Returns
    -------
    x : 2D array

    Examples
    -------
    >>> as2D(5)
    array([[5]])
    >>> as2D([1,2,3])
    array([[1, 2, 3]])
    >>> as2D([[3,4,5,6]])
    array([[3, 4, 5, 6]])
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray_chkfinite(x)
    if x.ndim < 1:
        x = np.asarray_chkfinite([x])
    while x.ndim < 2:
        x = x[np.newaxis, :]
    return x


def as3D(x):
    """ Convert input into to 3D numpy array.

    Returns
    -------
    x : 3D array

    Examples
    -------
    >>> as3D(5)
    array([[[5]]])
    >>> as3D([1,2,3])
    array([[[1, 2, 3]]])
    >>> as3D([[3,4,5,6]])
    array([[[3, 4, 5, 6]]])
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray_chkfinite(x)
    if x.ndim < 1:
        x = np.asarray_chkfinite([x])
    while x.ndim < 3:
        x = x[np.newaxis, :]
    return x

