'''
Differentiable transform for any array whose elements lie in (0,1).

logistic_sigmoid(x) :  1 / (1 + exp(-x))

Examples
--------
>>> from autograd import elementwise_grad
>>> g_auto = elementwise_grad(
...     _logistic_sigmoid_not_vectorized)
>>> g_manual = elementwise_grad(logistic_sigmoid)

# Create grid of possible inputs
>>> vals = np.linspace(-5000., 5000., 100)

# Verify two funcs compute the same answers for all grid elements
>>> for x in vals: assert np.allclose(g_auto(x), g_manual(x))

# Can successfully call g_manual on array of values
>>> np.all(np.isfinite(g_manual(vals)))
True

# Cannot do so with autograd using not-vectorized function
>>> g_auto(vals)
Traceback (most recent call last):
...
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
'''

import autograd.numpy as np
from autograd.scipy.misc import logsumexp
from autograd.core import primitive
from autograd import elementwise_grad, grad

MIN_VAL=1e-200
MAX_VAL=1 - 1e-14

@primitive
def logistic_sigmoid(x_real):
    ''' Compute logistic sigmoid transform from real line to unit interval.

    Numerically stable and fully vectorized.

    Args
    ----
    x_real : array-like, with values in (-infty, +infty)

    Returns
    -------
    p_real : array-like, size of x_real, with values in (0, 1)

    Examples
    --------
    >>> logistic_sigmoid(-55555.)
    0.0
    >>> logistic_sigmoid(0.0)
    0.5
    >>> logistic_sigmoid(55555.)
    1.0
    >>> logistic_sigmoid(np.asarray([-999999, 0, 999999.]))
    array([ 0. ,  0.5,  1. ])
    '''
    if not isinstance(x_real, float):
        out = np.zeros_like(x_real)
        mask1 = x_real > 50.0
        out[mask1] = 1.0 / (1.0 + np.exp(-x_real[mask1]))
        mask0 = np.logical_not(mask1)
        out[mask0] = np.exp(x_real[mask0])
        out[mask0] /= (1.0 + out[mask0])
        return out
    if x_real > 50.0:
        pos_real = np.exp(-x_real)
        return 1.0 / (1.0 + pos_real)
    else:
        pos_real = np.exp(x_real)
        return pos_real / (1.0 + pos_real)

def _logistic_sigmoid_not_vectorized(x_real):
    if x_real > 50.0:
        pos_real = np.exp(-x_real)
        return 1.0 / (1.0 + pos_real)
    else:
        pos_real = np.exp(x_real)
        return pos_real / (1.0 + pos_real)


# Definite gradient function via manual formula
# Supporting different versions of autograd
if hasattr(primitive, 'defvjp'):
    def _vjp__logistic_sigmoid(ans, g, vs, gvs, x):
        x = np.asarray(x)
        return np.full(x.shape, g) * ans * (1.0 - ans)
    logistic_sigmoid.defvjp(_vjp__logistic_sigmoid)
else:
    def _make_grad_prod(ans,x):
        x = np.asarray(x)
        def gradient_product(g):
            return np.full(x.shape, g) * ans * (1-ans)
        return gradient_product
    logistic_sigmoid.defgrad(_make_grad_prod)


def inv_logistic_sigmoid(
        p, do_force_safe=True):
    ''' Compute inverse logistic sigmoid from unit interval to reals.

    Numerically stable and fully vectorized.

    Args
    ----
    p : array-like, with values in (0, 1)

    Returns
    -------
    x : array-like, size of p, with values in (-infty, infty)

    Examples
    --------
    >>> np.round(inv_logistic_sigmoid(0.11), 6)
    -2.090741
    >>> np.round(inv_logistic_sigmoid(0.5), 6)
    0.0
    >>> np.round(inv_logistic_sigmoid(0.89), 6)
    2.090741

    >>> p_vec = np.asarray([
    ...     1e-100, 1e-10, 1e-5,
    ...     0.25, 0.75, .9999, 1-1e-14])
    >>> np.round(inv_logistic_sigmoid(p_vec), 2)
    array([-230.26,  -23.03,  -11.51,   -1.1 ,    1.1 ,    9.21,   32.24])
    '''
    if do_force_safe:
        p = np.minimum(np.maximum(p, MIN_VAL), MAX_VAL)
    return np.log(p) - np.log1p(-p)

def to_safe_common_arr(p):
    p = np.minimum(np.maximum(p, MIN_VAL), MAX_VAL)
    return p    

to_common_arr = logistic_sigmoid
to_diffable_arr = inv_logistic_sigmoid